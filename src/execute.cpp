/**
 * @file execute.cpp
 * @brief Depth-first join tree execution engine.
 *
 * Traverses plan tree: resolve inputs -> select build/probe -> algorithm
 * selection -> match collection -> output construction.
 *
 * Flow: execute() -> execute_impl() recursively -> resolve_join_input() for
 * ScanNode (ColumnarTable*) or JoinNode (ExecuteResult). Root produces
 * ColumnarTable; non-root produces ExecuteResult.
 *
 * Lifetimes: base tables live for query duration; ExecuteResult held on stack
 * until parent completes; VARCHAR refs valid via base table lifetime.
 *
 * Row order non-deterministic (work-stealing); semantically correct per SQL.
 *
 * @see plan.h, match_collector.h, materialize.h, construct_intermediate.h
 */
#include <foundation/attribute.h>
#if defined(__APPLE__) && defined(__aarch64__)
#include <platform/hardware_darwin.h>
#elif defined(SPC__USE_BENCHMARKVM_HARDWARE)
#include <platform/hardware_benchmarkvm.h>
#else
#include <platform/hardware.h>
#endif

#include <chrono>
#include <data_access/columnar_reader.h>
#include <data_model/intermediate.h>
#include <iostream>
#include <join_execution/hash_join.h>
#include <join_execution/hashtable.h>
#include <join_execution/join_setup.h>
#include <join_execution/nested_loop.h>
#include <materialization/construct_intermediate.h>
#include <materialization/materialize.h>
#include <optional>
#include <platform/arena.h>
#include <platform/worker_pool.h>
#include <variant>

#ifdef USE_DEFERRED_MATERIALIZATION
#include <data_model/deferred_intermediate.h>
#include <data_model/deferred_plan.h>
#include <materialization/construct_deferred.h>
#include <materialization/materialize_deferred.h>
#endif

namespace Contest {

using namespace join;

using materialize::construct_intermediate_from_buffers;
using materialize::create_empty_result;
using materialize::materialize_from_buffers;

/**
 * @brief Result variant: ExtendedResult (intermediate, with row ID tracking) or
 * ColumnarTable (final output per contest API).
 */
using JoinResult = std::variant<ExtendedResult, ColumnarTable>;

/**
 * @brief Recursive join execution with timing.
 * @param plan Query plan with nodes and base tables.
 * @param node_idx Current node index in plan.nodes.
 * @param is_root True -> ColumnarTable output; false -> ExecuteResult.
 * @param stats Timing accumulator.
 * @return JoinResult (intermediate or final).
 */
JoinResult execute_impl(const Plan &plan, size_t node_idx, bool is_root,
                        TimingStats &stats);

/**
 * @brief Resolve plan node to JoinInput.
 *
 * ScanNode -> non-owning ColumnarTable*; JoinNode -> recursive execution
 * returning owned ExtendedResult. Implements depth-first traversal.
 *
 * @param plan Query plan.
 * @param node_idx Node index to resolve.
 * @param stats Timing accumulator.
 * @return JoinInput with data variant and metadata.
 */
JoinInput resolve_join_input(const Plan &plan, size_t node_idx,
                             TimingStats &stats) {
    JoinInput input;
    const auto &node = plan.nodes[node_idx];
    input.node = &node;

    if (const auto *scan = std::get_if<ScanNode>(&node.data)) {
        input.data = &plan.inputs[scan->base_table_id];
        input.table_id = scan->base_table_id;
    } else {
        auto result = execute_impl(plan, node_idx, false, stats);
        input.data = std::get<ExtendedResult>(std::move(result));
        input.table_id = 0;
    }
    return input;
}

/**
 * @brief Unified probe + materialize helper templated on collection mode.
 *
 * Executes probe (nested loop or hash join) and materialization/intermediate
 * construction in a single function. Template parameter eliminates runtime
 * branching in hot loops.
 *
 * @tparam Mode Collection mode (BOTH, LEFT_ONLY, RIGHT_ONLY).
 */
template <MatchCollectionMode Mode>
JoinResult execute_join_with_mode(
    bool use_nested_loop, bool probe_is_columnar, bool is_root,
    const UnchainedHashtable *hash_table, const JoinInput &build_input,
    const JoinInput &probe_input, const BuildProbeConfig &config,
    const PlanNode &build_node, const PlanNode &probe_node, JoinSetup &setup,
    io::ColumnarReader &columnar_reader, const Plan &plan, TimingStats &stats) {

    std::vector<ThreadLocalMatchBuffer<Mode>> match_buffers;

    if (use_nested_loop) {
        auto nested_loop_start = std::chrono::high_resolution_clock::now();
        match_buffers = nested_loop_join<Mode>(
            build_input, probe_input, config.build_attr, config.probe_attr);
        auto nested_loop_end = std::chrono::high_resolution_clock::now();
        stats.nested_loop_join_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                nested_loop_end - nested_loop_start)
                .count();
    } else {
        auto probe_start = std::chrono::high_resolution_clock::now();
        if (probe_is_columnar) {
            match_buffers = probe_columnar<Mode>(*hash_table, probe_input,
                                                 config.probe_attr);
        } else {
            const auto &probe_result =
                std::get<ExtendedResult>(probe_input.data);
            match_buffers = probe_intermediate<Mode>(
                *hash_table, probe_result.columns[config.probe_attr]);
        }
        auto probe_end = std::chrono::high_resolution_clock::now();
        stats.hash_join_probe_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(probe_end -
                                                                  probe_start)
                .count();
    }

    size_t total_matches = 0;
    for (const auto &buf : match_buffers) {
        total_matches += buf.count();
    }

    if (is_root) {
        auto mat_start = std::chrono::high_resolution_clock::now();
        JoinResult final_result;
        if (total_matches == 0) {
            final_result = create_empty_result(config.remapped_attrs);
        } else {
            prepare_output_columns(
                columnar_reader, build_input, probe_input, build_node,
                probe_node, config.remapped_attrs, build_input.output_size());

            final_result = materialize_from_buffers<Mode>(
                match_buffers, build_input, probe_input, config.remapped_attrs,
                build_node, probe_node, build_input.output_size(),
                columnar_reader, plan);
        }
        auto mat_end = std::chrono::high_resolution_clock::now();
        stats.materialize_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(mat_end -
                                                                  mat_start)
                .count();
        return final_result;
    } else {
        auto inter_start = std::chrono::high_resolution_clock::now();
        if (total_matches > 0) {
            prepare_output_columns(
                columnar_reader, build_input, probe_input, build_node,
                probe_node, config.remapped_attrs, build_input.output_size());

            construct_intermediate_from_buffers<Mode>(
                match_buffers, build_input, probe_input, config.remapped_attrs,
                build_node, probe_node, build_input.output_size(),
                columnar_reader, setup.results, setup.merged_table_ids);
        }
        auto inter_end = std::chrono::high_resolution_clock::now();
        stats.intermediate_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(inter_end -
                                                                  inter_start)
                .count();
        return std::move(setup.results);
    }
}

/**
 * @brief Core recursive join execution.
 *
 * Phases: resolve L/R inputs -> select build/probe (smaller=build) -> algorithm
 * choice -> build/probe -> output construction.
 *
 * Algorithm: nested loop if build_rows < HASH_TABLE_THRESHOLD (8); else radix-
 * partitioned hash join.
 *
 * Memory: hash table and MatchCollector local (freed on return); child
 * ExecuteResults on stack until materialization; setup.results pre-allocated.
 */
JoinResult execute_impl(const Plan &plan, size_t node_idx, bool is_root,
                        TimingStats &stats) {
    auto &node = plan.nodes[node_idx];

    if (!std::holds_alternative<JoinNode>(node.data)) {
        return ExtendedResult{};
    }

    const auto &join = std::get<JoinNode>(node.data);
    const auto &output_attrs = node.output_attrs;
    const auto &left_node = plan.nodes[join.left];
    const auto &right_node = plan.nodes[join.right];

    JoinInput left_input = resolve_join_input(plan, join.left, stats);
    JoinInput right_input = resolve_join_input(plan, join.right, stats);

    /* Build/probe selection: smaller input = build side; remaps output_attrs.
     */
    auto setup_start = std::chrono::high_resolution_clock::now();
    auto config =
        select_build_probe_side(join, left_input, right_input, output_attrs);
    const JoinInput &build_input = config.build_left ? left_input : right_input;
    const JoinInput &probe_input = config.build_left ? right_input : left_input;
    const auto &build_node = config.build_left ? left_node : right_node;
    const auto &probe_node = config.build_left ? right_node : left_node;

    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();

    /* Nested loop for <8 rows (L1-resident, no hash overhead, SIMD). */
    const size_t HASH_TABLE_THRESHOLD = 8;
    size_t build_rows = build_input.row_count(config.build_attr);
    bool use_nested_loop = (build_rows < HASH_TABLE_THRESHOLD);

    /* Pre-allocate ExecuteResult; ColumnarReader PageIndex built lazily. */
    JoinSetup setup = setup_join(build_input, probe_input, build_node,
                                 probe_node, left_node, right_node, left_input,
                                 right_input, output_attrs, build_rows);
    auto setup_end = std::chrono::high_resolution_clock::now();
    auto setup_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        setup_end - setup_start);
    stats.setup_ms += setup_elapsed.count();

    /* Skip unused-side row IDs if output needs only one side (50% savings). */
    MatchCollectionMode collection_mode = determine_collection_mode(
        config.remapped_attrs, config.build_left ? left_input.output_size()
                                                 : right_input.output_size());

    /* Build hash table if needed (before mode dispatch). */
    std::optional<UnchainedHashtable> hash_table;
    if (!use_nested_loop) {
        auto build_start = std::chrono::high_resolution_clock::now();
        hash_table =
            build_is_columnar
                ? build_from_columnar(build_input, config.build_attr)
                : build_from_intermediate(build_input, config.build_attr);
        auto build_end = std::chrono::high_resolution_clock::now();
        stats.hashtable_build_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(build_end -
                                                                  build_start)
                .count();
    }

    /* Dispatch based on collection mode - single runtime branch, then
     * fully specialized template instantiation with zero branching in hot
     * loops. */
    switch (collection_mode) {
    case MatchCollectionMode::BOTH:
        return execute_join_with_mode<MatchCollectionMode::BOTH>(
            use_nested_loop, probe_is_columnar, is_root,
            use_nested_loop ? nullptr : &(*hash_table), build_input,
            probe_input, config, build_node, probe_node, setup,
            setup.columnar_reader, plan, stats);

    case MatchCollectionMode::LEFT_ONLY:
        return execute_join_with_mode<MatchCollectionMode::LEFT_ONLY>(
            use_nested_loop, probe_is_columnar, is_root,
            use_nested_loop ? nullptr : &(*hash_table), build_input,
            probe_input, config, build_node, probe_node, setup,
            setup.columnar_reader, plan, stats);

    case MatchCollectionMode::RIGHT_ONLY:
        return execute_join_with_mode<MatchCollectionMode::RIGHT_ONLY>(
            use_nested_loop, probe_is_columnar, is_root,
            use_nested_loop ? nullptr : &(*hash_table), build_input,
            probe_input, config, build_node, probe_node, setup,
            setup.columnar_reader, plan, stats);
    }

    // Should never reach here, but satisfy compiler
    return ExtendedResult{};
}

#ifdef USE_DEFERRED_MATERIALIZATION
// ============================================================================
// DEFERRED MATERIALIZATION PATH
// ============================================================================

using DeferredJoinResult = std::variant<DeferredResult, ColumnarTable>;

using materialize::construct_deferred_from_buffers;
using materialize::create_empty_deferred_result;
using materialize::materialize_deferred_from_buffers;

// Forward declaration
DeferredJoinResult execute_deferred_impl(const DeferredPlan &deferred_plan,
                                         size_t node_idx, bool is_root,
                                         TimingStats &stats);

/**
 * @brief Resolve deferred plan node to DeferredInput.
 */
DeferredInput resolve_deferred_input(const DeferredPlan &deferred_plan,
                                     size_t node_idx, TimingStats &stats) {
    DeferredInput input;
    const auto &dnode = deferred_plan[node_idx];
    const auto &pnode = deferred_plan.original_plan->nodes[node_idx];
    input.node = &pnode;
    input.deferred_node = &dnode;

    if (const auto *dscan = std::get_if<DeferredScanNode>(&dnode)) {
        input.data = &deferred_plan.original_plan->inputs[dscan->base_table_id];
        input.table_id = dscan->base_table_id;
    } else {
        auto result =
            execute_deferred_impl(deferred_plan, node_idx, false, stats);
        input.data = std::get<DeferredResult>(std::move(result));
        input.table_id = 0;
    }
    return input;
}

/**
 * @brief Select build/probe sides for deferred input.
 */
BuildProbeConfig select_deferred_build_probe_side(
    const JoinNode &join, const DeferredInput &left_input,
    const DeferredInput &right_input,
    const std::vector<std::tuple<size_t, DataType>> &output_attrs) {
    BuildProbeConfig config;

    size_t left_rows = left_input.row_count(join.left_attr);
    size_t right_rows = right_input.row_count(join.right_attr);
    config.build_left = left_rows <= right_rows;

    config.build_attr = config.build_left ? join.left_attr : join.right_attr;
    config.probe_attr = config.build_left ? join.right_attr : join.left_attr;

    config.remapped_attrs = output_attrs;
    size_t left_size = left_input.output_size();
    size_t build_size =
        config.build_left ? left_size : right_input.output_size();

    if (!config.build_left) {
        for (auto &[col_idx, dtype] : config.remapped_attrs) {
            if (col_idx < left_size) {
                col_idx = build_size + col_idx;
            } else {
                col_idx = col_idx - left_size;
            }
        }
    }
    return config;
}

/**
 * @brief Unified probe + materialize for deferred path.
 */
template <MatchCollectionMode Mode>
DeferredJoinResult execute_deferred_join_with_mode(
    bool use_nested_loop, bool probe_is_columnar, bool is_root,
    const UnchainedHashtable *hash_table, const DeferredInput &build_input,
    const DeferredInput &probe_input, const BuildProbeConfig &config,
    const DeferredJoinNode &join_node, io::ColumnarReader &columnar_reader,
    const DeferredPlan &deferred_plan, TimingStats &stats) {

    std::vector<ThreadLocalMatchBuffer<Mode>> match_buffers;

    if (use_nested_loop) {
        auto nested_loop_start = std::chrono::high_resolution_clock::now();
        match_buffers = nested_loop_join_deferred<Mode>(
            build_input, probe_input, config.build_attr, config.probe_attr);
        auto nested_loop_end = std::chrono::high_resolution_clock::now();
        stats.nested_loop_join_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(
                nested_loop_end - nested_loop_start)
                .count();
    } else {
        auto probe_start = std::chrono::high_resolution_clock::now();
        if (probe_is_columnar) {
            // Create JoinInput for columnar probe
            JoinInput probe_ji;
            probe_ji.node = probe_input.node;
            probe_ji.data = std::get<const ColumnarTable *>(probe_input.data);
            probe_ji.table_id = probe_input.table_id;
            match_buffers =
                probe_columnar<Mode>(*hash_table, probe_ji, config.probe_attr);
        } else {
            const auto &probe_result =
                std::get<DeferredResult>(probe_input.data);
            // Probe using materialized column (should be the join key)
            const auto *mat_col =
                probe_result.get_materialized(config.probe_attr);
            if (!mat_col) {
                std::fprintf(
                    stderr,
                    "ERROR: probe join key not materialized! probe_attr=%zu "
                    "mat_map_size=%zu num_rows=%zu\n",
                    config.probe_attr, probe_result.materialized_map.size(),
                    probe_result.num_rows);
                std::abort();
            }
            match_buffers = probe_intermediate<Mode>(*hash_table, *mat_col);
        }
        auto probe_end = std::chrono::high_resolution_clock::now();
        stats.hash_join_probe_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(probe_end -
                                                                  probe_start)
                .count();
    }

    size_t total_matches = 0;
    for (const auto &buf : match_buffers) {
        total_matches += buf.count();
    }

    if (is_root) {
        auto mat_start = std::chrono::high_resolution_clock::now();
        DeferredJoinResult final_result;
        if (total_matches == 0) {
            final_result =
                materialize::create_empty_deferred_final(config.remapped_attrs);
        } else {
            // Prepare page indices for final materialization
            materialize::prepare_final_deferred_columns(
                columnar_reader, build_input, probe_input, join_node,
                config.remapped_attrs, build_input.output_size(),
                config.build_left);

            final_result = materialize_deferred_from_buffers<Mode>(
                match_buffers, build_input, probe_input, join_node,
                config.remapped_attrs, build_input.output_size(),
                config.build_left, columnar_reader, deferred_plan);
        }
        auto mat_end = std::chrono::high_resolution_clock::now();
        stats.materialize_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(mat_end -
                                                                  mat_start)
                .count();
        return final_result;
    } else {
        auto inter_start = std::chrono::high_resolution_clock::now();
        DeferredResult result;
        if (total_matches > 0) {
            // Prepare page indices for intermediate construction
            materialize::prepare_deferred_columns(
                columnar_reader, build_input, probe_input, join_node,
                config.remapped_attrs, build_input.output_size(),
                config.build_left);

            construct_deferred_from_buffers<Mode>(
                match_buffers, build_input, probe_input, join_node,
                config.remapped_attrs, build_input.output_size(),
                config.build_left, columnar_reader, result, deferred_plan);
        } else {
            result = create_empty_deferred_result(join_node);
        }
        auto inter_end = std::chrono::high_resolution_clock::now();
        stats.intermediate_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(inter_end -
                                                                  inter_start)
                .count();
        return std::move(result);
    }
}

/**
 * @brief Recursive deferred join execution.
 */
DeferredJoinResult execute_deferred_impl(const DeferredPlan &deferred_plan,
                                         size_t node_idx, bool is_root,
                                         TimingStats &stats) {
    const auto &dnode = deferred_plan[node_idx];

    if (std::holds_alternative<DeferredScanNode>(dnode)) {
        return DeferredResult{};
    }

    const auto &djoin = std::get<DeferredJoinNode>(dnode);
    const auto &plan = *deferred_plan.original_plan;
    const auto &pnode = plan.nodes[node_idx];
    const auto &join = std::get<JoinNode>(pnode.data);

    // Resolve inputs
    DeferredInput left_input =
        resolve_deferred_input(deferred_plan, djoin.left_child_idx, stats);
    DeferredInput right_input =
        resolve_deferred_input(deferred_plan, djoin.right_child_idx, stats);

    // Build/probe selection
    auto setup_start = std::chrono::high_resolution_clock::now();
    auto config = select_deferred_build_probe_side(
        join, left_input, right_input, djoin.output_attrs);
    const DeferredInput &build_input =
        config.build_left ? left_input : right_input;
    const DeferredInput &probe_input =
        config.build_left ? right_input : left_input;

    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();

    const size_t HASH_TABLE_THRESHOLD = 8;
    size_t build_rows = build_input.row_count(config.build_attr);
    // Use nested loop for small build tables - works with both columnar and
    // DeferredResult inputs (join keys are always materialized).
    bool use_nested_loop = (build_rows < HASH_TABLE_THRESHOLD);

    io::ColumnarReader columnar_reader;
    auto setup_end = std::chrono::high_resolution_clock::now();
    stats.setup_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
                          setup_end - setup_start)
                          .count();

    // Use pre-computed collection mode from plan analysis.
    // base_collection_mode assumes build=left; flip if build=right at runtime.
    MatchCollectionMode mode = djoin.base_collection_mode;
    if (!config.build_left) {
        if (mode == MatchCollectionMode::LEFT_ONLY)
            mode = MatchCollectionMode::RIGHT_ONLY;
        else if (mode == MatchCollectionMode::RIGHT_ONLY)
            mode = MatchCollectionMode::LEFT_ONLY;
    }

    // Build hash table if needed
    std::optional<UnchainedHashtable> hash_table;
    if (!use_nested_loop) {
        auto build_start = std::chrono::high_resolution_clock::now();
        if (build_is_columnar) {
            JoinInput build_ji;
            build_ji.node = build_input.node;
            build_ji.data = std::get<const ColumnarTable *>(build_input.data);
            build_ji.table_id = build_input.table_id;
            hash_table = build_from_columnar(build_ji, config.build_attr);
        } else {
            const auto &dr = std::get<DeferredResult>(build_input.data);
            const auto *mat_col = dr.get_materialized(config.build_attr);
            if (!mat_col) {
                std::fprintf(
                    stderr,
                    "ERROR: build join key not materialized! build_attr=%zu "
                    "mat_map_size=%zu num_rows=%zu\n",
                    config.build_attr, dr.materialized_map.size(), dr.num_rows);
                // Fatal - this should never happen
                std::abort();
            }
            hash_table.emplace(mat_col->row_count());
            hash_table->build_intermediate(*mat_col);
        }
        auto build_end = std::chrono::high_resolution_clock::now();
        stats.hashtable_build_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(build_end -
                                                                  build_start)
                .count();
    }

    // Dispatch based on collection mode
    switch (mode) {
    case MatchCollectionMode::BOTH:
        return execute_deferred_join_with_mode<MatchCollectionMode::BOTH>(
            use_nested_loop, probe_is_columnar, is_root,
            use_nested_loop ? nullptr : &(*hash_table), build_input,
            probe_input, config, djoin, columnar_reader, deferred_plan, stats);

    case MatchCollectionMode::LEFT_ONLY:
        return execute_deferred_join_with_mode<MatchCollectionMode::LEFT_ONLY>(
            use_nested_loop, probe_is_columnar, is_root,
            use_nested_loop ? nullptr : &(*hash_table), build_input,
            probe_input, config, djoin, columnar_reader, deferred_plan, stats);

    case MatchCollectionMode::RIGHT_ONLY:
        return execute_deferred_join_with_mode<MatchCollectionMode::RIGHT_ONLY>(
            use_nested_loop, probe_is_columnar, is_root,
            use_nested_loop ? nullptr : &(*hash_table), build_input,
            probe_input, config, djoin, columnar_reader, deferred_plan, stats);
    }

    return DeferredResult{};
}

#endif // USE_DEFERRED_MATERIALIZATION

/**
 * @brief Public entry point: execute plan from root, return ColumnarTable.
 * @param plan Query plan with nodes and base tables.
 * @param context Reserved (unused).
 * @param stats_out Optional timing breakdown output.
 * @param show_detailed_timing Print timing to stdout if true.
 * @return Final ColumnarTable result.
 */
ColumnarTable execute(const Plan &plan, void *context, TimingStats *stats_out,
                      bool show_detailed_timing) {
    // Reset arena memory from previous query
    Contest::platform::g_arena_manager.reset_all();

    auto total_start = std::chrono::high_resolution_clock::now();

    TimingStats stats;

#ifdef USE_DEFERRED_MATERIALIZATION
    // Deferred materialization path: analyze plan, then execute with deferred
    // intermediate construction
    auto analyze_start = std::chrono::high_resolution_clock::now();
    DeferredPlan deferred_plan = analyze_plan(plan);
    auto analyze_end = std::chrono::high_resolution_clock::now();
    stats.analyze_plan_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(analyze_end -
                                                              analyze_start)
            .count();

    auto deferred_result =
        execute_deferred_impl(deferred_plan, plan.root, true, stats);
    ColumnarTable final_result =
        std::get<ColumnarTable>(std::move(deferred_result));
#else
    // Eager materialization path (original)
    auto result = execute_impl(plan, plan.root, true, stats);
    ColumnarTable final_result = std::get<ColumnarTable>(std::move(result));
#endif

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start);
    stats.total_execution_ms = total_elapsed.count();

    if (show_detailed_timing) {
        int64_t accounted = stats.hashtable_build_ms +
                            stats.hash_join_probe_ms +
                            stats.nested_loop_join_ms + stats.materialize_ms +
                            stats.setup_ms + stats.intermediate_ms;
#ifdef USE_DEFERRED_MATERIALIZATION
        accounted += stats.analyze_plan_ms;
#endif
        int64_t other = stats.total_execution_ms - accounted;

#ifdef USE_DEFERRED_MATERIALIZATION
        std::cout << "[DEFERRED] Plan Analysis Time: " << stats.analyze_plan_ms
                  << " ms\n";
#endif
        std::cout << "Hashtable Build Time: " << stats.hashtable_build_ms
                  << " ms\n";
        std::cout << "Hash Join Probe Time: " << stats.hash_join_probe_ms
                  << " ms\n";
        std::cout << "Nested Loop Join Time: " << stats.nested_loop_join_ms
                  << " ms\n";
        std::cout << "Materialization Time: " << stats.materialize_ms
                  << " ms\n";
        std::cout << "Intermediate Time: " << stats.intermediate_ms << " ms\n";
        std::cout << "Setup Time: " << stats.setup_ms << " ms\n";
        std::cout << "Other Overhead: " << other << " ms\n";
        std::cout << "Total Execution Time: " << stats.total_execution_ms
                  << " ms\n";
    }

    if (stats_out) {
        *stats_out = stats;
    }

    return std::move(final_result);
}

void *build_context() { return nullptr; }

void destroy_context(void *context) { (void)context; }

} // namespace Contest

/**
 *
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
 * @see plan.h, match_collector.h, materialize.h, construct_intermediate.h
 *
 **/
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

namespace Contest {

using namespace join;

using materialize::construct_intermediate_from_buffers;
using materialize::create_empty_result;
using materialize::materialize_from_buffers;

/**
 *
 * @brief Result variant: ExecuteResult (intermediate, value_t columns) or
 * ColumnarTable (final output per contest API).
 *
 **/
using JoinResult = std::variant<ExecuteResult, ColumnarTable>;

/**
 *
 * @brief Recursive join execution.
 * @param plan Query plan with JoinNodes and ScanNodes.
 * @param node_idx Current node index in plan.nodes.
 * @param is_root True -> ColumnarTable output; false -> ExecuteResult.
 * @param stats Timing accumulator.
 * @return JoinResult (intermediate or final).
 *
 **/
JoinResult execute_impl(const Plan &plan, size_t node_idx, bool is_root,
                        TimingStats &stats);

/**
 *
 * @brief Resolve ScanNode and JoinNode to JoinInput.
 *
 * ScanNode -> non-owning ColumnarTable*;
 * JoinNode -> recursive execution
 * returning owned ExecuteResult.
 * Implements depth-first traversal.
 *
 * @param plan Query plan.
 * @param node_idx Node index to resolve.
 * @param stats Timing accumulator.
 * @return JoinInput with data variant and metadata.
 *
 **/
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
        input.data = std::get<ExecuteResult>(std::move(result));
        input.table_id = 0;
    }
    return input;
}

/**
 *
 * @brief Unified probe + materialize helper templated on collection mode.
 *
 * Executes probe (nested loop or hash join) and materialization/intermediate
 * construction in a single function. Template parameter eliminates runtime
 * branching in hot loops.
 *
 * @tparam Mode Collection mode (BOTH, LEFT_ONLY, RIGHT_ONLY).
 *
 **/
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
                std::get<ExecuteResult>(probe_input.data);
            match_buffers = probe_intermediate<Mode>(
                *hash_table, probe_result[config.probe_attr]);
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
                columnar_reader, setup.results);
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
 *
 * @brief Core recursive join execution.
 *
 * Phases: resolve L/R inputs -> select build/probe (smaller=build) -> algorithm
 * choice -> build/probe -> output construction.
 *
 * Algorithm: nested loop if build_rows < HASH_TABLE_THRESHOLD (8); else radix-
 * partitioned hash join.
 *
 **/
JoinResult execute_impl(const Plan &plan, size_t node_idx, bool is_root,
                        TimingStats &stats) {
    auto &node = plan.nodes[node_idx];

    if (!std::holds_alternative<JoinNode>(node.data)) {
        return ExecuteResult{};
    }

    const auto &join = std::get<JoinNode>(node.data);
    const auto &output_attrs = node.output_attrs;
    const auto &left_node = plan.nodes[join.left];
    const auto &right_node = plan.nodes[join.right];

    JoinInput left_input = resolve_join_input(plan, join.left, stats);
    JoinInput right_input = resolve_join_input(plan, join.right, stats);

    /* Build/probe selection smaller input = build side; remaps output_attrs. */
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

    /* Skip unused-side row IDs if output needs only one side. */
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

    /**
     *
     * Dispatch based on collection mode - single runtime branch,
     * then fully specialized template instantiation with zero
     * branching in hot loops.
     *
     **/
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
    /* unreachable */
    return ExecuteResult{};
}

/**
 *
 * @brief Entry point: execute plan from root, return ColumnarTable.
 * @param plan Query plan with nodes and base tables.
 * @param context Reserved.
 * @param stats_out Optional timing breakdown output.
 * @return Final ColumnarTable result.
 *
 **/
ColumnarTable execute(const Plan &plan, void *context, TimingStats *stats_out) {

    Contest::platform::g_arena_manager.reset_all();
    auto total_start = std::chrono::high_resolution_clock::now();
    TimingStats stats;
    auto result = execute_impl(plan, plan.root, true, stats);
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start);
    stats.total_execution_ms = total_elapsed.count();
    if (stats_out) {
        *stats_out = stats;
    }
    return std::move(std::get<ColumnarTable>(result));
}

void *build_context() { return nullptr; }

void destroy_context(void *context) { (void)context; }

} // namespace Contest

/**
 * @file execute.cpp
 * @brief Depth-first join tree execution engine.
 *
 * Traverses plan tree: resolve inputs -> select build/probe -> algorithm
 * selection -> match collection -> output construction.
 *
 * Flow: execute() -> execute_impl() recursively -> resolve_input() for
 * ScanNode (ColumnarTable*) or JoinNode (IntermediateResult). Root produces
 * ColumnarTable; non-root produces IntermediateResult.
 *
 * Lifetimes: base tables live for query duration; IntermediateResult held on
 * stack until parent completes; VARCHAR refs valid via base table lifetime.
 *
 * Row order non-deterministic (work-stealing); semantically correct per SQL.
 *
 * @see plan.h, match_collector.h, materialize.h, construct_intermediate.h
 */
#include <cassert>
#include <foundation/attribute.h>
#include <functional>
#include <ostream>
#include <queue>
#include <string>
#include <utility>
#if defined(__APPLE__) && defined(__aarch64__)
#include <platform/hardware_darwin.h>
#elif defined(SPC__USE_BENCHMARKVM_HARDWARE)
#include <platform/hardware_benchmarkvm.h>
#else
#include <platform/hardware.h>
#endif

#include <chrono>
#include <data_access/columnar_reader.h>
#include <data_model/deferred_plan.h>
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

using materialize::construct_intermediate_with_tuples;
using materialize::create_empty_intermediate_result;
using materialize::materialize_from_buffers;

/**
 * @brief Result variant: IntermediateResult (non-root) or ColumnarTable (root).
 */
using JoinResult = std::variant<IntermediateResult, ColumnarTable>;

// Forward declaration
JoinResult execute_impl(const AnalyzedPlan &plan, size_t node_idx, bool is_root,
                        TimingStats &stats);

/**
 * @brief Resolve plan node to JoinInput.
 *
 * ScanNode -> non-owning ColumnarTable*; JoinNode -> recursive execution
 * returning owned IntermediateResult.
 */
JoinInput resolve_input(const AnalyzedPlan &plan, size_t node_idx,
                        TimingStats &stats) {
    JoinInput input;
    const auto &anode = plan[node_idx];
    const auto &pnode = plan.original_plan->nodes[node_idx];
    input.node = &pnode;
    input.analyzed_node = &anode;

    if (const auto *scan = std::get_if<AnalyzedScanNode>(&anode)) {
        input.data = &plan.original_plan->inputs[scan->base_table_id];
        input.table_id = scan->base_table_id;
    } else {
        auto result = execute_impl(plan, node_idx, false, stats);
        input.data = std::get<IntermediateResult>(std::move(result));
        input.table_id = 0;
    }
    return input;
}

/**
 * @brief Select build/probe sides for join input.
 */
BuildProbeConfig select_join_build_probe_side(
    const JoinNode &join, const JoinInput &left_input,
    const JoinInput &right_input,
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
 * @brief Unified probe + materialize helper templated on collection mode.
 */
template <MatchCollectionMode Mode>
JoinResult execute_join_with_mode(
    bool use_nested_loop, bool probe_is_columnar, bool is_root,
    const UnchainedHashtable *hash_table, const JoinInput &build_input,
    const JoinInput &probe_input, const BuildProbeConfig &config,
    const AnalyzedJoinNode &join_node, io::ColumnarReader &columnar_reader,
    const AnalyzedPlan &plan, TimingStats &stats) {

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
                std::get<IntermediateResult>(probe_input.data);

            // Use tuple-based probe if available
            if (probe_result.has_join_key_tuples() &&
                probe_result.join_key_idx.has_value() &&
                *probe_result.join_key_idx == config.probe_attr) {
                match_buffers = probe_tuples<Mode>(
                    *hash_table, *probe_result.join_key_tuples);
            } else {
                // Fall back to materialized column probe
                const auto *mat_col =
                    probe_result.get_materialized(config.probe_attr);
                if (!mat_col) {
                    std::fprintf(
                        stderr,
                        "ERROR: probe join key not materialized! "
                        "probe_attr=%zu "
                        "mat_map_size=%zu num_rows=%zu has_tuples=%d\n",
                        config.probe_attr, probe_result.materialized_map.size(),
                        probe_result.num_rows,
                        probe_result.has_join_key_tuples() ? 1 : 0);
                    std::abort();
                }
                match_buffers = probe_intermediate<Mode>(*hash_table, *mat_col);
            }
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
            final_result =
                materialize::create_empty_result(config.remapped_attrs);
        } else {
            // Prepare page indices for final materialization
            materialize::prepare_final_columns(
                columnar_reader, build_input, probe_input, join_node,
                config.remapped_attrs, build_input.output_size(),
                config.build_left);

            final_result = materialize_from_buffers<Mode>(
                match_buffers, build_input, probe_input, join_node,
                config.remapped_attrs, build_input.output_size(),
                config.build_left, columnar_reader, plan);
        }
        auto mat_end = std::chrono::high_resolution_clock::now();
        stats.materialize_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(mat_end -
                                                                  mat_start)
                .count();
        return final_result;
    } else {
        auto inter_start = std::chrono::high_resolution_clock::now();
        IntermediateResult result;
        if (total_matches > 0) {
            materialize::prepare_intermediate_columns(
                columnar_reader, build_input, probe_input, join_node,
                config.remapped_attrs, build_input.output_size(),
                config.build_left, join_node.parent_join_key_idx);

            construct_intermediate_with_tuples<Mode>(
                match_buffers, build_input, probe_input, join_node, config,
                config.build_left, *join_node.parent_join_key_idx,
                columnar_reader, result, plan);
        } else {
            result = create_empty_intermediate_result(join_node);
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
 * @brief Recursive join execution.
 */
JoinResult execute_impl(const AnalyzedPlan &plan, size_t node_idx, bool is_root,
                        TimingStats &stats) {
    const auto &anode = plan[node_idx];

    if (std::holds_alternative<AnalyzedScanNode>(anode)) {
        return IntermediateResult{};
    }

    const auto &ajoin = std::get<AnalyzedJoinNode>(anode);
    const auto &original_plan = *plan.original_plan;
    const auto &pnode = original_plan.nodes[node_idx];
    const auto &join = std::get<JoinNode>(pnode.data);

    // Resolve inputs
    JoinInput left_input = resolve_input(plan, ajoin.left_child_idx, stats);
    JoinInput right_input = resolve_input(plan, ajoin.right_child_idx, stats);

    // Build/probe selection
    auto setup_start = std::chrono::high_resolution_clock::now();
    auto config = select_join_build_probe_side(join, left_input, right_input,
                                               ajoin.output_attrs);
    const JoinInput &build_input = config.build_left ? left_input : right_input;
    const JoinInput &probe_input = config.build_left ? right_input : left_input;

    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();

    const size_t HASH_TABLE_THRESHOLD = 8;
    size_t build_rows = build_input.row_count(config.build_attr);
    bool use_nested_loop = (build_rows < HASH_TABLE_THRESHOLD);

    io::ColumnarReader columnar_reader;
    auto setup_end = std::chrono::high_resolution_clock::now();
    stats.setup_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
                          setup_end - setup_start)
                          .count();

    // Use pre-computed collection mode from plan analysis.
    // base_collection_mode assumes build=left; flip if build=right at runtime.
    MatchCollectionMode mode = ajoin.base_collection_mode;
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
            hash_table = build_from_columnar(build_input, config.build_attr);
        } else {
            const auto &ir = std::get<IntermediateResult>(build_input.data);

            // Use tuple-based build if available and matches build_attr
            if (ir.has_join_key_tuples() && ir.join_key_idx.has_value() &&
                *ir.join_key_idx == config.build_attr) {
                hash_table.emplace(ir.join_key_tuples->row_count());
                hash_table->build_from_tuples(*ir.join_key_tuples);
            } else {
                // Fall back to materialized column build
                const auto *mat_col = ir.get_materialized(config.build_attr);
                if (!mat_col) {
                    std::fprintf(
                        stderr,
                        "ERROR: build join key not materialized! "
                        "build_attr=%zu "
                        "mat_map_size=%zu num_rows=%zu has_tuples=%d\n",
                        config.build_attr, ir.materialized_map.size(),
                        ir.num_rows, ir.has_join_key_tuples() ? 1 : 0);
                    std::abort();
                }
                hash_table.emplace(mat_col->row_count());
                hash_table->build_intermediate(*mat_col);
            }
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
        return execute_join_with_mode<MatchCollectionMode::BOTH>(
            use_nested_loop, probe_is_columnar, is_root,
            use_nested_loop ? nullptr : &(*hash_table), build_input,
            probe_input, config, ajoin, columnar_reader, plan, stats);

    case MatchCollectionMode::LEFT_ONLY:
        return execute_join_with_mode<MatchCollectionMode::LEFT_ONLY>(
            use_nested_loop, probe_is_columnar, is_root,
            use_nested_loop ? nullptr : &(*hash_table), build_input,
            probe_input, config, ajoin, columnar_reader, plan, stats);

    case MatchCollectionMode::RIGHT_ONLY:
        return execute_join_with_mode<MatchCollectionMode::RIGHT_ONLY>(
            use_nested_loop, probe_is_columnar, is_root,
            use_nested_loop ? nullptr : &(*hash_table), build_input,
            probe_input, config, ajoin, columnar_reader, plan, stats);
    }

    return IntermediateResult{};
}

/**
 *
 * @brief Prints the plan tree with metadata.
 *
 * @param the query plan itself.
 * @param queue that should contain the root node.
 *
 **/
static std::function<void(const Plan&, std::queue<std::tuple<int, int>>)> 
print_plan = [](const Plan& plan, std::queue<std::tuple<int, int>> q) {
    if (q.empty()) return;
    int initial_size = q.size();
    for (int i = 0; i < initial_size; i++) {
        auto [parent_idx, node_idx] = q.front();
        q.pop();
        const auto& node = plan.nodes[node_idx];
        std::cout << "parent: " << parent_idx << ", node: "<< node_idx << " size: "
            << node.output_attrs.size() << " pairs: { ";
        for (int i = 0; i < node.output_attrs.size(); i++) {
            auto [col, type] = node.output_attrs[i];
            if (DataType::INT32 == type)
                std::cout << "(" << col << ", INT32)-";
            else
                std::cout << "(" << col << ", STR)-";
        }
        if (const auto* join = std::get_if<JoinNode>(&node.data)) {
            std::cout << "left_key: " << join->left_attr;
            std::cout << " right_key: " << join->right_attr;
            q.emplace(node_idx ,join->left);
            q.emplace(node_idx, join->right);
        }
        std::cout << "}\n";
    }
    std::cout << std::endl << std::endl << std::endl << std::endl ;
    print_plan(plan, std::move(q));
};

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

    // Analyze plan and execute with deferred intermediate construction
    auto analyze_start = std::chrono::high_resolution_clock::now();
    AnalyzedPlan analyzed_plan = analyze_plan(plan);
    auto analyze_end = std::chrono::high_resolution_clock::now();
    stats.analyze_plan_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(analyze_end -
                                                              analyze_start)
            .count();
    /*
    auto result = execute_impl(analyzed_plan, plan.root, true, stats);
    ColumnarTable final_result = std::get<ColumnarTable>(std::move(result));
    */
    std::queue<std::tuple<int,int>> q;
    q.emplace(0, plan.root);
    print_plan(plan, q);

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start);
    stats.total_execution_ms = total_elapsed.count();

    if (show_detailed_timing) {
        int64_t accounted =
            stats.hashtable_build_ms + stats.hash_join_probe_ms +
            stats.nested_loop_join_ms + stats.materialize_ms + stats.setup_ms +
            stats.intermediate_ms + stats.analyze_plan_ms;
        int64_t other = stats.total_execution_ms - accounted;

        std::cout << "Plan Analysis Time: " << stats.analyze_plan_ms << " ms\n";
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

    return ColumnarTable(); 
}

void *build_context() { return nullptr; }

void destroy_context(void *context) { (void)context; }

} // namespace Contest

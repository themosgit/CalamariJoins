#if defined(__APPLE__) && defined(__aarch64__)
#include <hardware_darwin.h>
#else
#include <hardware.h>
#endif

#include <columnar_reader.h>
#include <construct_intermediate.h>
#include <hash_join.h>
#include <hashtable.h>
#include <intermediate.h>
#include <join_setup.h>
#include <materialize.h>
#include <nested_loop.h>
#include <variant>
#include <chrono>
#include <iostream>

namespace Contest {

using ExecuteResult = std::vector<mema::column_t>;
using JoinResult = std::variant<ExecuteResult, ColumnarTable>;

/**
 * NOTE: intermediate means the column_t representation
 *       columnar means ColumnarTable page format
 *
 * this refactored version of execute focuses
 * on execute_impl and using modules to complete
 * each join. this is done because of the many
 * variations of joins that may arise.
 *
 * when is_root is true, we materialize directly to ColumnarTable
 * to avoid intermediate conversion overhead.
 *
 * our join pipeline starts by determining whether the tables
 * are in the intermediate column_t representation or in the
 * original ColumnarTable format
 *
 * it later selects which side is build or probe based on the
 * size of each table this remaps the output_attrs as well.
 *
 * next the join algorithm is determined nested loop joins for
 * small tables hash_join for bigger ones
 *
 * the setup step initializes all output column metadata and every
 * intermediate data structure we use to make the join pipeline
 * faster.
 *
 * we have intermediate and columnar mixed joins
 * for both hash_join and nested_loop
 *
 * after the joins are done and our MatchCollectors
 * are populated which hold match rowids from the just joined
 * table's join columns
 *
 * we call the appropriate materialization function in order
 * to produce a column_t intermediate result or if we are
 * on the root node the final ColumnarTable result.
 *
 **/

JoinResult execute_impl(const Plan &plan, size_t node_idx, bool is_root, TimingStats &stats) {
    auto &node = plan.nodes[node_idx];

    if (!std::holds_alternative<JoinNode>(node.data)) {
        return ExecuteResult{};
    }

    const auto &join = std::get<JoinNode>(node.data);
    const auto &output_attrs = node.output_attrs;
    const auto &left_node = plan.nodes[join.left];
    const auto &right_node = plan.nodes[join.right];

    /* determine intermediate and columnar tables */
    JoinInput left_input, right_input;
    left_input.node = &left_node;
    right_input.node = &right_node;

    /* if it is a columnar node we are at a leaf */
    if (const auto *left_scan = std::get_if<ScanNode>(&left_node.data)) {
        left_input.data = &plan.inputs[left_scan->base_table_id];
        left_input.table_id = left_scan->base_table_id;
    } else {
        auto result = execute_impl(plan, join.left, false, stats);
        left_input.data = std::get<ExecuteResult>(std::move(result));
        left_input.table_id = 0;
    }

    if (const auto *right_scan = std::get_if<ScanNode>(&right_node.data)) {
        right_input.data = &plan.inputs[right_scan->base_table_id];
        right_input.table_id = right_scan->base_table_id;
    } else {
        auto result = execute_impl(plan, join.right, false, stats);
        right_input.data = std::get<ExecuteResult>(std::move(result));
        right_input.table_id = 0;
    }

    /* select build probe sides based on table size and configure proper args */
    auto setup_start = std::chrono::high_resolution_clock::now();
    auto config =
        select_build_probe_side(join, left_input, right_input, output_attrs);
    const JoinInput &build_input = config.build_left ? left_input : right_input;
    const JoinInput &probe_input = config.build_left ? right_input : left_input;
    const auto &build_node = config.build_left ? left_node : right_node;
    const auto &probe_node = config.build_left ? right_node : left_node;

    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();

    /* select join algorithm */
    const size_t HASH_TABLE_THRESHOLD = 8;
    size_t build_rows = build_input.row_count(config.build_attr);
    bool use_nested_loop = (build_rows < HASH_TABLE_THRESHOLD);

    /* initialize join data */
    JoinSetup setup = setup_join(build_input, probe_input, build_node,
                                 probe_node, left_node, right_node, left_input,
                                 right_input, output_attrs, build_rows);
    auto setup_end = std::chrono::high_resolution_clock::now();
    auto setup_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(setup_end - setup_start);
    stats.setup_ms += setup_elapsed.count();

    /* nested loop join path */
    if (use_nested_loop) {
        MatchCollector collector;

        auto nested_loop_start = std::chrono::high_resolution_clock::now();
        if (build_is_columnar && probe_is_columnar) {
            nested_loop_from_columnar(build_input, probe_input,
                                      config.build_attr, config.probe_attr,
                                      setup.columnar_reader, collector);
        } else if (!build_is_columnar && !probe_is_columnar) {
            const auto &build_result =
                std::get<ExecuteResult>(build_input.data);
            const auto &probe_result =
                std::get<ExecuteResult>(probe_input.data);
            nested_loop_from_intermediate(build_result, probe_result,
                                          config.build_attr, config.probe_attr,
                                          collector);
        } else {
            nested_loop_mixed(build_input, probe_input, config.build_attr,
                              config.probe_attr, setup.columnar_reader,
                              collector);
        }
        auto nested_loop_end = std::chrono::high_resolution_clock::now();
        auto nested_loop_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(nested_loop_end - nested_loop_start);
        stats.nested_loop_join_ms += nested_loop_elapsed.count();

        auto materialize_start = std::chrono::high_resolution_clock::now();
        auto result = materialize_join_results(collector, build_input, probe_input,
                                               config, build_node, probe_node, setup,
                                               plan, is_root);
        auto materialize_end = std::chrono::high_resolution_clock::now();
        auto materialize_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(materialize_end - materialize_start);
        stats.materialize_ms += materialize_elapsed.count();

        return result;

        /* hash join path */
    } else {
        /* building hash table based on columnar or intermediate */
        auto build_start = std::chrono::high_resolution_clock::now();
        UnchainedHashtable hash_table =
            build_is_columnar
                ? build_from_columnar(build_input, config.build_attr)
                : build_from_intermediate(build_input, config.build_attr);
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(build_end - build_start);
        stats.hashtable_build_ms += build_elapsed.count();

        /* selecting proper probe */
        MatchCollector collector;
        collector.reserve(build_rows);

        auto probe_start = std::chrono::high_resolution_clock::now();
        if (probe_is_columnar) {
            probe_columnar(hash_table, probe_input, config.probe_attr,
                           collector);
        } else {
            const auto &probe_result =
                std::get<ExecuteResult>(probe_input.data);
            probe_intermediate(hash_table, probe_result[config.probe_attr],
                               collector);
        }
        auto probe_end = std::chrono::high_resolution_clock::now();
        auto probe_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(probe_end - probe_start);
        stats.hash_join_probe_ms += probe_elapsed.count();

        auto materialize_start = std::chrono::high_resolution_clock::now();
        auto result = materialize_join_results(collector, build_input, probe_input,
                                        config, build_node, probe_node, setup,
                                        plan, is_root);
        auto materialize_end = std::chrono::high_resolution_clock::now();
        auto materialize_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(materialize_end - materialize_start);
        stats.materialize_ms += materialize_elapsed.count();
        return  result;
    }
}

ColumnarTable execute(const Plan &plan, void *context, TimingStats *stats_out, bool show_detailed_timing) {
    auto total_start = std::chrono::high_resolution_clock::now();

    TimingStats stats;
    auto result = execute_impl(plan, plan.root, true, stats);

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    stats.total_execution_ms = total_elapsed.count();

    if (show_detailed_timing) {
        int64_t accounted = stats.hashtable_build_ms + stats.hash_join_probe_ms +
                           stats.nested_loop_join_ms + stats.materialize_ms + stats.setup_ms;
        int64_t other = stats.total_execution_ms - accounted;

        std::cout << "Hashtable Build Time: " << stats.hashtable_build_ms << " ms\n";
        std::cout << "Hash Join Probe Time: " << stats.hash_join_probe_ms << " ms\n";
        std::cout << "Nested Loop Join Time: " << stats.nested_loop_join_ms << " ms\n";
        std::cout << "Materialization Time: " << stats.materialize_ms << " ms\n";
        std::cout << "Setup Time: " << stats.setup_ms << " ms\n";
        std::cout << "Other Overhead: " << other << " ms\n";
        std::cout << "Total Execution Time: " << stats.total_execution_ms << " ms\n";
    }

    if (stats_out) {
        *stats_out = stats;
    }

    return std::move(std::get<ColumnarTable>(result));
}

void *build_context() { return nullptr; }

void destroy_context(void *context) {}

} // namespace Contest

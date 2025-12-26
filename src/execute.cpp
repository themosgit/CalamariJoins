#if defined(__APPLE__) && defined(__aarch64__)
#include <hardware_darwin.h>
#else
#include <hardware.h>
#endif

#include <chrono>
#include <columnar_reader.h>
#include <construct_intermediate.h>
#include <hash_join.h>
#include <hashtable.h>
#include <intermediate.h>
#include <iostream>
#include <join_setup.h>
#include <materialize.h>
#include <nested_loop.h>
#include <variant>

namespace Contest {

using ExecuteResult = std::vector<mema::column_t>;
using JoinResult = std::variant<ExecuteResult, ColumnarTable>;

// Static variables to track hashtable build times
static thread_local int64_t current_query_build_time_us = 0;
static thread_local int64_t total_build_time_us = 0;

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

JoinResult execute_impl(const Plan &plan, size_t node_idx, bool is_root) {
    auto &node = plan.nodes[node_idx];

    if (!std::holds_alternative<JoinNode>(node.data)) {
        return ExecuteResult{};
    }

    const auto &join = std::get<JoinNode>(node.data);
    const auto &output_attrs = node.output_attrs;
    const auto &left_node = plan.nodes[join.left];
    const auto &right_node = plan.nodes[join.right];

    bool left_is_columnar = std::holds_alternative<ScanNode>(left_node.data);
    bool right_is_columnar = std::holds_alternative<ScanNode>(right_node.data);

    /* determine intermediate and columnar tables */
    JoinInput left_input, right_input;
    left_input.node = &left_node;
    right_input.node = &right_node;

    /* if it is a columnar node we are at a leaf */
    if (left_is_columnar) {
        const auto &left_scan = std::get<ScanNode>(left_node.data);
        left_input.data = &plan.inputs[left_scan.base_table_id];
        left_input.table_id = left_scan.base_table_id;
    } else {
        auto result = execute_impl(plan, join.left, false);
        left_input.data = std::get<ExecuteResult>(std::move(result));
        left_input.table_id = 0;
    }

    if (right_is_columnar) {
        const auto &right_scan = std::get<ScanNode>(right_node.data);
        right_input.data = &plan.inputs[right_scan.base_table_id];
        right_input.table_id = right_scan.base_table_id;
    } else {
        auto result = execute_impl(plan, join.right, false);
        right_input.data = std::get<ExecuteResult>(std::move(result));
        right_input.table_id = 0;
    }

    /* select build probe sides based on table size and configure proper args */
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

    /* nested loop join path */
    if (use_nested_loop) {
        MatchCollector collector;

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

        return materialize_join_results(collector, build_input, probe_input,
                                        config, build_node, probe_node, setup,
                                        plan, is_root);

        /* hash join path */
    } else {
        /* building hash table based on columnar or intermediate */
        // Serial build timing
        auto serial_build_start = std::chrono::high_resolution_clock::now();
        UnchainedHashtable hash_table_serial =
            build_is_columnar
                ? build_from_columnar(build_input, config.build_attr)
                : build_from_intermediate(build_input, config.build_attr);
        auto serial_build_end = std::chrono::high_resolution_clock::now();
        auto serial_build_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            serial_build_end - serial_build_start);
        
        // Parallel build timing
        auto parallel_build_start = std::chrono::high_resolution_clock::now();
        UnchainedHashtable hash_table_parallel =
            build_is_columnar
                ? build_from_columnar_parallel(build_input, config.build_attr)
                : build_from_intermediate_parallel(build_input, config.build_attr);
        auto parallel_build_end = std::chrono::high_resolution_clock::now();
        auto parallel_build_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            parallel_build_end - parallel_build_start);
        
        // Print comparison
        // std::cout << "Hashtable build time - Serial: " << serial_build_duration.count() 
        //           << " us (" << serial_build_duration.count() / 1000.0 << " ms), "
        //           << "Parallel: " << parallel_build_duration.count() 
        //           << " us (" << parallel_build_duration.count() / 1000.0 << " ms), "
        //           << "Speedup: " << (double)serial_build_duration.count() / parallel_build_duration.count() 
        //           << "x" << std::endl;
        
        current_query_build_time_us += parallel_build_duration.count();
        
        // Use parallel build for execution
        UnchainedHashtable &hash_table = hash_table_parallel;

        /* selecting proper probe */
        MatchCollector collector;
        collector.reserve(build_rows);

        if (probe_is_columnar) {
            probe_columnar(hash_table, probe_input, config.probe_attr,
                           collector);
        } else {
            const auto &probe_result =
                std::get<ExecuteResult>(probe_input.data);
            probe_intermediate(hash_table, probe_result[config.probe_attr],
                               collector);
        }

        return materialize_join_results(collector, build_input, probe_input,
                                        config, build_node, probe_node, setup,
                                        plan, is_root);
    }
}

ColumnarTable execute(const Plan &plan, void *context) {
    // Reset per-query build time at the start of each query
    current_query_build_time_us = 0;
    
    auto result = execute_impl(plan, plan.root, true);
    
    // Print total build time for this query
    total_build_time_us += current_query_build_time_us;
    std::cout << "Total hashtable build time for query: " 
              << current_query_build_time_us << " microseconds ("
              << current_query_build_time_us / 1000.0 << " ms)" << std::endl;
    
    return std::move(std::get<ColumnarTable>(result));
}

void *build_context() { 
    // Reset total build time when context is built
    total_build_time_us = 0;
    return nullptr; 
}

void destroy_context(void *context) {
    // Print grand total build time across all queries
    std::cout << "Total hashtable build time for all queries: " 
              << total_build_time_us << " microseconds ("
              << total_build_time_us / 1000.0 << " ms)" << std::endl;
}

} // namespace Contest

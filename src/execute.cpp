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
#include <variant>

namespace Contest {

using JoinResult = std::variant<ExecuteResult, ColumnarTable>;

JoinResult execute_impl(const Plan &plan, size_t node_idx, bool is_root,
                        TimingStats &stats);

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

    /* determine intermediate and columnar tables */
    JoinInput left_input = resolve_join_input(plan, join.left, stats);
    JoinInput right_input = resolve_join_input(plan, join.right, stats);

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
    auto setup_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        setup_end - setup_start);
    stats.setup_ms += setup_elapsed.count();

    /* Determine collection mode based on output requirements */
    MatchCollectionMode collection_mode = determine_collection_mode(
        config.remapped_attrs, config.build_left ? left_input.output_size()
                                                 : right_input.output_size());

    MatchCollector collector(collection_mode);
    /* nested loop join path */
    if (use_nested_loop) {

        auto nested_loop_start = std::chrono::high_resolution_clock::now();

        nested_loop_join(build_input, probe_input, config.build_attr,
                         config.probe_attr, collector, collection_mode);
        auto nested_loop_end = std::chrono::high_resolution_clock::now();
        auto nested_loop_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                nested_loop_end - nested_loop_start);
        stats.nested_loop_join_ms += nested_loop_elapsed.count();
    } else {
        /* building hash table based on columnar or intermediate */
        auto build_start = std::chrono::high_resolution_clock::now();
        UnchainedHashtable hash_table =
            build_is_columnar
                ? build_from_columnar(build_input, config.build_attr)
                : build_from_intermediate(build_input, config.build_attr);
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(build_end -
                                                                  build_start);
        stats.hashtable_build_ms += build_elapsed.count();

        /* selecting proper probe */
        auto probe_start = std::chrono::high_resolution_clock::now();
        if (probe_is_columnar) {
            probe_columnar(hash_table, probe_input, config.probe_attr,
                           collector, collection_mode);
        } else {
            const auto &probe_result =
                std::get<ExecuteResult>(probe_input.data);
            probe_intermediate(hash_table, probe_result[config.probe_attr],
                               collector, collection_mode);
        }
        auto probe_end = std::chrono::high_resolution_clock::now();
        auto probe_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(probe_end -
                                                                  probe_start);
        stats.hash_join_probe_ms += probe_elapsed.count();
    }
    JoinResult final_result;
    if (is_root) {
        auto mat_start = std::chrono::high_resolution_clock::now();
        if (collector.size() == 0) {
            final_result = create_empty_result(config.remapped_attrs);
        } else {
            /* prepare PageIndex now - after probing, before materialization */
            prepare_output_columns(
                setup.columnar_reader, build_input, probe_input, build_node,
                probe_node, config.remapped_attrs, build_input.output_size());

            final_result = materialize(collector, build_input, probe_input,
                                       config.remapped_attrs, build_node,
                                       probe_node, build_input.output_size(),
                                       setup.columnar_reader, plan);
        }
        auto mat_end = std::chrono::high_resolution_clock::now();
        stats.materialize_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(mat_end -
                                                                  mat_start)
                .count();

    } else {
        auto inter_start = std::chrono::high_resolution_clock::now();
        if (collector.size() > 0) {
            /* prepare PageIndex now - after probing, before intermediate
             * construction */
            prepare_output_columns(
                setup.columnar_reader, build_input, probe_input, build_node,
                probe_node, config.remapped_attrs, build_input.output_size());

            construct_intermediate(collector, build_input, probe_input,
                                   config.remapped_attrs, build_node,
                                   probe_node, build_input.output_size(),
                                   setup.columnar_reader, setup.results);
        }
        final_result = std::move(setup.results);
        auto inter_end = std::chrono::high_resolution_clock::now();
        stats.intermediate_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(inter_end -
                                                                  inter_start)
                .count();
    }

    return final_result;
}

ColumnarTable execute(const Plan &plan, void *context, TimingStats *stats_out,
                      bool show_detailed_timing) {
    auto total_start = std::chrono::high_resolution_clock::now();

    TimingStats stats;
    auto result = execute_impl(plan, plan.root, true, stats);

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start);
    stats.total_execution_ms = total_elapsed.count();

    if (show_detailed_timing) {
        int64_t accounted =
            stats.hashtable_build_ms + stats.hash_join_probe_ms +
            stats.nested_loop_join_ms + stats.materialize_ms + stats.setup_ms;
        int64_t other = stats.total_execution_ms - accounted;

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

    return std::move(std::get<ColumnarTable>(result));
}

void *build_context() { return nullptr; }

void destroy_context(void *context) { (void)context; }

} // namespace Contest

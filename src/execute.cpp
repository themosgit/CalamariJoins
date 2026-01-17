#include "attribute.h"
#if defined(__APPLE__) && defined(__aarch64__)
#include <hardware_darwin.h>
#elif defined(SPC__USE_BENCHMARKVM_HARDWARE)
#include <hardware_benchmarkvm.h>
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

static thread_local int64_t current_query_build_time_us = 0;
static thread_local int64_t total_build_time_us = 0;

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

static std::pair<size_t, size_t>
find_attribute_source(const Plan &plan, size_t node_idx, size_t output_pos) {
    const auto &node = plan.nodes[node_idx];

    if (std::holds_alternative<ScanNode>(node.data)) {
        const auto &scan = std::get<ScanNode>(node.data);
        const auto &[col_idx, dtype] = node.output_attrs[output_pos];
        return {scan.base_table_id, col_idx};
    } else if (std::holds_alternative<JoinNode>(node.data)) {
        const auto &join = std::get<JoinNode>(node.data);
        const auto &left_node = plan.nodes[join.left];
        const auto &right_node = plan.nodes[join.right];

        const auto &[concat_idx, dtype] = node.output_attrs[output_pos];
        size_t left_output_count = left_node.output_attrs.size();

        if (concat_idx < left_output_count) {
            return find_attribute_source(plan, join.left, concat_idx);
        } else {
            return find_attribute_source(plan, join.right,
                                         concat_idx - left_output_count);
        }
    }

    return {0, 0};
}

static void print_plan_tree(const Plan &plan, size_t node_idx,
                            const std::string &prefix, bool is_last) {
    const auto &node = plan.nodes[node_idx];
    std::cout << prefix;
    std::cout << (is_last ? "└── " : "├── ");

    if (std::holds_alternative<ScanNode>(node.data)) {
        const auto &scan = std::get<ScanNode>(node.data);
        std::cout << "SCAN Table " << scan.base_table_id;

        std::cout << "\n"
                  << prefix << (is_last ? "    " : "│   ") << "Output: [";
        for (size_t j = 0; j < node.output_attrs.size(); ++j) {
            const auto &[col_idx, dtype] = node.output_attrs[j];
            if (j > 0)
                std::cout << ", ";
            auto [table_id, col_id] = find_attribute_source(plan, node_idx, j);
            std::cout << "pos" << j << "=T" << table_id << ".col" << col_id;
            std::cout << " (";
            switch (dtype) {
            case DataType::INT32:
                std::cout << "INT32";
                break;
            case DataType::VARCHAR:
                std::cout << "VARCHAR";
                break;
            }
            std::cout << ")";
        }
        std::cout << "]\n";

    } else if (std::holds_alternative<JoinNode>(node.data)) {
        const auto &join = std::get<JoinNode>(node.data);

        const auto &left_node = plan.nodes[join.left];
        const auto &right_node = plan.nodes[join.right];

        std::cout << "JOIN Node" << node_idx;

        auto [left_table, left_col] =
            find_attribute_source(plan, join.left, join.left_attr);
        auto [right_table, right_col] =
            find_attribute_source(plan, join.right, join.right_attr);

        std::cout << "\n"
                  << prefix << (is_last ? "    " : "│   ")
                  << "Join Condition: attr" << join.left_attr << " (T"
                  << left_table << ".col" << left_col << ")"
                  << " = attr" << join.right_attr << " (T" << right_table
                  << ".col" << right_col << ")"
                  << " [build=" << (join.build_left ? "left" : "right")
                  << "]\n";

        std::cout << prefix << (is_last ? "    " : "│   ") << "Output: [";

        for (size_t j = 0; j < node.output_attrs.size(); ++j) {
            const auto &[concat_idx, dtype] = node.output_attrs[j];
            if (j > 0)
                std::cout << ", ";

            auto [table_id, col_id] = find_attribute_source(plan, node_idx, j);
            std::cout << "pos" << j << "=T" << table_id << ".col" << col_id;

            size_t left_output_count = left_node.output_attrs.size();
            if (concat_idx == join.left_attr ||
                (concat_idx >= left_output_count &&
                 (concat_idx - left_output_count) == join.right_attr)) {
                std::cout << "*";
            }
        }
        std::cout << "]\n";
    }

    if (std::holds_alternative<JoinNode>(node.data)) {
        const auto &join = std::get<JoinNode>(node.data);
        std::string new_prefix = prefix + (is_last ? "    " : "│   ");
        print_plan_tree(plan, join.left, new_prefix, false);
        print_plan_tree(plan, join.right, new_prefix, true);
    }
}

ColumnarTable execute(const Plan &plan, void *context, TimingStats *stats_out,
                      bool show_detailed_timing) {
    auto total_start = std::chrono::high_resolution_clock::now();

    // Print the entire plan tree before recursion starts
    // std::cout << "\n========== QUERY PLAN TREE ==========\n";
    // std::cout << "Total nodes: " << plan.nodes.size() << "\n";
    // std::cout << "Total input tables: " << plan.inputs.size() << "\n\n";
    // print_plan_tree(plan, plan.root, "", true);
    // std::cout << "=====================================\n\n";
    //
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

void *build_context() {
    // Reset total build time when context is built
    total_build_time_us = 0;
    return nullptr;
}

void destroy_context(void *context) {}

} // namespace Contest

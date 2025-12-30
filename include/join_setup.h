#pragma once

#include <columnar_reader.h>
#include <intermediate.h>
#include <plan.h>
#include <tuple>
#include <variant>
#include <vector>

/**
 *
 *  join initialization and setup utilities
 *
 *  this header contains all the structures and functions needed
 *  to prepare a join operation before execution begins
 *
 *  JoinInput provides unified interface for both column_t
 *  intermediate results and ColumnarTable inputs allows join
 *  algorithms to work with any combination of intermediate and
 *  columnar data
 *
 *  BuildProbeConfig stores the decision of which side becomes
 *  build vs probe select_build_probe_side chooses smaller table
 *  as build side for efficiency remaps output attributes from
 *  left/right semantics to build/probe semantics
 *
 *  initialize_output_columns pre-allocates column_t output
 *  vector sets source_table and source_column metadata for
 *  varchar string tracking reserves space based on estimated
 *  match count to reduce reallocations
 *
 *  JoinSetup holds all initialized resources ready for join
 *  execution setup_join orchestrates initialization and prepares
 *  columnar_reader builds page indices for efficient random
 *  access during materialization
 *
 **/

namespace Contest {

using ExecuteResult = std::vector<mema::column_t>;

/**
 *
 *  unified input abstraction for join operations
 *  holds either column_t intermediate results or ColumnarTable pointer
 *  node provides output attribute mapping
 *  table_id tracks source table for columnar inputs
 *
 **/
struct JoinInput {
    std::variant<ExecuteResult, const ColumnarTable *> data;
    const PlanNode *node;
    uint8_t table_id;

    bool is_columnar() const {
        return std::holds_alternative<const ColumnarTable *>(data);
    }

    size_t row_count(size_t col_idx) const {
        if (is_columnar()) {
            auto *table = std::get<const ColumnarTable *>(data);
            auto [actual_col_idx, _] = node->output_attrs[col_idx];
            return table->num_rows;
        } else {
            return std::get<ExecuteResult>(data)[col_idx].row_count();
        }
    }

    size_t output_size() const { return node->output_attrs.size(); }
};

/**
 *
 *  configuration for build and probe side selection
 *  stores which side is build and remapped output attributes
 *  remapped_attrs translates left/right to build/probe semantics
 *
 **/
struct BuildProbeConfig {
    bool build_left;
    std::vector<std::tuple<size_t, DataType>> remapped_attrs;
    size_t build_attr;
    size_t probe_attr;
};

/**
 *
 *  selects build and probe sides based on table size
 *  smaller table becomes build side for hash join
 *  remaps output attributes from left/right to build/probe
 *  returns configuration for join execution
 *
 **/
inline BuildProbeConfig select_build_probe_side(
    const JoinNode &join, const JoinInput &left_input,
    const JoinInput &right_input,
    const std::vector<std::tuple<size_t, DataType>> &output_attrs) {
    BuildProbeConfig config;

    /* determine build and probe sides based on size */
    size_t left_rows = left_input.row_count(join.left_attr);
    size_t right_rows = right_input.row_count(join.right_attr);
    config.build_left = left_rows <= right_rows;

    config.build_attr = config.build_left ? join.left_attr : join.right_attr;
    config.probe_attr = config.build_left ? join.right_attr : join.left_attr;

    /* remap output_attrs from left/right to build/probe semantics */
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
 *
 *  initializes output column_t vector with proper metadata
 *  sets source_table and source_column for string materialization
 *  tracks original ColumnarTable location for varchar columns
 *  pre-reserves space based on estimated row count
 *
 **/
inline ExecuteResult initialize_output_columns(
    const std::vector<std::tuple<size_t, DataType>> &output_attrs,
    const PlanNode &left_node, const PlanNode &right_node,
    const JoinInput &left_input, const JoinInput &right_input,
    size_t estimated_rows) {
    ExecuteResult results;
    results.reserve(output_attrs.size());
    size_t left_size = left_input.output_size();

    auto set_column_metadata = [](mema::column_t &col, const JoinInput &input,
                                  const PlanNode &node, size_t col_idx) {
        auto [actual_col_idx, _] = node.output_attrs[col_idx];
        if (input.is_columnar()) {
            col.source_table = input.table_id;
            col.source_column = actual_col_idx;
        } else {
            const auto &result = std::get<ExecuteResult>(input.data);
            col.source_table = result[col_idx].source_table;
            col.source_column = result[col_idx].source_column;
        }
    };

    for (size_t i = 0; i < output_attrs.size(); ++i) {
        auto [col_idx, _] = output_attrs[i];

        mema::column_t col;

        if (col_idx < left_size) {
            set_column_metadata(col, left_input, left_node, col_idx);
        } else {
            set_column_metadata(col, right_input, right_node,
                                col_idx - left_size);
        }

        results.push_back(std::move(col));
    }

    return results;
}

/**
 *
 *  holds initialized resources for join execution
 *  results stores pre-allocated column_t output columns
 *  columnar_reader prepared with page indices if needed
 *  prepared flag indicates if columnar_reader is ready
 *
 **/
struct JoinSetup {
    ExecuteResult results;
    ColumnarReader columnar_reader;
    bool prepared;

    JoinSetup() : prepared(false) {}
};

/**
 *
 *  initializes all resources needed for join execution
 *  allocates output columns with proper metadata
 *  prepares columnar_reader if inputs are columnar
 *  returns setup ready for hash or nested loop join
 *
 **/
inline JoinSetup
setup_join(const JoinInput &build_input, const JoinInput &probe_input,
           const PlanNode &build_node, const PlanNode &probe_node,
           const PlanNode &left_node, const PlanNode &right_node,
           const JoinInput &left_input, const JoinInput &right_input,
           const std::vector<std::tuple<size_t, DataType>> &output_attrs,
           size_t estimated_rows) {
    JoinSetup setup;

    /* initialize output columns */
    setup.results =
        initialize_output_columns(output_attrs, left_node, right_node,
                                  left_input, right_input, estimated_rows);

    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();

    /* prepare columnar reader if any input is columnar */
    if (build_is_columnar || probe_is_columnar) {
        if (build_is_columnar) {
            std::vector<const Column *> build_columns;
            auto *build_table =
                std::get<const ColumnarTable *>(build_input.data);
            for (size_t i = 0; i < build_node.output_attrs.size(); ++i) {
                auto [actual_col_idx, _] = build_node.output_attrs[i];
                build_columns.push_back(&build_table->columns[actual_col_idx]);
            }
            setup.columnar_reader.prepare_build(build_columns);
        }

        if (probe_is_columnar) {
            std::vector<const Column *> probe_columns;
            auto *probe_table =
                std::get<const ColumnarTable *>(probe_input.data);
            for (size_t i = 0; i < probe_node.output_attrs.size(); ++i) {
                auto [actual_col_idx, _] = probe_node.output_attrs[i];
                probe_columns.push_back(&probe_table->columns[actual_col_idx]);
            }
            setup.columnar_reader.prepare_probe(probe_columns);
        }

        setup.prepared = true;
    }

    return setup;
}

} // namespace Contest

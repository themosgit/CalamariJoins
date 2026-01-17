#pragma once

#include <columnar_reader.h>
#include <intermediate.h>
#include <match_collector.h>
#include <plan.h>
#include <tuple>
#include <variant>
#include <vector>

namespace Contest {

using ExecuteResult = std::vector<mema::column_t>;

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

struct BuildProbeConfig {
    bool build_left;
    std::vector<std::tuple<size_t, DataType>> remapped_attrs;
    size_t build_attr;
    size_t probe_attr;
};

inline std::tuple<const JoinInput &, const PlanNode &, size_t>
resolve_input_source(size_t global_idx, size_t split_point,
                     const JoinInput &input_a, const PlanNode &node_a,
                     const JoinInput &input_b, const PlanNode &node_b) {
    if (global_idx < split_point) {
        return {input_a, node_a, global_idx};
    }
    return {input_b, node_b, global_idx - split_point};
}

inline BuildProbeConfig select_build_probe_side(
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

inline MatchCollectionMode determine_collection_mode(
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    size_t build_size) {

    bool needs_build = false;
    bool needs_probe = false;

    for (const auto &[col_idx, dtype] : remapped_attrs) {
        if (col_idx < build_size) {
            needs_build = true;
        } else {
            needs_probe = true;
        }

        if (needs_build && needs_probe) {
            return MatchCollectionMode::BOTH;
        }
    }

    if (needs_build && !needs_probe) {
        return MatchCollectionMode::LEFT_ONLY;
    }
    if (needs_probe && !needs_build) {
        return MatchCollectionMode::RIGHT_ONLY;
    }

    return MatchCollectionMode::BOTH;
}

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
        auto [input, node, local_idx] = resolve_input_source(
            col_idx, left_size, left_input, left_node, right_input, right_node);

        mema::column_t col;
        set_column_metadata(col, input, node, local_idx);
        results.push_back(std::move(col));
    }

    return results;
}

struct JoinSetup {
    ExecuteResult results;
    ColumnarReader columnar_reader;
    bool prepared;

    JoinSetup() : prepared(false) {}
};

inline JoinSetup
setup_join(const JoinInput &build_input, const JoinInput &probe_input,
           const PlanNode &build_node, const PlanNode &probe_node,
           const PlanNode &left_node, const PlanNode &right_node,
           const JoinInput &left_input, const JoinInput &right_input,
           const std::vector<std::tuple<size_t, DataType>> &output_attrs,
           size_t estimated_rows) {
    JoinSetup setup;

    setup.results =
        initialize_output_columns(output_attrs, left_node, right_node,
                                  left_input, right_input, estimated_rows);

    setup.prepared = false;

    return setup;
}

inline std::vector<const Column *>
collect_needed_columns(const JoinInput &input, const PlanNode &node,
                       const std::vector<bool> &needed) {
    std::vector<const Column *> columns(node.output_attrs.size(), nullptr);
    auto *table = std::get<const ColumnarTable *>(input.data);

    for (size_t i = 0; i < node.output_attrs.size(); ++i) {
        auto [actual_col_idx, _] = node.output_attrs[i];
        columns[i] = needed[i] ? &table->columns[actual_col_idx] : nullptr;
    }
    return columns;
}

inline void prepare_output_columns(
    ColumnarReader &reader, const JoinInput &build_input,
    const JoinInput &probe_input, const PlanNode &build_node,
    const PlanNode &probe_node,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    size_t build_size) {

    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();

    if (!build_is_columnar && !probe_is_columnar)
        return;

    std::vector<bool> build_needed(build_node.output_attrs.size(), false);
    std::vector<bool> probe_needed(probe_node.output_attrs.size(), false);

    for (const auto &[col_idx, dtype] : remapped_attrs) {
        if (col_idx < build_size) {
            if (build_is_columnar) {
                build_needed[col_idx] = true;
            }
        } else if (probe_is_columnar) {
            probe_needed[col_idx - build_size] = true;
        }
    }

    if (build_is_columnar) {
        reader.prepare_build(
            collect_needed_columns(build_input, build_node, build_needed));
    }

    if (probe_is_columnar) {
        reader.prepare_probe(
            collect_needed_columns(probe_input, probe_node, probe_needed));
    }
}

} // namespace Contest

/**
 * @file join_setup.h
 * @brief Join configuration and input abstraction.
 *
 * Provides JoinInput to abstract over columnar and intermediate data sources,
 * and utilities for selecting build/probe sides and preparing output columns.
 */
#pragma once

#include <data_access/columnar_reader.h>
#include <data_model/intermediate.h>
#include <data_model/plan.h>
#include <join_execution/match_collector.h>
#include <tuple>
#include <variant>
#include <vector>

/**
 * @namespace Contest::join
 * @brief JoinInput abstraction, build/probe selection, output column setup.
 */
namespace Contest::join {

using Contest::ExecuteResult;
using Contest::ExtendedResult;
using Contest::io::ColumnarReader;

/**
 * @brief Unified abstraction over columnar tables and intermediate results.
 *
 * Stores ColumnarTable* (base scans) or ExtendedResult (child joins). Node
 * provides output_attrs mapping for column resolution.
 */
struct JoinInput {
    std::variant<ExtendedResult, const ColumnarTable *> data;
    const PlanNode *node; /**< Provides output_attrs for column mapping. */
    uint8_t table_id;     /**< Source table ID for provenance tracking. */

    /** @brief True if data is columnar (base table), false if intermediate. */
    bool is_columnar() const {
        return std::holds_alternative<const ColumnarTable *>(data);
    }

    /**
     * @brief Row count for a given output column.
     * @param col_idx Index into node->output_attrs.
     */
    size_t row_count(size_t col_idx) const {
        if (is_columnar()) {
            auto *table = std::get<const ColumnarTable *>(data);
            auto [actual_col_idx, _] = node->output_attrs[col_idx];
            return table->num_rows;
        } else {
            return std::get<ExtendedResult>(data).columns[col_idx].row_count();
        }
    }

    /** @brief Number of output columns. */
    size_t output_size() const { return node->output_attrs.size(); }

    /**
     * @brief Get list of tables whose row IDs are tracked in this input.
     *
     * For columnar input: returns {table_id}.
     * For intermediate: returns the tracked table_ids from ExtendedResult.
     */
    std::vector<uint8_t> tracked_tables() const {
        if (is_columnar()) {
            return {table_id};
        }
        return std::get<ExtendedResult>(data).table_ids;
    }

    /**
     * @brief Get row ID column for a specific table.
     * @return nullptr for columnar inputs (row IDs encoded on-the-fly).
     */
    const mema::rowid_column_t *get_rowid_column(uint8_t tid) const {
        if (is_columnar())
            return nullptr;
        return std::get<ExtendedResult>(data).get_rowid_column(tid);
    }
};

/**
 * @brief Configuration for build/probe side assignment.
 *
 * Smaller table becomes build side. Contains remapped attribute indices.
 */
struct BuildProbeConfig {
    bool build_left; /**< True if left input is build side. */
    /**
     * Output attributes with indices remapped to (build_cols...,
     * probe_cols...). When build_left=false, left indices shift right by
     * build_size, right indices shift left by left_size to swap the ordering.
     */
    std::vector<std::tuple<size_t, DataType>> remapped_attrs;
    size_t build_attr; /**< Join key index in build's output_attrs. */
    size_t probe_attr; /**< Join key index in probe's output_attrs. */
};

/** @brief Resolves global output column index to source input. */
inline std::tuple<const JoinInput &, const PlanNode &, size_t>
resolve_input_source(size_t global_idx, size_t split_point,
                     const JoinInput &input_a, const PlanNode &node_a,
                     const JoinInput &input_b, const PlanNode &node_b) {
    if (global_idx < split_point) {
        return {input_a, node_a, global_idx};
    }
    return {input_b, node_b, global_idx - split_point};
}

/**
 * @brief Chooses build/probe sides based on cardinality.
 *
 * Smaller table becomes build. When flipped, remaps indices to (build, probe).
 */
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

/**
 * @brief Determines which row IDs needed based on output columns.
 *
 * Returns LEFT_ONLY, RIGHT_ONLY, or BOTH. Saves ~50% match storage when
 * only one side is projected.
 */
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

/**
 * @brief Creates output columns with provenance metadata from inputs.
 */
inline ExtendedResult initialize_output_columns(
    const std::vector<std::tuple<size_t, DataType>> &output_attrs,
    const PlanNode &left_node, const PlanNode &right_node,
    const JoinInput &left_input, const JoinInput &right_input,
    size_t estimated_rows) {
    ExtendedResult results;
    results.columns.reserve(output_attrs.size());
    size_t left_size = left_input.output_size();

    auto set_column_metadata = [](mema::column_t &col, const JoinInput &input,
                                  const PlanNode &node, size_t col_idx) {
        auto [actual_col_idx, _] = node.output_attrs[col_idx];
        if (input.is_columnar()) {
            col.source_table = input.table_id;
            col.source_column = actual_col_idx;
        } else {
            const auto &result = std::get<ExtendedResult>(input.data);
            col.source_table = result.columns[col_idx].source_table;
            col.source_column = result.columns[col_idx].source_column;
        }
    };

    for (size_t i = 0; i < output_attrs.size(); ++i) {
        auto [col_idx, _] = output_attrs[i];
        auto [input, node, local_idx] = resolve_input_source(
            col_idx, left_size, left_input, left_node, right_input, right_node);

        mema::column_t col;
        set_column_metadata(col, input, node, local_idx);
        results.columns.push_back(std::move(col));
    }

    return results;
}

/**
 * @brief Join output state and columnar reader.
 *
 * prepared flag implements lazy PageIndex construction.
 */
struct JoinSetup {
    ExtendedResult results; /**< Output columns + row ID columns. */
    ColumnarReader
        columnar_reader; /**< Page cursor caching for columnar access. */
    std::vector<uint8_t> merged_table_ids; /**< Tables tracked in output. */
    /**
     * True after prepare_output_columns called.
     */
    bool prepared;

    JoinSetup() : prepared(false) {}
};

/**
 * @brief Merge tracked table IDs from build and probe (sorted, unique).
 *
 * Both input vectors must be sorted. Output is sorted and deduplicated.
 */
inline std::vector<uint8_t>
merge_tracked_tables(const std::vector<uint8_t> &build_tables,
                     const std::vector<uint8_t> &probe_tables) {
    std::vector<uint8_t> merged;
    merged.reserve(build_tables.size() + probe_tables.size());

    size_t i = 0, j = 0;
    while (i < build_tables.size() && j < probe_tables.size()) {
        if (build_tables[i] < probe_tables[j]) {
            merged.push_back(build_tables[i++]);
        } else if (probe_tables[j] < build_tables[i]) {
            merged.push_back(probe_tables[j++]);
        } else {
            merged.push_back(build_tables[i++]);
            j++; // Skip duplicate
        }
    }
    while (i < build_tables.size())
        merged.push_back(build_tables[i++]);
    while (j < probe_tables.size())
        merged.push_back(probe_tables[j++]);

    return merged;
}

/**
 * @brief Initializes JoinSetup with output columns; call before join execution.
 *
 * PageIndex construction deferred to prepare_output_columns().
 * Computes merged table IDs from build and probe inputs.
 */
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

    // Compute merged table IDs from build and probe sides
    auto build_tables = build_input.tracked_tables();
    auto probe_tables = probe_input.tracked_tables();
    setup.merged_table_ids = merge_tracked_tables(build_tables, probe_tables);

    setup.prepared = false;

    return setup;
}

/**
 * @brief Collects Column pointers for needed output columns from columnar
 * input.
 *
 * Unused columns get nullptr to skip PageIndex construction.
 */
inline platform::ArenaVector<const Column *>
collect_needed_columns(const JoinInput &input, const PlanNode &node,
                       const platform::ArenaVector<uint8_t> &needed,
                       platform::ThreadArena &arena) {
    platform::ArenaVector<const Column *> columns(arena);
    columns.resize(node.output_attrs.size());
    std::memset(columns.data(), 0, columns.size() * sizeof(const Column *));
    auto *table = std::get<const ColumnarTable *>(input.data);

    for (size_t i = 0; i < node.output_attrs.size(); ++i) {
        auto [actual_col_idx, _] = node.output_attrs[i];
        columns[i] = needed[i] ? &table->columns[actual_col_idx] : nullptr;
    }
    return columns;
}

/**
 * @brief Prepares ColumnarReader with columns needed for materialization.
 *
 * Triggers lazy PageIndex construction only for projected columns.
 */
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

    auto &arena = Contest::platform::get_arena(0);

    platform::ArenaVector<uint8_t> build_needed(arena);
    build_needed.resize(build_node.output_attrs.size());
    std::memset(build_needed.data(), 0, build_needed.size());

    platform::ArenaVector<uint8_t> probe_needed(arena);
    probe_needed.resize(probe_node.output_attrs.size());
    std::memset(probe_needed.data(), 0, probe_needed.size());

    for (const auto &[col_idx, dtype] : remapped_attrs) {
        if (col_idx < build_size) {
            if (build_is_columnar) {
                build_needed[col_idx] = 1;
            }
        } else if (probe_is_columnar) {
            probe_needed[col_idx - build_size] = 1;
        }
    }

    if (build_is_columnar) {
        reader.prepare_build(collect_needed_columns(build_input, build_node,
                                                    build_needed, arena));
    }

    if (probe_is_columnar) {
        reader.prepare_probe(collect_needed_columns(probe_input, probe_node,
                                                    probe_needed, arena));
    }
}

} // namespace Contest::join

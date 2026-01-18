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
 * @brief Parallel hash join implementation for the SIGMOD contest.
 *
 * Key components in this file:
 * - JoinInput: Unified abstraction over columnar/intermediate sources
 * - BuildProbeConfig: Build/probe side selection with attribute remapping
 * - JoinSetup: Join execution state container with lazy initialization
 * - select_build_probe_side(): Cardinality-based build side selection
 * - prepare_output_columns(): Lazy PageIndex construction for needed columns
 */
namespace Contest::join {

// Types from Contest:: namespace
using Contest::ExecuteResult;

// Types from Contest::io:: namespace
using Contest::io::ColumnarReader;

// Note: Column, ColumnarTable, DataType, JoinNode, PAGE_SIZE, PlanNode
// are defined at global scope and accessible without qualification

/**
 * @brief Unified abstraction over columnar tables and intermediate results.
 *
 * Allows join code to handle both source types uniformly. Stores either a
 * pointer to ColumnarTable (for base table scans) or an ExecuteResult
 * (for intermediate results from child joins). The node pointer provides
 * output_attrs mapping for column resolution.
 */
struct JoinInput {
    std::variant<ExecuteResult, const ColumnarTable *> data;
    const PlanNode *node; /**< Provides output_attrs for column mapping. */
    uint8_t table_id;     /**< Source table ID for provenance tracking. */

    /** @brief True if data is columnar (base table), false if intermediate. */
    bool is_columnar() const {
        return std::holds_alternative<const ColumnarTable *>(data);
    }

    /**
     * @brief Returns the number of rows for a given output column.
     * @param col_idx Index into node->output_attrs (not physical table column).
     * @return Row count from underlying columnar table or intermediate result.
     */
    size_t row_count(size_t col_idx) const {
        if (is_columnar()) {
            auto *table = std::get<const ColumnarTable *>(data);
            auto [actual_col_idx, _] = node->output_attrs[col_idx];
            return table->num_rows;
        } else {
            return std::get<ExecuteResult>(data)[col_idx].row_count();
        }
    }

    /**
     * @brief Returns the number of output columns in this input.
     * @return Size of node->output_attrs vector.
     */
    size_t output_size() const { return node->output_attrs.size(); }
};

/**
 * @brief Configuration for build/probe side assignment.
 *
 * Determined by select_build_probe_side(): smaller table becomes build side.
 * Contains remapped attribute indices adjusted for the chosen assignment.
 *
 * When the optimizer's hint is overridden (right becomes build), remapped_attrs
 * are reordered from (left, right) to (build, probe) format to maintain uniform
 * indexing throughout join execution.
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

/**
 * @brief Resolves a global output column index to its source input.
 *
 * @param global_idx Column index in the combined output schema.
 * @param split_point Boundary between input_a and input_b columns.
 * @param input_a First input (typically left or build).
 * @param node_a PlanNode for input_a.
 * @param input_b Second input (typically right or probe).
 * @param node_b PlanNode for input_b.
 * @return Tuple of (source JoinInput, source PlanNode, local column index).
 */
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
 * Smaller table becomes build side to minimize hash table size and improve
 * cache efficiency. When the optimizer's build_left hint is overridden (right
 * is smaller), this performs index remapping to maintain uniform (build, probe)
 * ordering throughout join execution.
 *
 * Remapping algorithm when sides are flipped (!build_left):
 * - Left columns [0, left_size) shift right: new_idx = build_size + old_idx
 * - Right columns [left_size, ...) shift left: new_idx = old_idx - left_size
 * This swaps the two ranges to achieve (right_cols..., left_cols...) layout
 * which matches (build_cols..., probe_cols...) semantic ordering.
 *
 * @param join JoinNode containing left_attr and right_attr join keys.
 * @param left_input Left input providing row count for comparison.
 * @param right_input Right input providing row count for comparison.
 * @param output_attrs Output schema in original (left, right) order.
 * @return BuildProbeConfig with build_left flag, remapped attributes, and join
 * keys.
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
 * @brief Determines which row IDs are needed based on output columns.
 *
 * Analyzes remapped_attrs to check if output needs only build-side columns,
 * only probe-side columns, or both. Returns matching MatchCollectionMode
 * to avoid collecting unnecessary row IDs.
 *
 * Optimization rationale: When the query projects only columns from one side
 * (e.g., SELECT R.* FROM R JOIN S ON R.id = S.id), we can skip collecting
 * row IDs from the unused side. This saves approximately 50% of match storage
 * memory since each match pair normally stores two row IDs (build_row,
 * probe_row).
 *
 * The function short-circuits as soon as both sides are needed to minimize
 * iteration overhead.
 *
 * @param remapped_attrs Output schema with indices in (build, probe) order.
 * @param build_size Number of columns from build side (boundary point).
 * @return LEFT_ONLY if only build columns needed, RIGHT_ONLY if only probe,
 *         BOTH if columns from both sides are projected.
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
 *
 * Propagates source_table and source_column tracking through the join pipeline.
 * For columnar inputs, metadata comes directly from table_id and column
 * indices. For intermediate results, metadata is inherited from child join's
 * output.
 *
 * @param output_attrs Output schema in original (left, right) order.
 * @param left_node Left child's PlanNode for column resolution.
 * @param right_node Right child's PlanNode for column resolution.
 * @param left_input Left input data and metadata.
 * @param right_input Right input data and metadata.
 * @param estimated_rows Unused parameter (reserved for future capacity hints).
 * @return ExecuteResult with empty columns but correct provenance metadata.
 */
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

/**
 * @brief Encapsulates join output state and columnar reader.
 *
 * Passed through join execution to accumulate results and provide
 * access to columnar data for materialization.
 *
 * The prepared flag implements lazy PageIndex construction: ColumnarReader's
 * PageIndex structures are only built when prepare_output_columns() is called,
 * avoiding upfront cost for joins that may produce zero matches. This is
 * particularly beneficial for selective joins where early termination is
 * common.
 */
struct JoinSetup {
    ExecuteResult results; /**< Output columns being populated. */
    ColumnarReader
        columnar_reader; /**< Page cursor caching for columnar access. */
    /**
     * True after prepare_output_columns called. Guards against double
     * preparation and signals that columnar_reader is ready for
     * materialization.
     */
    bool prepared;

    JoinSetup() : prepared(false) {}
};

/**
 * @brief Initializes JoinSetup with output columns; call before join execution.
 *
 * Creates the output container with proper provenance tracking but leaves
 * columnar_reader unprepared (prepared=false). Actual PageIndex construction
 * is deferred to prepare_output_columns() to avoid initialization cost when
 * joins produce no matches.
 *
 * @param build_input Build side input (after select_build_probe_side).
 * @param probe_input Probe side input (after select_build_probe_side).
 * @param build_node Build side PlanNode.
 * @param probe_node Probe side PlanNode.
 * @param left_node Original left PlanNode (for provenance).
 * @param right_node Original right PlanNode (for provenance).
 * @param left_input Original left input (for provenance).
 * @param right_input Original right input (for provenance).
 * @param output_attrs Output schema in original (left, right) order.
 * @param estimated_rows Unused (reserved for future optimizations).
 * @return JoinSetup with initialized results and prepared=false.
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

    setup.prepared = false;

    return setup;
}

/**
 * @brief Collects Column pointers for needed output columns from columnar
 * input.
 *
 * Maps logical output indices to physical table Column pointers based on the
 * needed mask. Unused columns get nullptr to signal that ColumnarReader should
 * skip building PageIndex structures for them.
 *
 * @param input Columnar JoinInput (caller must ensure is_columnar() is true).
 * @param node PlanNode providing output_attrs mapping.
 * @param needed Mask indicating which output columns are actually projected.
 * @return Vector of Column pointers, nullptr for unneeded columns.
 */
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

/**
 * @brief Prepares ColumnarReader with columns needed for materialization.
 *
 * Analyzes remapped_attrs to determine which columns from build/probe sides
 * are needed, then initializes the reader with those Column pointers. This
 * triggers lazy construction of PageIndex structures only for columns that
 * appear in the output projection.
 *
 * Early returns if both inputs are intermediate (no columnar data to index).
 * For hybrid joins (one columnar, one intermediate), only the columnar side
 * gets prepared.
 *
 * @param reader ColumnarReader to populate with build/probe Column pointers.
 * @param build_input Build side input.
 * @param probe_input Probe side input.
 * @param build_node Build side PlanNode.
 * @param probe_node Probe side PlanNode.
 * @param remapped_attrs Output schema in (build, probe) order.
 * @param build_size Number of build columns (split point for remapped_attrs).
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

} // namespace Contest::join

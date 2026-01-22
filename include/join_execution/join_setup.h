/**
 * @file join_setup.h
 * @brief Join configuration and build/probe side selection.
 *
 * Provides utilities for selecting build/probe sides and determining
 * which row IDs to collect based on output columns.
 */
#pragma once

#include <data_model/intermediate.h>
#include <data_model/plan.h>
#include <join_execution/match_collector.h>
#include <tuple>
#include <vector>

/**
 * @namespace Contest::join
 * @brief Build/probe selection and collection mode determination.
 */
namespace Contest::join {

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

} // namespace Contest::join

namespace Contest {

// Forward declare AnalyzedJoinNode
struct AnalyzedJoinNode;

/**
 * @brief Tracking info for one side of a join (build or probe).
 *
 * Determines whether to embed base table row IDs or IR indices in the
 * output tuples for this side.
 */
struct SideTrackingInfo {
    bool track_base_rows =
        false; ///< True to embed base row IDs, false for IR indices
    uint8_t base_table_id = 0; ///< Base table to track (if track_base_rows)
};

/**
 * @brief Tracking configuration for intermediate construction.
 *
 * Determines what row IDs to embed in join key tuples and whether
 * DeferredTables are needed for non-tracked sides.
 */
struct TupleTrackingInfo {
    SideTrackingInfo build_tracking; ///< Tracking info for build side
    SideTrackingInfo probe_tracking; ///< Tracking info for probe side
    bool key_from_build =
        true; ///< True if parent join key comes from build side
};

/**
 * @brief Result of a join execution before intermediate construction.
 *
 * Contains match buffers and metadata needed for deferred IR construction.
 * Allows parent join to decide row ID format based on its cardinality
 * requirements before constructing the intermediate result.
 *
 * @tparam Mode Match collection mode for this join's buffers.
 */
template <join::MatchCollectionMode Mode> struct MatchResult {
    std::vector<join::ThreadLocalMatchBuffer<Mode>> buffers;
    size_t total_count = 0;

    /// The inputs that were joined (for resolving row IDs during IR
    /// construction)
    JoinInput build_input;
    JoinInput probe_input;

    /// Join configuration
    const AnalyzedJoinNode *join_node = nullptr;
    join::BuildProbeConfig config;

    /// Convenience accessors
    size_t count() const { return total_count; }
    bool empty() const { return total_count == 0; }
};

} // namespace Contest

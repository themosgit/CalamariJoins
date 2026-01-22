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

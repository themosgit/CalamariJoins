/**
 * @file deferred_plan.h
 * @brief Analyzed plan with materialization decisions for deferred execution.
 *
 * DeferredPlan mirrors the original Plan structure but includes pre-computed
 * decisions about which columns to materialize eagerly (join keys) vs defer
 * until final output. Each DeferredJoinNode tracks column provenance back to
 * base tables for efficient deferred resolution.
 *
 * @see analyze_plan.cpp for the analysis algorithm.
 * @see deferred_intermediate.h for the runtime result format.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <data_model/plan.h>
#include <join_execution/match_collector.h>

namespace Contest {

/**
 * @brief Materialization decision for an output column.
 *
 * MATERIALIZE: Column is needed as a join key by parent - materialize eagerly.
 * DEFER: Column only needed at final output - defer until root materialization.
 */
enum class ColumnResolution : uint8_t { MATERIALIZE, DEFER };

/**
 * @brief Tracks the base table origin of a column for deferred resolution.
 *
 * Used to resolve deferred columns at final materialization by looking up
 * the original value in the base table using row ID provenance.
 */
struct ColumnProvenance {
    uint8_t base_table_id;   ///< Index into Plan::inputs.
    uint8_t base_column_idx; ///< Column index within the base table.
};

/**
 * @brief Complete metadata for an output column in a deferred join.
 *
 * Combines materialization decision, provenance tracking, and child source
 * information for efficient intermediate construction and final resolution.
 */
struct DeferredColumnInfo {
    size_t original_idx; ///< Index in node's output_attrs.
    DataType type;       ///< INT32 or VARCHAR.

    ColumnResolution resolution; ///< MATERIALIZE or DEFER.
    ColumnProvenance provenance; ///< Base table source for deferred resolution.

    bool from_left;          ///< True if from left child, false if right.
    size_t child_output_idx; ///< Index in child's output_attrs.
};

/**
 * @brief Analyzed scan node for deferred execution.
 *
 * Wraps a ScanNode with output attribute information.
 */
struct DeferredScanNode {
    size_t node_idx;       ///< Index in original Plan::nodes.
    uint8_t base_table_id; ///< Index into Plan::inputs.
    std::vector<std::tuple<size_t, DataType>> output_attrs; ///< Projected cols.
};

/**
 * @brief Analyzed join node with pre-computed materialization decisions.
 *
 * Contains all information needed for deferred execution:
 * - Which columns to materialize eagerly (join keys for parent)
 * - Column provenance for deferred resolution
 * - Pre-computed match collection mode
 * - Table IDs tracked through this node
 */
struct DeferredJoinNode {
    size_t node_idx; ///< Index in original Plan::nodes.

    size_t left_child_idx;  ///< Left child index in Plan::nodes.
    size_t right_child_idx; ///< Right child index in Plan::nodes.
    size_t left_join_attr;  ///< Join key index in left child's output.
    size_t right_join_attr; ///< Join key index in right child's output.

    /// Original output attributes (global indexing).
    std::vector<std::tuple<size_t, DataType>> output_attrs;

    /// Per-column materialization decisions and provenance.
    std::vector<DeferredColumnInfo> columns;

    /// Pre-computed collection mode (assumes build=left; flip if build=right).
    join::MatchCollectionMode base_collection_mode;

    /// Sorted table IDs tracked through this node (union of children).
    std::vector<uint8_t> tracked_table_ids;

    /// Column index that parent needs as join key (nullopt if root).
    std::optional<size_t> parent_join_key_idx;

    /// True if this is the root node.
    bool is_root;
};

/**
 * @brief Plan node variant for deferred execution.
 */
using DeferredNode = std::variant<DeferredScanNode, DeferredJoinNode>;

/**
 * @brief Analyzed plan with materialization decisions.
 *
 * Mirrors Plan structure but includes pre-computed decisions for deferred
 * materialization. The original_plan pointer provides access to base tables
 * for value resolution.
 */
struct DeferredPlan {
    std::vector<DeferredNode> nodes; ///< Analyzed nodes (same indices as Plan).
    size_t root;                     ///< Root node index.
    const Plan *original_plan;       ///< Non-owning reference to original plan.

    const DeferredNode &operator[](size_t idx) const { return nodes[idx]; }
};

/**
 * @brief Analyze plan and compute materialization decisions.
 *
 * Walks the plan tree in post-order, determining for each join node:
 * 1. Which column the parent needs as join key (MATERIALIZE)
 * 2. All other columns (DEFER)
 * 3. Provenance for each column back to base table
 * 4. Pre-computed collection mode based on output columns
 *
 * @param plan Original query plan.
 * @return DeferredPlan with materialization decisions.
 */
DeferredPlan analyze_plan(const Plan &plan);

} // namespace Contest

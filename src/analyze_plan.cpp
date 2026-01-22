/**
 * @file analyze_plan.cpp
 * @brief Analyzes query plan and computes materialization decisions.
 *
 * Walks the plan tree in post-order to determine which columns should be
 * materialized eagerly (join keys needed by parent) vs deferred until final
 * output. Traces column provenance back to base tables for deferred resolution.
 *
 * @see deferred_plan.h for AnalyzedPlan structure.
 */
#include <functional>
#include <unordered_map>

#include <data_model/deferred_plan.h>

namespace Contest {

namespace {

/**
 * @brief Parent relationship info for a node.
 */
struct ParentInfo {
    size_t parent_idx;  ///< Parent node index in Plan::nodes.
    bool is_left_child; ///< True if this node is parent's left child.
};

/**
 * @brief Build map of node_idx → parent info.
 *
 * Root node will not have an entry in the map.
 */
std::unordered_map<size_t, ParentInfo> build_parent_map(const Plan &plan) {
    std::unordered_map<size_t, ParentInfo> parent_map;

    for (size_t i = 0; i < plan.nodes.size(); ++i) {
        const auto &node = plan.nodes[i];
        if (const auto *join = std::get_if<JoinNode>(&node.data)) {
            parent_map[join->left] = {i, true};
            parent_map[join->right] = {i, false};
        }
    }
    return parent_map;
}

/**
 * @brief Trace column provenance to base table.
 *
 * Recursively follows column through join nodes until reaching a scan node.
 *
 * @param plan Original query plan.
 * @param node_idx Current node index.
 * @param column_idx Column index in node's output_attrs.
 * @return ColumnProvenance with base table ID and column index.
 */
ColumnProvenance trace_provenance(const Plan &plan, size_t node_idx,
                                  size_t column_idx) {
    const auto &node = plan.nodes[node_idx];

    if (const auto *scan = std::get_if<ScanNode>(&node.data)) {
        // Base case: column comes directly from scan
        auto [actual_col_idx, _] = node.output_attrs[column_idx];
        return ColumnProvenance{static_cast<uint8_t>(scan->base_table_id),
                                static_cast<uint8_t>(actual_col_idx)};
    }

    // Join node: determine which child the column comes from
    const auto &join = std::get<JoinNode>(node.data);
    const auto &left_node = plan.nodes[join.left];
    size_t left_size = left_node.output_attrs.size();

    auto [col_idx, _] = node.output_attrs[column_idx];

    if (col_idx < left_size) {
        // Column from left child
        return trace_provenance(plan, join.left, col_idx);
    } else {
        // Column from right child
        return trace_provenance(plan, join.right, col_idx - left_size);
    }
}

/**
 * @brief Find which column index in this node the parent needs as join key.
 *
 * @param plan Original query plan.
 * @param node_idx Current node index.
 * @param parent_map Map of node → parent relationship.
 * @return Column index parent uses as join key, or nullopt if root.
 */
std::optional<size_t>
find_parent_join_key(const Plan &plan, size_t node_idx,
                     const std::unordered_map<size_t, ParentInfo> &parent_map) {
    auto it = parent_map.find(node_idx);
    if (it == parent_map.end()) {
        return std::nullopt; // Root node
    }

    const auto &parent_node = plan.nodes[it->second.parent_idx];
    const auto &parent_join = std::get<JoinNode>(parent_node.data);

    // Parent's join key for this child
    return it->second.is_left_child ? parent_join.left_attr
                                    : parent_join.right_attr;
}

/**
 * @brief Compute base collection mode based on which sides have output columns.
 *
 * Assumes build=left. If build=right at runtime, caller flips
 * LEFT_ONLY/RIGHT_ONLY.
 */
join::MatchCollectionMode
compute_base_collection_mode(const std::vector<AnalyzedColumnInfo> &columns,
                             size_t left_output_size) {
    bool needs_left = false;
    bool needs_right = false;

    for (const auto &col : columns) {
        if (col.from_left) {
            needs_left = true;
        } else {
            needs_right = true;
        }
        if (needs_left && needs_right) {
            return join::MatchCollectionMode::BOTH;
        }
    }

    if (needs_left && !needs_right)
        return join::MatchCollectionMode::LEFT_ONLY;
    if (needs_right && !needs_left)
        return join::MatchCollectionMode::RIGHT_ONLY;
    return join::MatchCollectionMode::BOTH;
}

} // anonymous namespace

AnalyzedPlan analyze_plan(const Plan &plan) {
    AnalyzedPlan analyzed;
    analyzed.original_plan = &plan;
    analyzed.nodes.resize(plan.nodes.size());
    analyzed.root = plan.root;

    auto parent_map = build_parent_map(plan);

    // Build post-order traversal (children before parents)
    std::vector<size_t> post_order;
    post_order.reserve(plan.nodes.size());
    std::vector<bool> visited(plan.nodes.size(), false);

    std::function<void(size_t)> visit = [&](size_t idx) {
        if (visited[idx])
            return;
        visited[idx] = true;

        const auto &node = plan.nodes[idx];
        if (const auto *join = std::get_if<JoinNode>(&node.data)) {
            visit(join->left);
            visit(join->right);
        }
        post_order.push_back(idx);
    };
    visit(plan.root);

    // PASS 1: Build structure and initial materialization decisions
    for (size_t node_idx : post_order) {
        const auto &node = plan.nodes[node_idx];

        if (const auto *scan = std::get_if<ScanNode>(&node.data)) {
            // Scan node: simple wrapper
            AnalyzedScanNode ascan;
            ascan.node_idx = node_idx;
            ascan.base_table_id = scan->base_table_id;
            ascan.output_attrs = node.output_attrs;
            analyzed.nodes[node_idx] = std::move(ascan);

        } else {
            // Join node: compute materialization decisions
            const auto &join = std::get<JoinNode>(node.data);
            AnalyzedJoinNode ajoin;
            ajoin.node_idx = node_idx;
            ajoin.left_child_idx = join.left;
            ajoin.right_child_idx = join.right;
            ajoin.left_join_attr = join.left_attr;
            ajoin.right_join_attr = join.right_attr;
            ajoin.output_attrs = node.output_attrs;
            ajoin.is_root = (node_idx == plan.root);

            // Find which column parent needs as join key
            ajoin.parent_join_key_idx =
                find_parent_join_key(plan, node_idx, parent_map);

            // Get child sizes for determining column source
            const auto &left_node = plan.nodes[join.left];
            size_t left_size = left_node.output_attrs.size();

            // Build column info for each output column
            for (size_t i = 0; i < node.output_attrs.size(); ++i) {
                auto [col_idx, col_type] = node.output_attrs[i];

                AnalyzedColumnInfo info;
                info.original_idx = i;
                info.type = col_type;

                // Determine if column is from left or right child
                // col_idx is the combined L+R index:
                // - [0, left_size) = position in left child's output
                // - [left_size, ...) = position in right child's output +
                // left_size
                if (col_idx < left_size) {
                    info.from_left = true;
                    info.child_output_idx = col_idx;
                } else {
                    info.from_left = false;
                    info.child_output_idx = col_idx - left_size;
                }

                // Materialization decision:
                // - At root: ALL columns must be materialized (final output)
                // - At intermediate: only parent's join key is materialized
                if (ajoin.is_root) {
                    // Root node: materialize everything
                    info.resolution = ColumnResolution::MATERIALIZE;
                } else if (ajoin.parent_join_key_idx.has_value() &&
                           i == *ajoin.parent_join_key_idx) {
                    info.resolution = ColumnResolution::MATERIALIZE;
                } else {
                    info.resolution = ColumnResolution::DEFER;
                }

                // Trace provenance to base table
                info.provenance = trace_provenance(plan, node_idx, i);

                ajoin.columns.push_back(std::move(info));
            }

            // Compute collection mode and count deferred columns
            ajoin.base_collection_mode =
                compute_base_collection_mode(ajoin.columns, left_size);

            // Count deferred columns for pre-allocation
            ajoin.num_deferred_columns = 0;
            for (const auto &col : ajoin.columns) {
                if (col.resolution == ColumnResolution::DEFER) {
                    ++ajoin.num_deferred_columns;
                }
            }

            analyzed.nodes[node_idx] = std::move(ajoin);
        }
    }

    // PASS 2: Propagate materialization requirements to children
    // Process in reverse post-order (parents before children)
    for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
        size_t node_idx = *it;
        auto *ajoin = std::get_if<AnalyzedJoinNode>(&analyzed.nodes[node_idx]);
        if (!ajoin)
            continue;

        // For each column that must be MATERIALIZE, ensure the child also
        // materializes it
        for (const auto &col : ajoin->columns) {
            if (col.resolution != ColumnResolution::MATERIALIZE)
                continue;

            // Find which child this column comes from
            size_t child_idx =
                col.from_left ? ajoin->left_child_idx : ajoin->right_child_idx;

            auto *child_ajoin =
                std::get_if<AnalyzedJoinNode>(&analyzed.nodes[child_idx]);
            if (!child_ajoin)
                continue; // Child is a scan - always has data

            // Mark child's column as MATERIALIZE
            if (col.child_output_idx < child_ajoin->columns.size()) {
                child_ajoin->columns[col.child_output_idx].resolution =
                    ColumnResolution::MATERIALIZE;
            }
        }
    }

    // PASS 3: Recount num_deferred_columns after propagation
    for (size_t node_idx : post_order) {
        auto *ajoin = std::get_if<AnalyzedJoinNode>(&analyzed.nodes[node_idx]);
        if (!ajoin)
            continue;

        ajoin->num_deferred_columns = 0;
        for (const auto &col : ajoin->columns) {
            if (col.resolution == ColumnResolution::DEFER) {
                ++ajoin->num_deferred_columns;
            }
        }
    }

    return analyzed;
}

} // namespace Contest

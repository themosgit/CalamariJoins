/**
 * @file deferred_intermediate.h
 * @brief Lightweight intermediate result for deferred materialization.
 *
 * DeferredResult stores only materialized columns (join keys) plus row ID
 * provenance columns. Deferred columns are resolved at final materialization
 * by following row IDs back to base tables.
 *
 * @see deferred_plan.h for DeferredJoinNode with column decisions.
 * @see construct_deferred.h for building DeferredResult.
 * @see materialize_deferred.h for final resolution.
 */
#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <data_model/deferred_plan.h>
#include <data_model/intermediate.h>

namespace Contest {

/**
 * @brief Lightweight intermediate result with only join keys materialized.
 *
 * Unlike ExtendedResult which stores all projected columns, DeferredResult
 * stores only columns marked MATERIALIZE (typically just the parent's join
 * key). All other columns are resolved at final materialization using row ID
 * provenance.
 *
 * Memory savings: For a join projecting N columns where only 1 is a join key,
 * DeferredResult uses ~1/N the memory of ExtendedResult for data columns.
 *
 * @see DeferredColumnInfo for materialization decisions.
 * @see DeferredJoinNode for column provenance tracking.
 */
struct DeferredResult {
    /// Only columns marked MATERIALIZE (typically 1 join key).
    std::vector<mema::column_t> materialized;

    /// Map: original column index → index in materialized (nullopt if
    /// deferred).
    std::vector<std::optional<size_t>> materialized_map;

    /// Row ID tracking for provenance (same as ExtendedResult).
    std::vector<mema::rowid_column_t> row_ids;

    /// Which base tables are tracked (sorted).
    std::vector<uint8_t> table_ids;

    /// Reference to node info for column provenance resolution.
    const DeferredJoinNode *node_info = nullptr;

    /// Total row count.
    size_t num_rows = 0;

    DeferredResult() = default;
    DeferredResult(DeferredResult &&) = default;
    DeferredResult &operator=(DeferredResult &&) = default;
    DeferredResult(const DeferredResult &) = delete;
    DeferredResult &operator=(const DeferredResult &) = delete;

    /** @brief Total row count. */
    size_t row_count() const { return num_rows; }

    /** @brief Check if column was materialized (not deferred). */
    bool is_materialized(size_t orig_idx) const {
        return orig_idx < materialized_map.size() &&
               materialized_map[orig_idx].has_value();
    }

    /** @brief Get materialized column, or nullptr if deferred. */
    const mema::column_t *get_materialized(size_t orig_idx) const {
        if (!is_materialized(orig_idx))
            return nullptr;
        return &materialized[*materialized_map[orig_idx]];
    }

    /** @brief Find row ID column index for a table, or -1 if not found. */
    int find_rowid_index(uint8_t tid) const {
        for (size_t i = 0; i < table_ids.size(); ++i) {
            if (table_ids[i] == tid)
                return static_cast<int>(i);
        }
        return -1;
    }

    /** @brief Get row ID column for a table, or nullptr if not found. */
    const mema::rowid_column_t *get_rowid_column(uint8_t tid) const {
        int idx = find_rowid_index(tid);
        return (idx >= 0) ? &row_ids[idx] : nullptr;
    }

    /** @brief Get mutable row ID column for a table, or nullptr. */
    mema::rowid_column_t *get_rowid_column_mut(uint8_t tid) {
        int idx = find_rowid_index(tid);
        return (idx >= 0) ? &row_ids[idx] : nullptr;
    }
};

/**
 * @brief Input abstraction for deferred execution path.
 *
 * Similar to JoinInput but works with DeferredResult instead of ExtendedResult.
 * Provides uniform interface for columnar (base table) and deferred
 * intermediate data sources.
 */
struct DeferredInput {
    /// Either base table pointer or owned DeferredResult.
    std::variant<const ColumnarTable *, DeferredResult> data;

    /// Original plan node for output_attrs mapping.
    const PlanNode *node = nullptr;

    /// Deferred plan node for materialization decisions.
    const DeferredNode *deferred_node = nullptr;

    /// Base table ID (for columnar inputs).
    uint8_t table_id = 0;

    /** @brief True if data is columnar (base table). */
    bool is_columnar() const {
        return std::holds_alternative<const ColumnarTable *>(data);
    }

    /** @brief Row count for join key column. */
    size_t row_count(size_t col_idx) const {
        if (is_columnar()) {
            const auto *table = std::get<const ColumnarTable *>(data);
            return table->num_rows;
        }
        return std::get<DeferredResult>(data).row_count();
    }

    /** @brief Total row count. */
    size_t row_count() const {
        if (is_columnar()) {
            const auto *table = std::get<const ColumnarTable *>(data);
            return table->num_rows;
        }
        return std::get<DeferredResult>(data).row_count();
    }

    /** @brief Number of output columns. */
    size_t output_size() const {
        if (node)
            return node->output_attrs.size();
        return 0;
    }

    /** @brief Get list of tracked table IDs. */
    std::vector<uint8_t> tracked_tables() const {
        if (is_columnar()) {
            return {table_id};
        }
        return std::get<DeferredResult>(data).table_ids;
    }

    /** @brief Get row ID column for a table. */
    const mema::rowid_column_t *get_rowid_column(uint8_t tid) const {
        if (is_columnar())
            return nullptr;
        return std::get<DeferredResult>(data).get_rowid_column(tid);
    }
};

} // namespace Contest

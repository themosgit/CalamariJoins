/**
 * @file deferred_intermediate.h
 * @brief Lightweight intermediate result for deferred materialization.
 *
 * DeferredResult stores only materialized columns (join keys) plus
 * per-deferred-column provenance using 64-bit encoding (table_id, column_idx,
 * row_id). Deferred columns are resolved at final materialization by decoding
 * the provenance and reading directly from base tables.
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
 * key). All other columns are resolved at final materialization using
 * per-column 64-bit provenance (table_id, column_idx, row_id).
 *
 * Memory savings: For a join projecting N columns where only 1 is a join key,
 * DeferredResult uses ~1/N the memory of ExtendedResult for data columns.
 * Additionally, we only track provenance for deferred columns (not all tables).
 *
 * @see DeferredColumnInfo for materialization decisions.
 * @see DeferredProvenance for 64-bit encoding scheme.
 */
struct DeferredResult {
    /// Only columns marked MATERIALIZE (typically 1 join key).
    std::vector<mema::column_t> materialized;

    /// Map: original column index → index in materialized (nullopt if
    /// deferred).
    std::vector<std::optional<size_t>> materialized_map;

    /// Per-deferred-column provenance (64-bit encoded table_id+column_idx+row).
    /// One deferred_column_t per DEFER column, stores full provenance per row.
    std::vector<mema::deferred_column_t> deferred_columns;

    /// Map: original column index → index in deferred_columns (nullopt if
    /// materialized).
    std::vector<std::optional<size_t>> deferred_map;

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

    /** @brief Check if column is deferred. */
    bool is_deferred(size_t orig_idx) const {
        return orig_idx < deferred_map.size() &&
               deferred_map[orig_idx].has_value();
    }

    /** @brief Get materialized column, or nullptr if deferred. */
    const mema::column_t *get_materialized(size_t orig_idx) const {
        if (!is_materialized(orig_idx))
            return nullptr;
        return &materialized[*materialized_map[orig_idx]];
    }

    /** @brief Get deferred column provenance, or nullptr if materialized. */
    const mema::deferred_column_t *get_deferred(size_t orig_idx) const {
        if (!is_deferred(orig_idx))
            return nullptr;
        return &deferred_columns[*deferred_map[orig_idx]];
    }

    /** @brief Get mutable deferred column provenance, or nullptr. */
    mema::deferred_column_t *get_deferred_mut(size_t orig_idx) {
        if (!is_deferred(orig_idx))
            return nullptr;
        return &deferred_columns[*deferred_map[orig_idx]];
    }

    /** @brief Number of deferred columns. */
    size_t num_deferred() const { return deferred_columns.size(); }
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

    /**
     * @brief Get deferred column provenance for a column index.
     *
     * For columnar inputs, returns nullptr (caller must encode fresh).
     * For DeferredResult inputs, returns existing provenance column.
     */
    const mema::deferred_column_t *get_deferred_column(size_t col_idx) const {
        if (is_columnar())
            return nullptr;
        return std::get<DeferredResult>(data).get_deferred(col_idx);
    }
};

} // namespace Contest

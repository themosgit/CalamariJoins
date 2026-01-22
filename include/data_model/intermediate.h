/**
 * @file intermediate.h
 * @brief Intermediate join result types and input abstraction.
 *
 * Provides:
 * - mema::value_t: 4-byte value encoding (INT32 direct, VARCHAR as page/offset)
 * - mema::column_t: 16KB-paged column for materialized values
 * - mema::DeferredTable: 16KB-paged 32-bit row ID storage per base table
 * - IntermediateResult: Lightweight result with selective materialization
 * - JoinInput: Unified abstraction over columnar tables and intermediate
 * results
 *
 * Base tables must outlive execution.
 *
 * @see plan.h for ColumnarTable, construct_intermediate.h for building results.
 * @see deferred_plan.h for AnalyzedJoinNode with column decisions.
 */
#pragma once

#include <cstdint>
#include <data_access/table.h>
#include <data_model/deferred_plan.h>
#include <data_model/plan.h>
#include <foundation/common.h>
#include <optional>
#include <platform/arena.h>
#include <variant>
#include <vector>

/**
 * @namespace mema
 * @brief Compact join intermediate: value_t (4B) + column_t (16KB pages) +
 * DeferredTable (32-bit row IDs).
 *
 * value_t: INT32 direct or VARCHAR page/offset ref. column_t: arena-allocated
 * pages with write_at(). DeferredTable: 32-bit row ID storage per base table.
 *
 * @see Contest::IntermediateResult, plan.h ColumnarTable.
 */
namespace mema {

/**
 * @brief 4-byte value: INT32 direct, VARCHAR packed (19-bit page + 13-bit
 * offset).
 *
 * NULL = INT32_MIN, long string offset = 0x1FFF. Refs valid only while
 * source exists.
 */
struct alignas(4) value_t {
    int32_t value;

    /** @brief Encode VARCHAR as page/offset. */
    static inline value_t encode_string(int32_t page_idx, int32_t offset_idx) {
        return {(offset_idx << 19) | (page_idx & 0x7FFFF)};
    }

    /** @brief Decode VARCHAR to page/offset. */
    static inline void decode_string(int32_t encoded, int32_t &page_idx,
                                     int32_t &offset_idx) {

        page_idx = encoded & 0x7FFFF;
        offset_idx = (static_cast<uint32_t>(encoded) >> 19) & 0x1FFF;
    }

    /** @brief Sentinel for long strings. */
    static constexpr int32_t LONG_STRING_OFFSET = 0x1FFF;

    /** @brief NULL sentinel for both types. */
    static constexpr int32_t NULL_VALUE = INT32_MIN;

    /** @brief Check if this value represents NULL. */
    inline bool is_null() const { return value == NULL_VALUE; }
};

/**
 * @brief Page size for intermediate results (16KB, larger than ColumnarTable).
 */
constexpr size_t IR_PAGE_SIZE = 1 << 14;

/** @brief Number of value_t entries per intermediate page. */
constexpr size_t CAP_PER_PAGE = IR_PAGE_SIZE / sizeof(value_t);

/**
 * @brief Intermediate column: 16KB pages of value_t, no header.
 *
 * All pages are arena-allocated (memory freed on arena reset between queries).
 * write_at() is thread-safe after pages are set up; source_table/column track
 * VARCHAR provenance. Move-only.
 *
 * @see construct_intermediate.h, plan.h, materialize.h
 */
struct column_t {
  public:
    /** @brief Intermediate page: fixed array of value_t entries. */
    struct alignas(IR_PAGE_SIZE) Page {
        value_t data[CAP_PER_PAGE];
    };

  private:
    size_t num_values = 0; /**< Total value count across all pages. */

  public:
    std::vector<Page *> pages; /**< Pointers to arena-allocated pages. */

    /** @brief Base table index for VARCHAR dereferencing. */
    uint8_t source_table = 0;

    /** @brief Column index within source table. */
    uint8_t source_column = 0;

  public:
    column_t() = default;

    column_t(column_t &&other) noexcept
        : num_values(other.num_values), pages(std::move(other.pages)),
          source_table(other.source_table), source_column(other.source_column) {
        other.pages.clear();
        other.num_values = 0;
    }

    column_t &operator=(column_t &&other) noexcept {
        if (this != &other) {
            num_values = other.num_values;
            pages = std::move(other.pages);
            source_table = other.source_table;
            source_column = other.source_column;

            other.pages.clear();
            other.num_values = 0;
        }
        return *this;
    }

    column_t(const column_t &) = delete;
    column_t &operator=(const column_t &) = delete;

    ~column_t() = default;

    /**
     * @brief O(1) read: idx>>12 for page, idx&0xFFF for offset.
     * @note No bounds check.
     */
    inline const value_t &operator[](size_t idx) const {
        return pages[idx >> 12]->data[idx & 0xFFF];
    }

    /** @brief Total value count. */
    size_t row_count() const { return num_values; }

    /** @brief Pre-allocate pages from arena. */
    inline void pre_allocate_from_arena(Contest::platform::ThreadArena &arena,
                                        size_t count) {
        static_assert(sizeof(Page) ==
                          Contest::platform::ChunkSize<
                              Contest::platform::ChunkType::IR_PAGE>::value,
                      "Page size mismatch with IR_PAGE chunk size");
        size_t pages_needed = (count + CAP_PER_PAGE - 1) / CAP_PER_PAGE;
        pages.reserve(pages_needed);
        for (size_t i = 0; i < pages_needed; ++i) {
            void *ptr =
                arena.alloc_chunk<Contest::platform::ChunkType::IR_PAGE>();
            pages.push_back(reinterpret_cast<Page *>(ptr));
        }
        num_values = count;
    }

    /** @brief Set row count without allocation (for assembly pattern). */
    inline void set_row_count(size_t count) { num_values = count; }

    /** @brief Thread-safe write at idx (requires pages to be set up first). */
    inline void write_at(size_t idx, const value_t &val) {
        pages[idx >> 12]->data[idx & 0xFFF] = val;
    }
};

/** @brief Alias for a collection of intermediate columns. */
using Columnar = std::vector<column_t>;

/**
 * @brief Per-base-table deferred row ID storage with multi-column tracking.
 *
 * Stores 32-bit row IDs for a single base table. All columns from this
 * base table share the same row ID lookup, reducing memory from 8 bytes
 * per column to 4 bytes per table.
 *
 * Uses 16KB pages (reuses IR_PAGE arena chunk) with 4096 uint32_t entries.
 */
struct DeferredTable {
    static constexpr size_t PAGE_SIZE = 1 << 14; // 16KB
    static constexpr size_t ENTRIES_PER_PAGE =
        PAGE_SIZE / sizeof(uint32_t);         // 4096
    static constexpr size_t ENTRY_SHIFT = 12; // log2(4096)
    static constexpr size_t ENTRY_MASK = ENTRIES_PER_PAGE - 1;

    struct alignas(PAGE_SIZE) Page {
        uint32_t data[ENTRIES_PER_PAGE];
    };

    std::vector<Page *> pages;
    size_t num_values = 0;

    /// Base table ID this deferred table references
    uint8_t base_table_id = 0;

    /// True if this deferred table comes from build side (vs probe)
    bool from_build = false;

    /// Column indices from this base table that need deferred resolution
    std::vector<uint8_t> column_indices;

    DeferredTable() = default;

    DeferredTable(DeferredTable &&other) noexcept
        : pages(std::move(other.pages)), num_values(other.num_values),
          base_table_id(other.base_table_id), from_build(other.from_build),
          column_indices(std::move(other.column_indices)) {
        other.pages.clear();
        other.num_values = 0;
    }

    DeferredTable &operator=(DeferredTable &&other) noexcept {
        if (this != &other) {
            pages = std::move(other.pages);
            num_values = other.num_values;
            base_table_id = other.base_table_id;
            from_build = other.from_build;
            column_indices = std::move(other.column_indices);
            other.pages.clear();
            other.num_values = 0;
        }
        return *this;
    }

    DeferredTable(const DeferredTable &) = delete;
    DeferredTable &operator=(const DeferredTable &) = delete;

    ~DeferredTable() = default;

    /// O(1) read: idx >> 12 for page, idx & 0xFFF for offset
    inline uint32_t operator[](size_t idx) const {
        return pages[idx >> ENTRY_SHIFT]->data[idx & ENTRY_MASK];
    }

    /// Thread-safe write at idx (requires pages set up first)
    inline void write_at(size_t idx, uint32_t row_id) {
        pages[idx >> ENTRY_SHIFT]->data[idx & ENTRY_MASK] = row_id;
    }

    size_t row_count() const { return num_values; }
    void set_row_count(size_t count) { num_values = count; }

    /// Check if this table tracks a specific base column
    bool has_column(uint8_t col_idx) const {
        for (uint8_t c : column_indices) {
            if (c == col_idx)
                return true;
        }
        return false;
    }
};

} // namespace mema

namespace Contest {

/**
 * @brief Reference from a column to its deferred table.
 */
struct DeferredColumnRef {
    uint8_t table_idx; ///< Index into IntermediateResult::deferred_tables
    uint8_t base_col;  ///< Base column index in Plan::inputs[base_table_id]
};

/**
 * @brief Lightweight intermediate result with selective materialization.
 *
 * Stores only columns marked MATERIALIZE (typically just the parent's join
 * key). Deferred columns use per-table 32-bit row ID storage instead of
 * per-column 64-bit provenance, achieving up to 10x memory reduction for
 * multi-column deferred scenarios.
 *
 * Memory savings example: For 5 columns from same base table:
 * - Old: 5 columns × 8 bytes = 40 bytes per row
 * - New: 1 DeferredTable × 4 bytes = 4 bytes per row
 *
 * @see AnalyzedColumnInfo for materialization decisions.
 * @see DeferredTable for 32-bit row ID storage.
 */
struct IntermediateResult {
    /// Only columns marked MATERIALIZE (typically 1 join key).
    std::vector<mema::column_t> materialized;

    /// Map: original column index -> index in materialized (nullopt if
    /// deferred).
    std::vector<std::optional<size_t>> materialized_map;

    /// Per-base-table deferred row ID storage. One DeferredTable per unique
    /// (from_build, base_table_id) pair. All columns from same base table share
    /// the same row ID lookup.
    std::vector<mema::DeferredTable> deferred_tables;

    /// Map: original column index -> DeferredColumnRef (nullopt if
    /// materialized). The ref contains table_idx (into deferred_tables) and
    /// base_col for resolution.
    std::vector<std::optional<DeferredColumnRef>> deferred_map;

    /// Reference to node info for column provenance resolution.
    const AnalyzedJoinNode *node_info = nullptr;

    /// Total row count.
    size_t num_rows = 0;

    IntermediateResult() = default;
    IntermediateResult(IntermediateResult &&) = default;
    IntermediateResult &operator=(IntermediateResult &&) = default;
    IntermediateResult(const IntermediateResult &) = delete;
    IntermediateResult &operator=(const IntermediateResult &) = delete;

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

    /** @brief Get deferred table for a column, or nullptr if materialized. */
    const mema::DeferredTable *get_deferred_table(size_t orig_idx) const {
        if (!is_deferred(orig_idx))
            return nullptr;
        return &deferred_tables[deferred_map[orig_idx]->table_idx];
    }

    /** @brief Get mutable deferred table for a column, or nullptr. */
    mema::DeferredTable *get_deferred_table_mut(size_t orig_idx) {
        if (!is_deferred(orig_idx))
            return nullptr;
        return &deferred_tables[deferred_map[orig_idx]->table_idx];
    }

    /** @brief Get base column index for deferred column. */
    uint8_t get_deferred_base_col(size_t orig_idx) const {
        if (!is_deferred(orig_idx))
            return 0;
        return deferred_map[orig_idx]->base_col;
    }

    /** @brief Get full DeferredColumnRef for a column, or nullptr. */
    const DeferredColumnRef *get_deferred_ref(size_t orig_idx) const {
        if (!is_deferred(orig_idx))
            return nullptr;
        return &(*deferred_map[orig_idx]);
    }

    /** @brief Number of deferred tables (unique base tables). */
    size_t num_deferred_tables() const { return deferred_tables.size(); }
};

/**
 * @brief Unified abstraction over columnar tables and intermediate results.
 *
 * Stores ColumnarTable* (base scans) or IntermediateResult (child joins).
 * Provides uniform interface for columnar (base table) and intermediate
 * data sources.
 *
 * @see IntermediateResult for intermediate join results.
 * @see ColumnarTable for base table storage.
 */
struct JoinInput {
    /// Either base table pointer or owned IntermediateResult.
    std::variant<const ColumnarTable *, IntermediateResult> data;

    /// Original plan node for output_attrs mapping.
    const PlanNode *node = nullptr;

    /// Analyzed plan node for materialization decisions.
    const AnalyzedNode *analyzed_node = nullptr;

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
        return std::get<IntermediateResult>(data).row_count();
    }

    /** @brief Total row count. */
    size_t row_count() const {
        if (is_columnar()) {
            const auto *table = std::get<const ColumnarTable *>(data);
            return table->num_rows;
        }
        return std::get<IntermediateResult>(data).row_count();
    }

    /** @brief Number of output columns. */
    size_t output_size() const {
        if (node)
            return node->output_attrs.size();
        return 0;
    }

    /**
     * @brief Get deferred table for a column index.
     *
     * For columnar inputs, returns nullptr (caller must encode fresh).
     * For IntermediateResult inputs, returns existing deferred table.
     */
    const mema::DeferredTable *get_deferred_table(size_t col_idx) const {
        if (is_columnar())
            return nullptr;
        return std::get<IntermediateResult>(data).get_deferred_table(col_idx);
    }

    /**
     * @brief Get base column index for a deferred column.
     *
     * For columnar inputs, returns 0 (caller must use column metadata).
     * For IntermediateResult inputs, returns stored base column index.
     */
    uint8_t get_deferred_base_col(size_t col_idx) const {
        if (is_columnar())
            return 0;
        return std::get<IntermediateResult>(data).get_deferred_base_col(
            col_idx);
    }

    /**
     * @brief Get full DeferredColumnRef for a column index.
     *
     * For columnar inputs, returns nullptr (caller must encode fresh).
     * For IntermediateResult inputs, returns pointer to DeferredColumnRef.
     */
    const DeferredColumnRef *get_deferred_ref(size_t col_idx) const {
        if (is_columnar())
            return nullptr;
        return std::get<IntermediateResult>(data).get_deferred_ref(col_idx);
    }
};

} // namespace Contest

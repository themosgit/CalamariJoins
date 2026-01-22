/**
 * @file intermediate.h
 * @brief Intermediate join result types and input abstraction.
 *
 * Provides:
 * - mema::value_t: 4-byte value encoding (INT32 direct, VARCHAR as page/offset)
 * - mema::column_t: 16KB-paged column for materialized values
 * - mema::deferred_column_t: 32KB-paged column for 64-bit provenance encoding
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
 * @brief Compact join intermediate: value_t (4B) + column_t (16KB pages).
 *
 * value_t: INT32 direct or VARCHAR page/offset ref. column_t: arena-allocated
 * pages with write_at().
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
 * @brief 64-bit provenance column for deferred materialization.
 *
 * Stores encoded (table_id, column_idx, row_id) for each row using
 * DeferredProvenance encoding. Uses 32KB pages with 4096 entries each.
 *
 * @see DeferredProvenance for encoding scheme.
 * @see IntermediateResult for usage.
 */
struct deferred_column_t {
    static constexpr size_t PAGE_SIZE = 1 << 15; // 32KB
    static constexpr size_t ENTRIES_PER_PAGE =
        PAGE_SIZE / sizeof(uint64_t);         // 4096
    static constexpr size_t ENTRY_SHIFT = 12; // log2(4096)
    static constexpr size_t ENTRY_MASK = ENTRIES_PER_PAGE - 1;

    struct alignas(PAGE_SIZE) Page {
        uint64_t data[ENTRIES_PER_PAGE];
    };

    std::vector<Page *> pages;
    size_t num_values = 0;

    deferred_column_t() = default;

    deferred_column_t(deferred_column_t &&other) noexcept
        : pages(std::move(other.pages)), num_values(other.num_values) {
        other.pages.clear();
        other.num_values = 0;
    }

    deferred_column_t &operator=(deferred_column_t &&other) noexcept {
        if (this != &other) {
            pages = std::move(other.pages);
            num_values = other.num_values;
            other.pages.clear();
            other.num_values = 0;
        }
        return *this;
    }

    deferred_column_t(const deferred_column_t &) = delete;
    deferred_column_t &operator=(const deferred_column_t &) = delete;

    ~deferred_column_t() = default;

    /** @brief O(1) read: idx>>12 for page, idx&0xFFF for offset. */
    inline uint64_t operator[](size_t idx) const {
        return pages[idx >> ENTRY_SHIFT]->data[idx & ENTRY_MASK];
    }

    /** @brief Thread-safe write at idx (requires pages to be set up first). */
    inline void write_at(size_t idx, uint64_t val) {
        pages[idx >> ENTRY_SHIFT]->data[idx & ENTRY_MASK] = val;
    }

    /** @brief Total value count. */
    size_t row_count() const { return num_values; }

    /** @brief Set row count without allocation (for assembly pattern). */
    inline void set_row_count(size_t count) { num_values = count; }

    /** @brief Pre-allocate pages from arena. */
    inline void pre_allocate_from_arena(Contest::platform::ThreadArena &arena,
                                        size_t count) {
        static_assert(
            sizeof(Page) ==
                Contest::platform::ChunkSize<
                    Contest::platform::ChunkType::DEFERRED_PAGE>::value,
            "Page size mismatch with DEFERRED_PAGE chunk size");
        size_t pages_needed = (count + ENTRIES_PER_PAGE - 1) / ENTRIES_PER_PAGE;
        pages.reserve(pages_needed);
        for (size_t i = 0; i < pages_needed; ++i) {
            void *ptr =
                arena
                    .alloc_chunk<Contest::platform::ChunkType::DEFERRED_PAGE>();
            pages.push_back(reinterpret_cast<Page *>(ptr));
        }
        num_values = count;
    }
};

} // namespace mema

namespace Contest {

/**
 * @brief Lightweight intermediate result with selective materialization.
 *
 * Stores only columns marked MATERIALIZE (typically just the parent's join
 * key). All other columns are resolved at final materialization using
 * per-column 64-bit provenance (table_id, column_idx, row_id).
 *
 * Memory savings: For a join projecting N columns where only 1 is a join key,
 * IntermediateResult uses ~1/N the memory for data columns. Additionally, we
 * only track provenance for deferred columns (not all tables).
 *
 * @see AnalyzedColumnInfo for materialization decisions.
 * @see DeferredProvenance for 64-bit encoding scheme.
 */
struct IntermediateResult {
    /// Only columns marked MATERIALIZE (typically 1 join key).
    std::vector<mema::column_t> materialized;

    /// Map: original column index -> index in materialized (nullopt if
    /// deferred).
    std::vector<std::optional<size_t>> materialized_map;

    /// Per-deferred-column provenance (64-bit encoded table_id+column_idx+row).
    /// One deferred_column_t per DEFER column, stores full provenance per row.
    std::vector<mema::deferred_column_t> deferred_columns;

    /// Map: original column index -> index in deferred_columns (nullopt if
    /// materialized).
    std::vector<std::optional<size_t>> deferred_map;

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
     * @brief Get deferred column provenance for a column index.
     *
     * For columnar inputs, returns nullptr (caller must encode fresh).
     * For IntermediateResult inputs, returns existing provenance column.
     */
    const mema::deferred_column_t *get_deferred_column(size_t col_idx) const {
        if (is_columnar())
            return nullptr;
        return std::get<IntermediateResult>(data).get_deferred(col_idx);
    }
};

} // namespace Contest

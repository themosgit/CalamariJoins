/**
 * @file intermediate.h
 * @brief Intermediate join format: VARCHAR as page/offset refs (no string
 * copy).
 *
 * Base tables must outlive execution. @see plan.h ColumnarTable,
 * construct_intermediate.h
 */
#pragma once

#include <cstdint>
#include <data_access/table.h>
#include <data_model/plan.h>
#include <foundation/common.h>
#include <platform/arena.h>
#include <vector>

/**
 * @namespace mema
 * @brief Compact join intermediate: value_t (4B) + column_t (16KB pages).
 *
 * value_t: INT32 direct or VARCHAR page/offset ref. column_t: arena-allocated
 * pages with write_at(). @see Contest::ExecuteResult, plan.h ColumnarTable.
 */
namespace mema {

/**
 * @brief 4-byte value: INT32 direct, VARCHAR packed (19-bit page + 13-bit
 * offset), NULL = INT32_MIN, long string offset = 0x1FFF. Refs valid only while
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

    static constexpr int32_t LONG_STRING_OFFSET =
        0x1FFF; /**< Sentinel for long strings. */
    static constexpr int32_t NULL_VALUE =
        INT32_MIN; /**< NULL sentinel for both types. */

    /** @brief Check if this value represents NULL. */
    inline bool is_null() const { return value == NULL_VALUE; }
};

/** @brief Page size for intermediate results (16KB, larger than ColumnarTable).
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
    uint8_t source_table =
        0; /**< Base table index for VARCHAR dereferencing. */
    uint8_t source_column = 0; /**< Column index within source table. */

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

    /** @brief O(1) read: idx>>12 for page, idx&0xFFF for offset. No bounds
     * check. */
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
 * @brief Row ID column storing encoded global row IDs.
 *
 * Parallel structure to column_t but stores uint32_t (encoded table_id +
 * row_id). One column per base table participating in joins up to this point.
 * Uses same page size and arena allocation as column_t.
 *
 * @see GlobalRowId for encoding scheme, ExtendedResult for usage.
 */
struct rowid_column_t {
    /** @brief Page for row ID storage: fixed array of uint32_t entries. */
    struct alignas(IR_PAGE_SIZE) Page {
        uint32_t data[CAP_PER_PAGE];
    };

    std::vector<Page *> pages; ///< Pointers to arena-allocated pages.
    size_t num_values = 0;     ///< Total row ID count across all pages.
    uint8_t table_id = 0;      ///< Which base table this column tracks.

    rowid_column_t() = default;

    rowid_column_t(rowid_column_t &&other) noexcept
        : pages(std::move(other.pages)), num_values(other.num_values),
          table_id(other.table_id) {
        other.pages.clear();
        other.num_values = 0;
    }

    rowid_column_t &operator=(rowid_column_t &&other) noexcept {
        if (this != &other) {
            pages = std::move(other.pages);
            num_values = other.num_values;
            table_id = other.table_id;
            other.pages.clear();
            other.num_values = 0;
        }
        return *this;
    }

    rowid_column_t(const rowid_column_t &) = delete;
    rowid_column_t &operator=(const rowid_column_t &) = delete;

    ~rowid_column_t() = default;

    /** @brief O(1) read: idx>>12 for page, idx&0xFFF for offset. */
    inline uint32_t operator[](size_t idx) const {
        return pages[idx >> 12]->data[idx & 0xFFF];
    }

    /** @brief Thread-safe write at idx (requires pages to be set up first). */
    inline void write_at(size_t idx, uint32_t val) {
        pages[idx >> 12]->data[idx & 0xFFF] = val;
    }

    /** @brief Total row ID count. */
    size_t row_count() const { return num_values; }

    /** @brief Set row count without allocation (for assembly pattern). */
    inline void set_row_count(size_t count) { num_values = count; }

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
};

/**
 * @brief 64-bit provenance column for deferred materialization.
 *
 * Stores encoded (table_id, column_idx, row_id) for each row using
 * DeferredProvenance encoding. Uses 32KB pages with 4096 entries each.
 *
 * @see DeferredProvenance for encoding scheme.
 * @see deferred_intermediate.h for DeferredResult usage.
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

/**
 * @brief Convert column_t vector to ColumnarTable. Dereferences VARCHAR refs.
 * @see materialize.h
 */
ColumnarTable to_columnar(const Columnar &table, const Plan &plan);
} /* namespace mema */

/** @namespace Contest @brief Contest API. @see Plan, execute.cpp */
namespace Contest {
/** @brief Result type for non-root joins (intermediate format). */
using ExecuteResult = std::vector<mema::column_t>;

/**
 * @brief Extended intermediate result with row ID tracking.
 *
 * Wraps ExecuteResult with parallel row ID columns that track
 * which original scan rows contributed to each intermediate row.
 * One rowid_column_t per base table participating in the join tree.
 *
 * @see GlobalRowId for encoding, construct_intermediate.h for population.
 */
struct ExtendedResult {
    ExecuteResult columns;                     ///< Data columns (value_t).
    std::vector<mema::rowid_column_t> row_ids; ///< One per participating table.
    std::vector<uint8_t> table_ids; ///< Which tables are tracked (sorted).

    ExtendedResult() = default;

    ExtendedResult(ExtendedResult &&) = default;
    ExtendedResult &operator=(ExtendedResult &&) = default;

    ExtendedResult(const ExtendedResult &) = delete;
    ExtendedResult &operator=(const ExtendedResult &) = delete;

    /** @brief Total row count (from first data column). */
    size_t row_count() const {
        return columns.empty() ? 0 : columns[0].row_count();
    }

    /** @brief Find row ID column index for a specific table, or -1 if not
     * found. */
    int find_rowid_index(uint8_t tid) const {
        for (size_t i = 0; i < table_ids.size(); ++i) {
            if (table_ids[i] == tid)
                return static_cast<int>(i);
        }
        return -1;
    }

    /** @brief Get row ID column for a specific table, or nullptr if not found.
     */
    const mema::rowid_column_t *get_rowid_column(uint8_t tid) const {
        int idx = find_rowid_index(tid);
        return (idx >= 0) ? &row_ids[idx] : nullptr;
    }

    /** @brief Get mutable row ID column for a specific table, or nullptr. */
    mema::rowid_column_t *get_rowid_column_mut(uint8_t tid) {
        int idx = find_rowid_index(tid);
        return (idx >= 0) ? &row_ids[idx] : nullptr;
    }
};

} /* namespace Contest */

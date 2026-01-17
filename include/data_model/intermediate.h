/**
 * @file intermediate.h
 * @brief Intermediate result format for multi-way joins.
 *
 * Defines column_t, a compact columnar format for passing results between
 * join stages. Uses larger pages (16KB) than ColumnarTable (8KB) for efficient
 * bulk allocation. Values stored as value_t: INT32 directly, VARCHAR as
 * encoded references to original ColumnarTable pages.
 *
 * **Key difference from ColumnarTable:** Intermediate format avoids copying
 * string data between join stages by encoding VARCHAR as page/offset references
 * into the original base tables. This requires base tables to remain valid
 * until final materialization.
 *
 * @see plan.h for ColumnarTable (final output format).
 * @see construct_intermediate.h for building intermediate results.
 */
#pragma once

#include <cstdint>
#include <data_access/table.h>
#include <data_model/plan.h>
#include <foundation/common.h>
#include <memory>
#include <vector>

namespace mema {

/**
 * @brief Compact 4-byte value for intermediate join results.
 *
 * Encoding depends on column type:
 * - **INT32:** Value stored directly in the 4-byte field.
 * - **VARCHAR:** Packed reference to original ColumnarTable page/offset.
 *   - Bits 0-18 (19 bits): page_idx - supports up to 2^19 pages per table
 *   - Bits 19-31 (13 bits): offset_idx - supports up to 2^13 offsets per page
 * - **NULL:** Sentinel value INT32_MIN for both types.
 * - **Long string:** offset_idx = 0x1FFF sentinel, page_idx indicates start.
 *
 * The 4-byte size maximizes cache efficiency during joins. The bit packing
 * limits are sufficient for the IMDB/JOB benchmark dataset used in the contest.
 *
 * @note VARCHAR references remain valid only while the source ColumnarTable
 *       (base table) exists. Final materialization dereferences these.
 */
struct alignas(4) value_t {
    int32_t value;

    /**
     * @brief Encode a VARCHAR reference as page/offset pair.
     * @param page_idx   Page index in source Column (max 2^19 - 1).
     * @param offset_idx Offset index within page (max 2^13 - 1).
     */
    static inline value_t encode_string(int32_t page_idx, int32_t offset_idx) {
        return {(offset_idx << 19) | (page_idx & 0x7FFFF)};
    }

    /**
     * @brief Decode a VARCHAR reference into page/offset pair.
     * @param encoded   The packed value_t.value field.
     * @param page_idx  [out] Page index in source Column.
     * @param offset_idx [out] Offset index within page.
     */
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
 * @brief Columnar storage for intermediate join results.
 *
 * Stores value_t entries in fixed-capacity pages. Unlike ColumnarTable,
 * intermediate pages have no header - just packed value_t arrays.
 *
 * **Memory management:**
 * - By default, owns_pages=true and pages are new/deleted individually.
 * - With BatchAllocator, owns_pages=false and external_memory holds shared_ptr
 *   to the mmap'd region. This enables single-allocation for all columns.
 *
 * **Provenance tracking:**
 * - source_table/source_column track which base table column this data
 *   originated from (for VARCHAR dereferencing during materialization).
 *
 * @note Move-only type. Thread-safe random writes via write_at() after
 *       pre_allocate() or resize().
 */
struct column_t {
  private:
    /** @brief Intermediate page: fixed array of value_t entries. */
    struct alignas(IR_PAGE_SIZE) Page {
        value_t data[CAP_PER_PAGE];
    };

    size_t num_values = 0;  /**< Total value count across all pages. */
    bool owns_pages = true; /**< If true, destructor deletes pages. */
    std::shared_ptr<void> external_memory; /**< Keeps BatchAllocator alive. */

  public:
    std::vector<Page *> pages; /**< Pointers to data pages. */
    uint8_t source_table =
        0; /**< Base table index for VARCHAR dereferencing. */
    uint8_t source_column = 0; /**< Column index within source table. */

  public:
    column_t() = default;

    column_t(column_t &&other) noexcept
        : num_values(other.num_values), owns_pages(other.owns_pages),
          external_memory(std::move(other.external_memory)),
          pages(std::move(other.pages)), source_table(other.source_table),
          source_column(other.source_column) {
        other.owns_pages = false;
        other.pages.clear();
        other.num_values = 0;
    }

    column_t &operator=(column_t &&other) noexcept {
        if (this != &other) {
            if (owns_pages) {
                for (auto *p : pages)
                    delete p;
            }

            num_values = other.num_values;
            owns_pages = other.owns_pages;
            external_memory = std::move(other.external_memory);
            pages = std::move(other.pages);
            source_table = other.source_table;
            source_column = other.source_column;

            other.owns_pages = false;
            other.pages.clear();
            other.num_values = 0;
        }
        return *this;
    }

    column_t(const column_t &) = delete;
    column_t &operator=(const column_t &) = delete;

    ~column_t() {
        if (owns_pages) {
            for (auto *page : pages)
                delete page;
        }
    }

    /**
     * @brief Resize column to hold exactly count elements.
     *
     * Allocates new pages as needed. Must be called before parallel write_at()
     * to ensure all pages exist.
     */
    inline void resize(size_t count) {
        size_t current_capacity = pages.size() * CAP_PER_PAGE;
        if (count > current_capacity) {
            size_t needed = count - current_capacity;
            size_t pages_needed = (needed + CAP_PER_PAGE - 1) / CAP_PER_PAGE;
            pages.reserve(pages.size() + pages_needed);
            for (size_t i = 0; i < pages_needed; ++i) {
                pages.push_back(new Page());
            }
        }
        num_values = count;
    }

    /** @brief Append a value, allocating new page if needed (not thread-safe).
     */
    inline void append(const value_t &val) {
        if ((num_values & (CAP_PER_PAGE - 1)) == 0) {
            pages.push_back(new Page());
        }
        pages.back()->data[num_values & (CAP_PER_PAGE - 1)] = val;
        num_values++;
    }

    /**
     * @brief Pre-allocate all pages for parallel writing.
     *
     * Must be called from main thread before spawning workers. Each page
     * is independently allocated, enabling parallel write_at() calls.
     */
    inline void pre_allocate(size_t count) {
        size_t pages_needed = (count + CAP_PER_PAGE - 1) / CAP_PER_PAGE;
        pages.reserve(pages_needed);
        for (size_t i = 0; i < pages_needed; ++i) {
            pages.push_back(new Page());
        }
        num_values = count;
    }

    /** @brief Read-only access by global index. Uses bit masking for fast page
     * lookup. */
    inline const value_t &operator[](size_t idx) const {
        return pages[idx >> 12]->data[idx & 0xFFF];
    }

    /** @brief Total number of values in this column. */
    size_t row_count() const { return num_values; }

    /**
     * @brief Pre-allocate pages from a contiguous memory block
     * (BatchAllocator).
     *
     * Uses arena allocation for cache-friendly bulk memory. The shared_ptr
     * keeps the memory region alive until all columns using it are destroyed.
     *
     * @param block         Start of mmap'd memory region.
     * @param offset        [in/out] Current offset into block; advanced by
     * allocation.
     * @param count         Number of values to accommodate.
     * @param memory_keeper Shared ownership of the memory region.
     */
    inline void pre_allocate_from_block(void *block, size_t &offset,
                                        size_t count,
                                        std::shared_ptr<void> memory_keeper) {
        size_t pages_needed = (count + CAP_PER_PAGE - 1) / CAP_PER_PAGE;
        pages.reserve(pages_needed);
        char *base = static_cast<char *>(block);
        for (size_t i = 0; i < pages_needed; ++i) {
            pages.push_back(reinterpret_cast<Page *>(base + offset));
            offset += sizeof(Page);
        }
        num_values = count;
        owns_pages = false;
        external_memory = memory_keeper;
    }

    /**
     * @brief Thread-safe random write to pre-allocated pages.
     *
     * Safe for concurrent writes to different indices. Pages must be
     * pre-allocated via pre_allocate() or pre_allocate_from_block().
     */
    inline void write_at(size_t idx, const value_t &val) {
        pages[idx >> 12]->data[idx & 0xFFF] = val;
    }
};

/** @brief Alias for a collection of intermediate columns. */
using Columnar = std::vector<column_t>;

/** @brief Convert intermediate format to final ColumnarTable output. */
ColumnarTable to_columnar(const Columnar &table, const Plan &plan);
} /* namespace mema */

namespace Contest {
/** @brief Result type for non-root joins (intermediate format). */
using ExecuteResult = std::vector<mema::column_t>;
} /* namespace Contest */

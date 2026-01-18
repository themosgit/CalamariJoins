/**
 * @file intermediate.h
 * @brief Intermediate result format for multi-way joins.
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

/**
 * @namespace mema
 * @brief Memory-efficient intermediate result format for multi-way joins.
 *
 * Provides column_t, a compact columnar storage using value_t encoding
 * (4 bytes per value) for passing results between join stages. Designed
 * for cache efficiency and minimal memory allocation during parallel joins.
 *
 * Key components:
 * - **value_t**: Compact 4-byte encoding for INT32 (direct) and VARCHAR
 *   (page/offset reference into source ColumnarTable).
 * - **column_t**: Page-based storage (16KB pages) with support for both
 *   sequential append and parallel random-access writes.
 * - **Columnar**: Type alias for std::vector<column_t>.
 *
 * @see Contest::ExecuteResult for the result type of non-root joins.
 * @see plan.h ColumnarTable for the final output format.
 */
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
 * Stores value_t entries in fixed-capacity pages (16KB each, larger than
 * ColumnarTable's 8KB for efficient bulk allocation). Pages have no header -
 * just packed value_t arrays for maximum cache density.
 *
 * **Memory management:**
 * - **owns_pages=true** (default): Individual page allocation via new/delete.
 *   Use for single-threaded sequential construction with append().
 * - **owns_pages=false** (BatchAllocator): All pages allocated from single
 *   mmap'd region. external_memory holds shared_ptr keeping region alive
 *   until all columns are destroyed. Use for parallel construction.
 *
 * **Thread-safety:**
 * - write_at() is thread-safe after pre_allocate() or pre_allocate_from_block()
 *   (pages never reallocate, each index maps to distinct memory location).
 * - append() is NOT thread-safe (allocates pages on-demand, modifies
 * num_values).
 *
 * **Provenance tracking:**
 * - source_table/source_column identify which base table column this data
 *   originated from, enabling VARCHAR dereferencing during materialization.
 *
 * **Key invariants:**
 * - VARCHAR references in value_t remain valid only while source ColumnarTable
 * exists.
 * - Page count limited to 2^19, offset count to 2^13 (sufficient for IMDB/JOB
 * benchmark).
 *
 * **Usage examples:**
 * @code
 * // Single-threaded sequential build
 * column_t col;
 * col.append(value_t{42});
 * col.append(value_t::encode_string(page_idx, offset_idx));
 *
 * // Parallel construction
 * column_t col;
 * col.pre_allocate(1000000);  // Allocate all pages upfront
 * worker_pool.execute([&](size_t thread_id, size_t start, size_t end) {
 *     for (size_t i = start; i < end; ++i) {
 *         col.write_at(i, compute_value(i));  // Thread-safe
 *     }
 * });
 * @endcode
 *
 * @note Move-only type (copy constructor/assignment deleted).
 * @see construct_intermediate.h for BatchAllocator integration.
 * @see plan.h for ColumnarTable (final output format).
 * @see materialize.h for VARCHAR dereferencing from source tables.
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
     * Allocates new pages as needed to reach target capacity. Call before
     * parallel write_at() operations to ensure all pages exist (prevents
     * race conditions from on-demand allocation).
     *
     * @param count Target number of value_t entries.
     *
     * @note If count <= current capacity, only updates num_values (no
     * deallocation).
     * @note Pages allocated individually via new (owns_pages=true).
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

    /**
     * @brief Append a value, allocating new page if needed.
     *
     * Sequential append with automatic page allocation. When current page fills
     * (num_values reaches CAP_PER_PAGE boundary), allocates next page via new.
     *
     * @param val The value_t to append (INT32 or VARCHAR reference).
     *
     * @warning NOT thread-safe. Modifies num_values and potentially allocates
     *          pages. Use only for single-threaded construction. For parallel
     *          construction, use pre_allocate() + write_at().
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
     * Allocates all required pages upfront (via new), enabling lock-free
     * parallel writes. Must be called from main thread before spawning
     * worker threads. Pages never reallocate, so each write_at() index
     * maps to a distinct, stable memory location.
     *
     * @param count Total number of value_t entries to accommodate.
     *
     * @note Pages allocated individually (owns_pages=true). For bulk allocation
     *       from single mmap, use pre_allocate_from_block() instead.
     */
    inline void pre_allocate(size_t count) {
        size_t pages_needed = (count + CAP_PER_PAGE - 1) / CAP_PER_PAGE;
        pages.reserve(pages_needed);
        for (size_t i = 0; i < pages_needed; ++i) {
            pages.push_back(new Page());
        }
        num_values = count;
    }

    /**
     * @brief Read-only access by global index.
     *
     * Fast O(1) lookup using bit masking instead of division/modulo:
     * - Page index: idx >> 12 (divide by CAP_PER_PAGE = 4096 = 2^12)
     * - Offset within page: idx & 0xFFF (modulo 4096)
     *
     * @param idx Global value index across all pages.
     * @return Reference to the value_t at that position.
     *
     * @note Assumes idx < num_values (no bounds checking for performance).
     */
    inline const value_t &operator[](size_t idx) const {
        return pages[idx >> 12]->data[idx & 0xFFF];
    }

    /**
     * @brief Total number of values in this column.
     *
     * @return Count of value_t entries across all pages.
     */
    size_t row_count() const { return num_values; }

    /**
     * @brief Pre-allocate pages from a contiguous memory block
     * (BatchAllocator).
     *
     * Arena allocation from single mmap'd region for optimal cache locality
     * and reduced allocation overhead. Pages assigned via pointer arithmetic
     * into the block. The shared_ptr keeps the entire memory region alive
     * until all columns using it are destroyed (reference counting).
     *
     * @param block         Start of pre-allocated mmap'd memory region.
     * @param offset        [in/out] Current byte offset into block; advanced
     *                      by this allocation (enables sequential assignment).
     * @param count         Number of value_t entries to accommodate.
     * @param memory_keeper Shared ownership of the memory region (prevents
     *                      premature munmap).
     *
     * @note Sets owns_pages=false (destructor skips delete, munmap happens
     *       when last shared_ptr reference drops).
     * @note Pages 16KB-aligned for cache-friendly access.
     * @see construct_intermediate.h for BatchAllocator implementation.
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
     * Lock-free writes enabled by pre-allocation guarantee: pages never
     * reallocate, and each index maps to a distinct memory location.
     * Multiple threads can safely write to different indices concurrently.
     *
     * @param idx Global value index (0 to num_values-1).
     * @param val The value_t to write (INT32 or VARCHAR reference).
     *
     * @warning Pages MUST be pre-allocated via pre_allocate() or
     *          pre_allocate_from_block() before calling from worker threads.
     * @note Uses bit masking for fast page/offset calculation (same as
     * operator[]).
     */
    inline void write_at(size_t idx, const value_t &val) {
        pages[idx >> 12]->data[idx & 0xFFF] = val;
    }
};

/** @brief Alias for a collection of intermediate columns. */
using Columnar = std::vector<column_t>;

/**
 * @brief Convert intermediate format to final ColumnarTable output.
 *
 * Transforms vector of column_t (intermediate format with value_t encoding)
 * into ColumnarTable (contest API output format). Dereferences VARCHAR
 * value_t references to copy actual string bytes into output pages.
 *
 * @param table Intermediate result columns (vector of column_t).
 * @param plan  Execution plan containing schema information and base tables.
 * @return ColumnarTable with proper page headers, null bitmaps, and
 * materialized strings.
 *
 * @note This is the root-node materialization path. Non-root joins use
 *       construct_intermediate() to preserve value_t encoding.
 * @see materialize.h for the implementation.
 * @see construct_intermediate.h for non-root intermediate construction.
 */
ColumnarTable to_columnar(const Columnar &table, const Plan &plan);
} /* namespace mema */

/**
 * @namespace Contest
 * @brief Contest execution API and related types.
 *
 * Provides the main execution interface (execute()), timing instrumentation
 * (TimingStats), and result types for query execution.
 *
 * @see Plan for the query plan structure.
 * @see execute.cpp for the main execution implementation.
 */
namespace Contest {
/** @brief Result type for non-root joins (intermediate format). */
using ExecuteResult = std::vector<mema::column_t>;
} /* namespace Contest */

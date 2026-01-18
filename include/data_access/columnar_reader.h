/**
 * @file columnar_reader.h
 * @brief High-performance random access reader for columnar storage.
 *
 * Provides O(log P) page lookup via PageIndex with O(1) amortized access
 * through per-thread Cursor caching. Handles both dense (no NULLs) and
 * sparse (with bitmap) pages, plus multi-page long strings.
 *
 * **Access pattern optimizations:**
 * - Sequential access: cursor caching avoids binary search
 * - Random access: PageIndex provides efficient page lookup
 * - Sparse pages: prefix sums enable O(1) popcount-based value index
 *
 * **Versioning:** Global version counters invalidate cursors when
 * prepare_build/prepare_probe is called for a new join phase.
 *
 * @see PageIndex for cumulative row count index.
 * @see Cursor for thread-local access state.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <data_model/intermediate.h>
#include <data_model/plan.h>
#include <foundation/attribute.h>
#include <vector>

/** @brief Branch prediction hint for likely paths. */
#define SPC_LIKELY(x) __builtin_expect(!!(x), 1)
/** @brief Branch prediction hint for unlikely paths. */
#define SPC_UNLIKELY(x) __builtin_expect(!!(x), 0)

/**
 * @namespace Contest::io
 * @brief Columnar I/O access layer for Contest execution engine.
 *
 * Contains columnar storage access components:
 * - **Columnar Access**: ColumnarReader with cursor caching
 * - **Page Indexing**: PageIndex for O(log P) row lookup
 *
 * @see Plan for query plan structure.
 * @see mema namespace for intermediate result format.
 */
namespace Contest::io {

inline std::atomic<uint64_t> global_build_version{0};
inline std::atomic<uint64_t> global_probe_version{0};

/** @brief Pre-computed page index for O(log P) row lookup in ColumnarTable. */
struct alignas(8) PageIndex {
    /**
     * Cumulative row counts per page, enabling O(log P) binary search.
     * Element i contains total rows from pages [0..i], so upper_bound(row_id)
     * yields the containing page. This avoids linear scanning for row lookup.
     */
    std::vector<uint32_t> cumulative_rows;

    /**
     * Per-page prefix sums of bitmap popcount for sparse pages.
     * Each inner vector has one entry per 64-bit bitmap chunk, storing
     * the cumulative count of set bits before that chunk. This enables O(1)
     * value index calculation: prefix_sum[chunk] + popcount(word & mask).
     * Empty for dense pages where no indirection is needed.
     */
    std::vector<std::vector<uint32_t>> page_prefix_sums;

    /**
     * Optimization flag: true if all pages are dense (no NULLs).
     * When true, bitmap checks can be skipped entirely, enabling the
     * fastest path. Set to false if any page is sparse or special.
     */
    bool all_dense = true;

    /** @brief Builds page index for a column, computing cumulative row counts.
     */
    inline void build(const Column &column) {
        uint32_t total = 0;
        cumulative_rows.reserve(column.pages.size());
        page_prefix_sums.reserve(column.pages.size());
        all_dense = true;

        size_t page_idx = 0;
        for (auto *page_obj : column.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page);

            if (SPC_UNLIKELY(num_rows == 0xfffe)) {
                all_dense = false;
            } else if (SPC_UNLIKELY(num_rows == 0xffff)) {
                all_dense = false;
                total += 1;
            } else {
                total += num_rows;
            }
            cumulative_rows.push_back(total);

            auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
            std::vector<uint32_t> prefix_sums;

            /* sparse page: build prefix sums for bitmap popcount */
            if (num_rows != 0xfffe && num_rows != 0xffff &&
                num_rows != num_values) {
                all_dense = false;

                size_t bitmap_size = (num_rows + 7) / 8;
                auto *bitmap_bytes = reinterpret_cast<const uint8_t *>(
                    page + PAGE_SIZE - bitmap_size);

                size_t num_chunks = (num_rows + 63) / 64;
                prefix_sums.reserve(num_chunks);
                uint32_t sum = 0;
                for (size_t i = 0; i < num_chunks; ++i) {
                    prefix_sums.push_back(sum);
                    uint64_t word = 0;
                    size_t offset = i * 8;
                    size_t remaining =
                        bitmap_size > offset ? bitmap_size - offset : 0;
                    size_t bytes_to_read = std::min(remaining, size_t(8));
                    if (bytes_to_read > 0) {
                        std::memcpy(&word, bitmap_bytes + offset,
                                    bytes_to_read);
                    }
                    sum += __builtin_popcountll(word);
                }
            }
            page_prefix_sums.push_back(std::move(prefix_sums));
            page_idx++;
        }
    }

    /** @brief Finds the page containing row_id via binary search. */
    inline size_t find_page(uint32_t row_id) const {
        auto it = std::upper_bound(cumulative_rows.begin(),
                                   cumulative_rows.end(), row_id);
        return static_cast<size_t>(it - cumulative_rows.begin());
    }

    /** @brief Returns the starting row number for a given page. */
    inline uint32_t page_start_row(size_t page_num) const {
        return page_num == 0 ? 0 : cumulative_rows[page_num - 1];
    }
};

/**
 * @brief High-performance random access reader for columnar data.
 *
 * Uses page indices for O(log P) lookup and per-thread cursor caching
 * for O(1) sequential access. Maintains separate indices for build/probe.
 */
class ColumnarReader {
  public:
    ColumnarReader() = default;

    /**
     * @brief Per-thread cursor for spatial locality (cache-aligned).
     *
     * Caches page metadata to avoid repeated binary searches during
     * sequential access patterns. Thread-local ownership eliminates
     * synchronization overhead.
     */
    struct alignas(8) Cursor {
        /**
         * Cache invalidation mechanism tied to global version counters.
         * Incremented by prepare_build/prepare_probe to invalidate all
         * cursors when switching join phases, preventing stale page pointers.
         */
        uint64_t version = 0;

        /**
         * Column index of the cached page. Set to ~0u when uninitialized.
         * Used to detect column switches that require page reload.
         */
        size_t cached_col = ~0u;

        /**
         * Inclusive start of the row range [cached_start, cached_end) covered
         * by the cached page. Enables O(1) range checks for cursor validity.
         */
        uint32_t cached_start = 0;

        /**
         * Exclusive end of the row range [cached_start, cached_end).
         * Fast-path check: row_id in range implies cached page is valid.
         */
        uint32_t cached_end = 0;

        /**
         * Page number currently cached. Set to ~0u when uninitialized.
         * Used for sequential access optimization: if row_id == cached_end,
         * directly load next page without binary search.
         */
        size_t cached_page = ~0u;

        /**
         * Pointer to the page's value array (int32_t[] for INT32 or string
         * offsets). Cached to avoid repeated pointer arithmetic from page base
         * address.
         */
        const int32_t *data_ptr = nullptr;

        /**
         * Pointer to the page's bitmap (for sparse pages only).
         * Each bit indicates whether the corresponding row is non-NULL.
         * Null for dense pages to avoid unnecessary dereferences.
         */
        const uint8_t *bitmap_ptr = nullptr;

        /**
         * Pointer to the prefix sum array (for sparse pages only).
         * Enables O(1) value index calculation via popcount.
         * Null for dense pages.
         */
        const uint32_t *prefix_sum_ptr = nullptr;

        /**
         * Page index as int32_t for string encoding (page_idx | local_offset).
         * Stored to avoid repeated casts during string value construction.
         */
        int32_t page_idx_val = 0;

        /**
         * True if the cached page is dense (num_rows == num_values).
         * Dense pages skip bitmap checks, using direct indexing:
         * data_ptr[local_row].
         */
        bool is_dense = false;

        /**
         * True if the cached page is special (num_rows == 0xffff, multi-page
         * string). Special pages return a constant encoded value without
         * indexing.
         */
        bool is_special = false;

        /**
         * True if all pages in this column are dense (copied from
         * PageIndex.all_dense). Enables additional fast-path optimizations when
         * no NULL handling is needed.
         */
        bool col_all_dense = false;

        /**
         * Explicit padding for alignment. Ensures struct size is a multiple
         * of 8 bytes for cache line efficiency. Not used for logic.
         */
        uint8_t _padding = 0;
    };

    /**
     * @brief Builds page indices for build-side columns and invalidates
     * cursors.
     *
     * Constructs cumulative row counts and prefix sums for all columns.
     * Increments global_build_version to invalidate all existing build cursors,
     * preventing them from using stale page pointers after a new join phase.
     * Must be called before read_value() is used for build-side access.
     *
     * @param columns Build-side columns to index. Null columns create empty
     * indices.
     */
    inline void prepare_build(const std::vector<const Column *> &columns) {
        build_page_indices.clear();
        build_page_indices.reserve(columns.size());
        for (const auto *column : columns) {
            if (column) {
                PageIndex page_idx;
                page_idx.build(*column);
                build_page_indices.push_back(std::move(page_idx));
            } else {
                build_page_indices.emplace_back();
            }
        }
        global_build_version.fetch_add(1, std::memory_order_relaxed);
    }

    /**
     * @brief Builds page indices for probe-side columns and invalidates
     * cursors.
     *
     * Constructs cumulative row counts and prefix sums for all columns.
     * Increments global_probe_version to invalidate all existing probe cursors,
     * ensuring cursors cannot access freed or reallocated page memory.
     * Must be called before read_value() is used for probe-side access.
     *
     * @param columns Probe-side columns to index. Null columns create empty
     * indices.
     */
    inline void prepare_probe(const std::vector<const Column *> &columns) {
        probe_page_indices.clear();
        probe_page_indices.reserve(columns.size());
        for (const auto *column : columns) {
            if (column) {
                PageIndex page_idx;
                page_idx.build(*column);
                probe_page_indices.push_back(std::move(page_idx));
            } else {
                probe_page_indices.emplace_back();
            }
        }
        global_probe_version.fetch_add(1, std::memory_order_relaxed);
    }

    /** @brief Fast path: check cursor cache, dispatch to appropriate handler.
     */
    template <bool IsBuild>
    inline mema::value_t
    read_value_internal(const Column &column, size_t col_idx, uint32_t row_id,
                        DataType data_type, Cursor &cursor) const {

        uint64_t current_version;
        if constexpr (IsBuild) {
            current_version =
                global_build_version.load(std::memory_order_relaxed);
        } else {
            current_version =
                global_probe_version.load(std::memory_order_relaxed);
        }

        if (SPC_LIKELY(cursor.version == current_version &&
                       col_idx == cursor.cached_col &&
                       row_id >= cursor.cached_start &&
                       row_id < cursor.cached_end)) {
            uint32_t local_row = row_id - cursor.cached_start;
            if (SPC_LIKELY(cursor.is_dense)) {
                if (data_type == DataType::INT32) {
                    return mema::value_t{cursor.data_ptr[local_row]};
                } else {
                    return mema::value_t::encode_string(
                        cursor.page_idx_val, static_cast<int32_t>(local_row));
                }
            }
            if (SPC_UNLIKELY(cursor.is_special)) {
                return mema::value_t::encode_string(
                    cursor.page_idx_val, mema::value_t::LONG_STRING_OFFSET);
            }

            return read_sparse(local_row, data_type, cursor);
        }

        /* sequential access optimization: skip binary search for next page */
        if (SPC_LIKELY(cursor.version == current_version &&
                       col_idx == cursor.cached_col &&
                       row_id == cursor.cached_end)) {
            const PageIndex &page_index = IsBuild ? build_page_indices[col_idx]
                                                  : probe_page_indices[col_idx];
            size_t next_page = cursor.cached_page + 1;
            if (SPC_LIKELY(next_page < page_index.cumulative_rows.size())) {
                return read_value_load_page<IsBuild>(
                    column, col_idx, row_id, data_type, cursor, current_version,
                    page_index, next_page);
            }
        }

        return read_value_slow<IsBuild>(column, col_idx, row_id, data_type,
                                        cursor, current_version);
    }

    /**
     * @brief Loads page metadata into cursor cache to enable fast-path access.
     *
     * Caches page boundaries, pointers, and flags so subsequent reads within
     * the same page bypass binary search. For sparse pages, caches bitmap and
     * prefix sum pointers to enable O(1) value index calculation via popcount.
     *
     * **Why cache this:** Page metadata access involves pointer chasing and
     * branches. Caching amortizes this cost across all reads within a page,
     * critical for sequential access patterns (e.g., full table scans).
     */
    template <bool IsBuild>
    inline void
    load_page_into_cursor(const Column &column, const PageIndex &page_index,
                          size_t page_num, size_t col_idx,
                          uint64_t current_version, Cursor &cursor) const {
        cursor.version = current_version;
        cursor.cached_col = col_idx;
        cursor.cached_page = page_num;
        cursor.cached_start = page_index.page_start_row(page_num);
        cursor.cached_end = page_index.cumulative_rows[page_num];
        cursor.page_idx_val = static_cast<int32_t>(page_num);
        cursor.col_all_dense = page_index.all_dense;

        auto *page_data = column.pages[page_num]->data;
        auto num_rows = *reinterpret_cast<const uint16_t *>(page_data);
        auto num_values = *reinterpret_cast<const uint16_t *>(page_data + 2);

        cursor.is_special = (num_rows == 0xffff);
        cursor.is_dense = (num_rows == num_values);
        cursor.data_ptr = reinterpret_cast<const int32_t *>(page_data + 4);

        if (!cursor.is_dense && !cursor.is_special) {
            size_t bitmap_size = (num_rows + 7) / 8;
            cursor.bitmap_ptr = reinterpret_cast<const uint8_t *>(
                page_data + PAGE_SIZE - bitmap_size);
            cursor.prefix_sum_ptr =
                page_index.page_prefix_sums[page_num].data();
        }
    }

    /**
     * @brief O(1) sequential access optimization: load page directly without
     * search.
     *
     * Called when row_id == cursor.cached_end, indicating sequential access to
     * the next page. Skips binary search since page_num is known (cached_page +
     * 1). Critical for sequential scans where binary search overhead would
     * dominate.
     *
     * **Performance:** O(1) page load vs. O(log P) binary search in
     * read_value_slow.
     */
    template <bool IsBuild>
    inline mema::value_t
    read_value_load_page(const Column &column, size_t col_idx, uint32_t row_id,
                         DataType data_type, Cursor &cursor,
                         uint64_t current_version, const PageIndex &page_index,
                         size_t page_num) const {
        load_page_into_cursor<IsBuild>(column, page_index, page_num, col_idx,
                                       current_version, cursor);
        return read_value_internal<IsBuild>(column, col_idx, row_id, data_type,
                                            cursor);
    }

    /**
     * @brief Slow path: O(log P) binary search to find page, then load and
     * read.
     *
     * Called when cursor is invalid (wrong column, version mismatch, or row_id
     * out of cached range). Performs binary search on cumulative_rows to locate
     * the containing page, then loads it into the cursor for subsequent fast
     * access.
     *
     * **Why slow:** Binary search adds O(log P) overhead. Fast path (cursor
     * hit) avoids this by checking cached range first. Random access patterns
     * pay this cost on every read; sequential patterns amortize it across page
     * boundaries.
     */
    template <bool IsBuild>
    inline mema::value_t read_value_slow(const Column &column, size_t col_idx,
                                         uint32_t row_id, DataType data_type,
                                         Cursor &cursor,
                                         uint64_t current_version) const {

        const PageIndex &page_index =
            IsBuild ? build_page_indices[col_idx] : probe_page_indices[col_idx];
        size_t page_num = page_index.find_page(row_id);
        load_page_into_cursor<IsBuild>(column, page_index, page_num, col_idx,
                                       current_version, cursor);
        return read_value_internal<IsBuild>(column, col_idx, row_id, data_type,
                                            cursor);
    }

    /**
     * @brief Reads a single value with cursor caching for efficient access.
     *
     * **Fast path (cursor hit):** O(1) amortized for sequential access within
     * the same page. Checks cached range, then directly indexes or uses
     * popcount for sparse pages.
     *
     * **Slow path (cursor miss):** O(log P) binary search to find page, then
     * loads page metadata into cursor. Subsequent reads within the same page
     * hit the fast path.
     *
     * **Sequential optimization:** When row_id == cursor.cached_end, skips
     * binary search and loads next page directly (O(1)).
     *
     * **Performance characteristics:**
     * - Sequential access: O(1) amortized via cursor caching
     * - Random access: O(log P) per read due to binary search
     * - Sparse pages: O(1) value lookup via prefix sums + popcount
     *
     * @param column Column to read from (must match prepare_build/probe).
     * @param col_idx Column index in page indices array.
     * @param row_id Global row ID to read.
     * @param data_type INT32 or STRING, determines value encoding.
     * @param cursor Thread-local cursor for caching page state.
     * @param from_build True for build side, false for probe side (selects
     * version counter).
     * @return Encoded value (int32_t or string page_idx|offset), or NULL_VALUE.
     */
    inline mema::value_t read_value(const Column &column, size_t col_idx,
                                    uint32_t row_id, DataType data_type,
                                    Cursor &cursor, bool from_build) const {
        if (from_build) {
            return read_value_internal<true>(column, col_idx, row_id, data_type,
                                             cursor);
        } else {
            return read_value_internal<false>(column, col_idx, row_id,
                                              data_type, cursor);
        }
    }

    inline const PageIndex &get_build_page_index(size_t col_idx) const {
        return build_page_indices[col_idx];
    }

    inline const PageIndex &get_probe_page_index(size_t col_idx) const {
        return probe_page_indices[col_idx];
    }

  private:
    /** @brief Reads from sparse pages using bitmap and popcount. */
    inline mema::value_t read_sparse(uint32_t local_row, DataType data_type,
                                     const Cursor &cursor) const {
        bool is_valid =
            cursor.bitmap_ptr[local_row >> 3] & (1u << (local_row & 7));

        if (!is_valid)
            return mema::value_t{mema::value_t::NULL_VALUE};

        size_t chunk_idx = local_row >> 6;
        size_t bit_offset = local_row & 0x3F;
        uint16_t data_idx = cursor.prefix_sum_ptr[chunk_idx];

        uint64_t word;
        std::memcpy(&word, cursor.bitmap_ptr + (chunk_idx * 8), 8);
        data_idx += __builtin_popcountll(word & ((1ULL << bit_offset) - 1));

        if (data_type == DataType::INT32) {
            return mema::value_t{cursor.data_ptr[data_idx]};
        } else {
            return mema::value_t::encode_string(cursor.page_idx_val,
                                                static_cast<int32_t>(data_idx));
        }
    }
    std::vector<PageIndex> build_page_indices;
    std::vector<PageIndex> probe_page_indices;
};
} // namespace Contest::io

// Bring types into Contest:: namespace for backward compatibility
namespace Contest {
using Contest::io::ColumnarReader;
using Contest::io::global_build_version;
using Contest::io::global_probe_version;
using Contest::io::PageIndex;
} // namespace Contest

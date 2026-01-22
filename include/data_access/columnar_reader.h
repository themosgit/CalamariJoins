/**
 * @file columnar_reader.h
 * @brief Columnar random access: O(log P) PageIndex + O(1) cursor caching.
 *
 * Dense/sparse pages, long strings. Version counters invalidate cursors on
 * prepare_build/probe. @see PageIndex, Cursor
 */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <data_model/intermediate.h>
#include <data_model/plan.h>
#include <foundation/attribute.h>
#include <platform/arena.h>
#include <platform/arena_vector.h>
#include <vector>

/** @brief Branch prediction hint for likely paths. */
#define SPC_LIKELY(x) __builtin_expect(!!(x), 1)
/** @brief Branch prediction hint for unlikely paths. */
#define SPC_UNLIKELY(x) __builtin_expect(!!(x), 0)

/**
 * @namespace Contest::io
 * @brief Columnar I/O: ColumnarReader (cursor caching) + PageIndex (O(log P)
 * lookup).
 * @see Plan, mema
 */
namespace Contest::io {

inline std::atomic<uint64_t> global_build_version{0};
inline std::atomic<uint64_t> global_probe_version{0};

/** @brief Pre-computed page index for O(log P) row lookup in ColumnarTable. */
struct alignas(8) PageIndex {
    /** Cumulative row counts: upper_bound(row_id) yields containing page. */
    platform::ArenaVector<uint32_t> cumulative_rows;

    /** Per-page prefix sums of bitmap popcount for sparse pages.
     * O(1) value index: prefix_sum[chunk] + popcount(word & mask). Empty for
     * dense. */
    std::vector<platform::ArenaVector<uint32_t>> page_prefix_sums;

    /** All pages dense (no NULLs) → skip bitmap checks. */
    bool all_dense = true;

    /** Dense INT32 optimization: all pages dense AND column is INT32.
     * Enables O(1) arithmetic-based page lookup instead of binary search. */
    bool is_dense_int32 = false;

    /** Rows per full page for dense INT32 columns (last page may have fewer).
     */
    uint16_t rows_per_full_page = 0;

    /** Direct pointer to pages vector for O(1) dense INT32 access. */
    std::vector<Page *> const *pages_ptr = nullptr;

    /** Arena for allocation. */
    platform::ThreadArena *arena_ = nullptr;

    /** Default constructor - must call set_arena before build(). */
    PageIndex() = default;

    /** Constructor with arena. */
    explicit PageIndex(platform::ThreadArena &arena)
        : cumulative_rows(arena), arena_(&arena) {}

    /** Move constructor. */
    PageIndex(PageIndex &&other) noexcept
        : cumulative_rows(std::move(other.cumulative_rows)),
          page_prefix_sums(std::move(other.page_prefix_sums)),
          all_dense(other.all_dense), is_dense_int32(other.is_dense_int32),
          rows_per_full_page(other.rows_per_full_page),
          pages_ptr(other.pages_ptr), arena_(other.arena_) {}

    /** Move assignment. */
    PageIndex &operator=(PageIndex &&other) noexcept {
        if (this != &other) {
            cumulative_rows = std::move(other.cumulative_rows);
            page_prefix_sums = std::move(other.page_prefix_sums);
            all_dense = other.all_dense;
            is_dense_int32 = other.is_dense_int32;
            rows_per_full_page = other.rows_per_full_page;
            pages_ptr = other.pages_ptr;
            arena_ = other.arena_;
        }
        return *this;
    }

    /** Deleted copy operations. */
    PageIndex(const PageIndex &) = delete;
    PageIndex &operator=(const PageIndex &) = delete;

    /**
     * @brief Builds page index for a column, computing cumulative row counts.
     *
     * For INT32 columns where all pages are dense (num_rows == num_values),
     * enables O(1) arithmetic-based page lookup instead of binary search.
     */
    inline void build(const Column &column) {
        uint32_t total = 0;
        cumulative_rows.reserve(column.pages.size());
        page_prefix_sums.reserve(column.pages.size());
        all_dense = true;

        /* Dense INT32 detection: track if all pages are dense and uniform */
        bool candidate_dense_int32 = (column.type == DataType::INT32);
        uint16_t first_full_page_rows = 0;

        size_t page_idx = 0;
        for (auto *page_obj : column.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page);

            if (SPC_UNLIKELY(num_rows == 0xfffe)) {
                all_dense = false;
                candidate_dense_int32 = false;
            } else if (SPC_UNLIKELY(num_rows == 0xffff)) {
                all_dense = false;
                candidate_dense_int32 = false;
                total += 1;
            } else {
                total += num_rows;
            }
            cumulative_rows.push_back(total);

            auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
            platform::ArenaVector<uint32_t> prefix_sums(*arena_);

            /* sparse page: build prefix sums for bitmap popcount */
            if (num_rows != 0xfffe && num_rows != 0xffff &&
                num_rows != num_values) {
                all_dense = false;
                candidate_dense_int32 = false;

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

            /* Track rows_per_full_page: use first page as reference */
            if (candidate_dense_int32) {
                if (page_idx == 0) {
                    first_full_page_rows = num_rows;
                } else if (page_idx < column.pages.size() - 1) {
                    /* All full pages (except last) must have same row count */
                    if (num_rows != first_full_page_rows) {
                        candidate_dense_int32 = false;
                    }
                }
            }

            page_idx++;
        }

        /* Finalize dense INT32 metadata */
        is_dense_int32 = candidate_dense_int32 && !column.pages.empty() &&
                         first_full_page_rows > 0;
        if (is_dense_int32) {
            rows_per_full_page = first_full_page_rows;
            pages_ptr = &column.pages;
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

/** @brief Random access reader: O(log P) page lookup + O(1) cursor caching.
 * Separate build/probe indices. */
class ColumnarReader {
  public:
    ColumnarReader() = default;

    /** @brief Cache-aligned cursor: caches page metadata for O(1) sequential
     * access. Thread-local. */
    struct alignas(64) Cursor {
        uint64_t version = 0;
        size_t cached_col = ~0u;
        uint32_t cached_start = 0;
        uint32_t cached_end = 0;
        size_t cached_page = ~0u;

        const int32_t *data_ptr = nullptr;
        const uint8_t *bitmap_ptr = nullptr;
        const uint32_t *prefix_sum_ptr = nullptr;

        int32_t page_idx_val = 0;

        bool is_dense = false;
        bool is_special = false;
        bool col_all_dense = false;

        uint8_t _padding = 0;
    };

    /** @brief Build page indices for build-side columns. Increments
     * global_build_version to invalidate cursors. */
    inline void
    prepare_build(const platform::ArenaVector<const Column *> &columns) {
        auto &arena = platform::get_arena(0);
        build_page_indices.clear();
        build_page_indices.reserve(columns.size());
        for (size_t i = 0; i < columns.size(); ++i) {
            const auto *column = columns[i];
            if (column) {
                PageIndex page_idx(arena);
                page_idx.build(*column);
                build_page_indices.push_back(std::move(page_idx));
            } else {
                PageIndex empty_idx(arena);
                build_page_indices.push_back(std::move(empty_idx));
            }
        }
        global_build_version.fetch_add(1, std::memory_order_relaxed);
    }

    /** @brief Build page indices for probe-side columns. Increments
     * global_probe_version to invalidate cursors. */
    inline void
    prepare_probe(const platform::ArenaVector<const Column *> &columns) {
        auto &arena = platform::get_arena(0);
        probe_page_indices.clear();
        probe_page_indices.reserve(columns.size());
        for (size_t i = 0; i < columns.size(); ++i) {
            const auto *column = columns[i];
            if (column) {
                PageIndex page_idx(arena);
                page_idx.build(*column);
                probe_page_indices.push_back(std::move(page_idx));
            } else {
                PageIndex empty_idx(arena);
                probe_page_indices.push_back(std::move(empty_idx));
            }
        }
        global_probe_version.fetch_add(1, std::memory_order_relaxed);
    }

    // ========================================================================
    // Base Table Page Index Methods (for O(1) deferred column resolution)
    // ========================================================================

    /** @brief Reset base table prepared flags for new query. */
    inline void reset_base_tables() {
        base_table_prepared_.fill(false);
        base_table_version_++;
    }

    /**
     * @brief Prepare page index for a base table column.
     *
     * Called once per unique (table_id, col_idx) before deferred resolution.
     * Enables O(log P) page lookup instead of O(P) linear scan per read.
     *
     * @param table_id Base table ID (0-15).
     * @param col_idx Column index within base table (0-15).
     * @param column The Column to build page index for.
     */
    inline void prepare_base_column(uint8_t table_id, uint8_t col_idx,
                                    const Column &column) {
        size_t idx = (static_cast<size_t>(table_id) << BASE_TABLE_SHIFT) |
                     static_cast<size_t>(col_idx);
        if (idx >= MAX_BASE_TABLE_INDICES)
            return;

        if (!base_table_prepared_[idx]) {
            auto &arena = platform::get_arena(0);
            base_table_indices_[idx] = PageIndex(arena);
            base_table_indices_[idx].build(column);
            base_table_prepared_[idx] = true;
        }
    }

    /** @brief Check if base column page index is prepared. */
    inline bool is_base_column_prepared(uint8_t table_id,
                                        uint8_t col_idx) const {
        size_t idx = (static_cast<size_t>(table_id) << BASE_TABLE_SHIFT) |
                     static_cast<size_t>(col_idx);
        return idx < MAX_BASE_TABLE_INDICES && base_table_prepared_[idx];
    }

    /**
     * @brief Read value from base table using prepared page index.
     *
     * O(1) with cursor caching for sequential access, O(log P) on cache miss.
     * Falls back to O(P) linear scan if page index not prepared.
     *
     * @param column The base table column.
     * @param table_id Base table ID.
     * @param col_idx Column index within base table.
     * @param row_id Row ID within the column.
     * @param data_type Data type of the column.
     * @param cursor Thread-local cursor for caching.
     * @return The value at the specified row.
     */
    inline mema::value_t read_base_table_value(const Column &column,
                                               uint8_t table_id,
                                               uint8_t col_idx, uint32_t row_id,
                                               DataType data_type,
                                               Cursor &cursor) const {
        size_t idx = (static_cast<size_t>(table_id) << BASE_TABLE_SHIFT) |
                     static_cast<size_t>(col_idx);

        if (idx >= MAX_BASE_TABLE_INDICES || !base_table_prepared_[idx]) {
            // Fallback to O(P) linear scan
            return read_value_direct(column, row_id, data_type);
        }

        const PageIndex &page_index = base_table_indices_[idx];

        // Dense INT32 fast path: O(1) arithmetic lookup
        if (data_type == DataType::INT32 && page_index.is_dense_int32) {
            return mema::value_t{read_dense_int32(page_index, row_id)};
        }

        // Check cursor cache (version uses base_table_version_ + idx for
        // uniqueness)
        uint64_t effective_version = base_table_version_ + idx;
        bool cache_hit =
            cursor.version == effective_version && cursor.cached_col == idx &&
            row_id >= cursor.cached_start && row_id < cursor.cached_end;

        if (!cache_hit) {
            // Check sequential access optimization
            if (cursor.version == effective_version &&
                cursor.cached_col == idx && row_id == cursor.cached_end) {
                size_t next_page = cursor.cached_page + 1;
                if (next_page < page_index.cumulative_rows.size()) {
                    load_page_into_cursor_base(column, page_index, next_page,
                                               idx, effective_version, cursor);
                } else {
                    // Past end of data
                    return mema::value_t{mema::value_t::NULL_VALUE};
                }
            } else {
                // Binary search for page
                size_t page_num = page_index.find_page(row_id);
                load_page_into_cursor_base(column, page_index, page_num, idx,
                                           effective_version, cursor);
            }
        }

        // Now cursor is loaded for the correct page
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

    /** @brief Fast path: check cursor cache, dispatch to appropriate handler.
     */
    template <bool IsBuild>
    inline mema::value_t
    read_value_internal(const Column &column, size_t col_idx, uint32_t row_id,
                        DataType data_type, Cursor &cursor) const {

        /* Dense INT32 fast path: O(1) arithmetic lookup, bypasses cursor */
        if (data_type == DataType::INT32) {
            size_t pidx_size =
                IsBuild ? build_page_indices.size() : probe_page_indices.size();
            if (SPC_LIKELY(col_idx < pidx_size)) {
                const PageIndex &page_index = IsBuild
                                                  ? build_page_indices[col_idx]
                                                  : probe_page_indices[col_idx];
                if (SPC_LIKELY(page_index.is_dense_int32)) {
                    return mema::value_t{read_dense_int32(page_index, row_id)};
                }
            }
        }

        uint64_t current_version;
        if constexpr (IsBuild) {
            current_version =
                global_build_version.load(std::memory_order_relaxed);
        } else {
            current_version =
                global_probe_version.load(std::memory_order_relaxed);
        }

        bool cache_hit =
            cursor.version == current_version && col_idx == cursor.cached_col &&
            row_id >= cursor.cached_start && row_id < cursor.cached_end;
        if (SPC_LIKELY(cache_hit)) {
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
        size_t pidx_count =
            IsBuild ? build_page_indices.size() : probe_page_indices.size();
        if (SPC_LIKELY(cursor.version == current_version &&
                       col_idx == cursor.cached_col &&
                       row_id == cursor.cached_end && col_idx < pidx_count)) {
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

    /** @brief Load page metadata into cursor. Caches bounds, pointers, flags
     * for fast-path access. */
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

    /** @brief O(1) sequential optimization: load next page directly when row_id
     * == cursor.cached_end. */
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

    /** @brief Slow path: O(log P) binary search on cursor miss, then load page
     * into cursor. */
    template <bool IsBuild>
    inline mema::value_t read_value_slow(const Column &column, size_t col_idx,
                                         uint32_t row_id, DataType data_type,
                                         Cursor &cursor,
                                         uint64_t current_version) const {

        size_t pidx_size =
            IsBuild ? build_page_indices.size() : probe_page_indices.size();
        if (SPC_UNLIKELY(col_idx >= pidx_size)) {
            // No page index prepared - use direct page read
            return read_value_direct(column, row_id, data_type);
        }
        const PageIndex &page_index =
            IsBuild ? build_page_indices[col_idx] : probe_page_indices[col_idx];
        size_t page_num = page_index.find_page(row_id);
        load_page_into_cursor<IsBuild>(column, page_index, page_num, col_idx,
                                       current_version, cursor);
        return read_value_internal<IsBuild>(column, col_idx, row_id, data_type,
                                            cursor);
    }

    /** @brief Read value: O(1) cursor hit, O(log P) miss. Sequential → O(1)
     * amortized, sparse → popcount. */
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

    /**
     * @brief Direct value read bypassing page index cache.
     *
     * Used for deferred column resolution when reading from base tables
     * that don't have prepared page indices. O(n) page scan per read.
     */
    inline mema::value_t read_value_direct_public(const Column &column,
                                                  uint32_t row_id,
                                                  DataType data_type) const {
        return read_value_direct(column, row_id, data_type);
    }

    inline const PageIndex &get_build_page_index(size_t col_idx) const {
        return build_page_indices[col_idx];
    }

    inline const PageIndex &get_probe_page_index(size_t col_idx) const {
        return probe_page_indices[col_idx];
    }

  private:
    /** @brief O(1) direct read for dense INT32 columns. No cursor, no binary
     * search. */
    inline int32_t read_dense_int32(const PageIndex &page_index,
                                    uint32_t row_id) const {
        uint32_t page_num = row_id / page_index.rows_per_full_page;
        uint32_t local_row =
            row_id - (page_num * page_index.rows_per_full_page);

        auto *page_data = (*page_index.pages_ptr)[page_num]->data;
        return reinterpret_cast<const int32_t *>(page_data + 4)[local_row];
    }

    /**
     * @brief Direct value read without prepared page index.
     *
     * Used when page indices aren't available (e.g., reading base tables
     * during deferred resolution). O(n) page scan - slower than cached path.
     */
    inline mema::value_t read_value_direct(const Column &column,
                                           uint32_t row_id,
                                           DataType data_type) const {
        // Linear scan to find page containing row_id
        uint32_t cumulative = 0;
        for (size_t page_num = 0; page_num < column.pages.size(); ++page_num) {
            auto *page_data = column.pages[page_num]->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page_data);
            auto num_values =
                *reinterpret_cast<const uint16_t *>(page_data + 2);

            // Handle special pages
            if (num_rows == 0xffff) {
                // Long string page - single row
                if (row_id == cumulative) {
                    return mema::value_t::encode_string(
                        static_cast<int32_t>(page_num),
                        mema::value_t::LONG_STRING_OFFSET);
                }
                cumulative += 1;
                continue;
            }
            if (num_rows == 0xfffe) {
                // Skip special marker pages
                continue;
            }

            if (row_id < cumulative + num_rows) {
                // Found the page
                uint32_t local_row = row_id - cumulative;
                bool is_dense = (num_rows == num_values);
                const auto *data_ptr =
                    reinterpret_cast<const int32_t *>(page_data + 4);

                if (is_dense) {
                    if (data_type == DataType::INT32) {
                        return mema::value_t{data_ptr[local_row]};
                    } else {
                        return mema::value_t::encode_string(
                            static_cast<int32_t>(page_num),
                            static_cast<int32_t>(local_row));
                    }
                } else {
                    // Sparse page - check bitmap
                    size_t bitmap_size = (num_rows + 7) / 8;
                    const auto *bitmap_ptr = reinterpret_cast<const uint8_t *>(
                        page_data + PAGE_SIZE - bitmap_size);

                    bool is_valid =
                        bitmap_ptr[local_row >> 3] & (1u << (local_row & 7));
                    if (!is_valid) {
                        return mema::value_t{mema::value_t::NULL_VALUE};
                    }

                    // Compute data index via popcount
                    uint32_t data_idx = 0;
                    for (uint32_t i = 0; i < local_row; ++i) {
                        if (bitmap_ptr[i >> 3] & (1u << (i & 7))) {
                            data_idx++;
                        }
                    }

                    if (data_type == DataType::INT32) {
                        return mema::value_t{data_ptr[data_idx]};
                    } else {
                        return mema::value_t::encode_string(
                            static_cast<int32_t>(page_num),
                            static_cast<int32_t>(data_idx));
                    }
                }
            }
            cumulative += num_rows;
        }
        // Row not found - return NULL
        return mema::value_t{mema::value_t::NULL_VALUE};
    }

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

    /** @brief Load page into cursor for base table access. */
    inline void load_page_into_cursor_base(const Column &column,
                                           const PageIndex &page_index,
                                           size_t page_num, size_t col_idx,
                                           uint64_t version,
                                           Cursor &cursor) const {
        cursor.version = version;
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

    std::vector<PageIndex> build_page_indices;
    std::vector<PageIndex> probe_page_indices;

    // Base table page indices for deferred column resolution.
    // Index = (table_id << 4) | col_idx, supports 16 tables × 16 cols = 256.
    static constexpr size_t BASE_TABLE_SHIFT = 4;
    static constexpr size_t MAX_BASE_TABLE_INDICES = 256;
    std::array<PageIndex, MAX_BASE_TABLE_INDICES> base_table_indices_;
    std::array<bool, MAX_BASE_TABLE_INDICES> base_table_prepared_{};
    uint64_t base_table_version_ = 0;
};
} // namespace Contest::io

namespace Contest {
using Contest::io::ColumnarReader;
using Contest::io::global_build_version;
using Contest::io::global_probe_version;
using Contest::io::PageIndex;
} // namespace Contest

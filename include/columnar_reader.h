#pragma once

#include <algorithm>
#include <atomic>
#include <intermediate.h>
#include <plan.h>
#include <vector>
#include <cstring> // Added for memcpy

/* likely/unlikely macros for branch prediction optimization */
#define SPC_LIKELY(x) __builtin_expect(!!(x), 1)
#define SPC_UNLIKELY(x) __builtin_expect(!!(x), 0)

namespace Contest {

/* global version counters to invalidate thread_local caches across different ColumnarReader instances */
inline std::atomic<uint64_t> global_build_version{0};
inline std::atomic<uint64_t> global_probe_version{0};

/**
 *
 * page index for efficient row lookup in ColumnarTable pages
 * pre-computed metadata to accelerate random access during materialization
 *
 **/
struct PageIndex {
    std::vector<uint32_t> cumulative_rows;
    std::vector<std::vector<uint32_t>> page_prefix_sums;

    /**
     *
     * builds page index for entire column
     * processes all pages to construct cumulative_rows and page_prefix_sums
     *
     **/
    inline void build(const Column &column) {
        uint32_t total = 0;
        page_prefix_sums.reserve(column.pages.size());
        cumulative_rows.reserve(column.pages.size());

        for (auto *page_obj : column.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page);

            if (SPC_UNLIKELY(num_rows == 0xfffe)) {
                /* long string continuation - contributes 0 logical rows */
            } else if (SPC_UNLIKELY(num_rows == 0xffff)) {
                /* long string start - contributes 1 row */
                total += 1;
            } else {
                total += num_rows;
            }
            cumulative_rows.push_back(total);

            auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
            std::vector<uint32_t> prefix_sums;

            /* sparse page check */
            if (num_rows != 0xfffe && num_rows != 0xffff &&
                num_rows != num_values) {
                
                /* FIX: Use uint8_t* to calculate offset safely */
                size_t bitmap_size = (num_rows + 7) / 8;
                auto *bitmap_bytes = reinterpret_cast<const uint8_t *>(
                    page + PAGE_SIZE - bitmap_size);
                
                size_t num_chunks = (num_rows + 63) / 64;

                prefix_sums.reserve(num_chunks);
                uint32_t sum = 0;
                for (size_t i = 0; i < num_chunks; ++i) {
                    prefix_sums.push_back(sum);
                    
                    /* FIX: Safe read to avoid reading past PAGE_SIZE/bitmap end */
                    uint64_t word = 0;
                    size_t offset = i * 8;
                    size_t remaining = bitmap_size > offset ? bitmap_size - offset : 0;
                    size_t bytes_to_read = std::min(remaining, size_t(8));
                    
                    if (bytes_to_read > 0) {
                        std::memcpy(&word, bitmap_bytes + offset, bytes_to_read);
                    }
                    
                    sum += __builtin_popcountll(word);
                }
            }
            page_prefix_sums.push_back(std::move(prefix_sums));
        }
    }

    /**
     *
     * binary search to find which page contains row_id
     * uses cumulative_rows for O(log P) lookup
     *
     **/
    inline size_t find_page(uint32_t row_id) const {
        auto it = std::upper_bound(cumulative_rows.begin(),
                                   cumulative_rows.end(), row_id);
        return static_cast<size_t>(it - cumulative_rows.begin());
    }

    /**
     *
     * returns starting row number for given page
     *
     **/
    inline uint32_t page_start_row(size_t page_num) const {
        return page_num == 0 ? 0 : cumulative_rows[page_num - 1];
    }
};

/**
 *
 * efficient reader for ColumnarTable inputs
 * provides two-level optimization for random row access
 * lazily populated caches exploit spatial locality
 *
 **/
class ColumnarReader {
  public:
    ColumnarReader() = default;

    inline void prepare_build(const std::vector<const Column *> &columns) {
        build_page_indices.clear();
        build_page_indices.reserve(columns.size());
        for (const auto *column : columns) {
            PageIndex page_idx;
            page_idx.build(*column);
            build_page_indices.push_back(std::move(page_idx));
        }
        global_build_version.fetch_add(1, std::memory_order_relaxed);
    }

    inline void prepare_probe(const std::vector<const Column *> &columns) {
        probe_page_indices.clear();
        probe_page_indices.reserve(columns.size());
        for (const auto *column : columns) {
            PageIndex page_idx;
            page_idx.build(*column);
            probe_page_indices.push_back(std::move(page_idx));
        }
        global_probe_version.fetch_add(1, std::memory_order_relaxed);
    }

    /* internal implementation template to deduplicate build/probe logic */
    template <bool IsBuild>
    inline mema::value_t read_value_internal(const Column &column, size_t col_idx,
                                             uint32_t row_id, DataType data_type) const {
        /* thread-local cache */
        thread_local size_t tl_cached_col = ~0u;
        thread_local size_t tl_cached_page = ~0u;
        thread_local uint32_t tl_cached_start = 0;
        thread_local uint32_t tl_cached_end = 0;
        thread_local uint64_t tl_version = 0;

        /* relaxed load is sufficient for cache validation in this context */
        uint64_t current_version;
        if constexpr (IsBuild) {
            current_version = global_build_version.load(std::memory_order_relaxed);
        } else {
            current_version = global_probe_version.load(std::memory_order_relaxed);
        }

        /* fast path: cache hit */
        if (SPC_LIKELY(tl_version == current_version &&
                       col_idx == tl_cached_col && 
                       row_id >= tl_cached_start &&
                       row_id < tl_cached_end)) {
            
            const auto& indices = IsBuild ? build_page_indices : probe_page_indices;
            return read_from_page(column, indices[col_idx],
                                  tl_cached_page,
                                  row_id - tl_cached_start, data_type);
        }

        /* slow path: cache miss / version mismatch */
        const PageIndex &page_index = IsBuild ? build_page_indices[col_idx] 
                                              : probe_page_indices[col_idx];
        
        size_t page_num = page_index.find_page(row_id);
        uint32_t page_start = page_index.page_start_row(page_num);
        uint32_t page_end = page_index.cumulative_rows[page_num];

        tl_version = current_version;
        tl_cached_col = col_idx;
        tl_cached_page = page_num;
        tl_cached_start = page_start;
        tl_cached_end = page_end;

        return read_from_page(column, page_index, page_num, row_id - page_start,
                              data_type);
    }

    inline mema::value_t read_value_build(const Column &column, size_t col_idx,
                                          uint32_t row_id, DataType data_type) const {
        return read_value_internal<true>(column, col_idx, row_id, data_type);
    }

    inline mema::value_t read_value_probe(const Column &column, size_t col_idx,
                                          uint32_t row_id, DataType data_type) const {
        return read_value_internal<false>(column, col_idx, row_id, data_type);
    }

    inline const PageIndex &get_build_page_index(size_t col_idx) const {
        return build_page_indices[col_idx];
    }

    inline const PageIndex &get_probe_page_index(size_t col_idx) const {
        return probe_page_indices[col_idx];
    }

  private:
    inline mema::value_t read_from_page(const Column &column,
                                        const PageIndex &page_index,
                                        size_t page_num, uint32_t local_row,
                                        DataType data_type) const {
        auto *page = column.pages[page_num]->data;
        auto num_rows = *reinterpret_cast<const uint16_t *>(page);
        auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);

        /* check for long string continuation */
        if (SPC_UNLIKELY(num_rows == 0xffff)) {
            return mema::value_t::encode_string(
                static_cast<int32_t>(page_num), mema::value_t::LONG_STRING_OFFSET);
        }

        auto *data_begin = reinterpret_cast<const int32_t *>(page + 4);

        /* dense page optimization (no nulls) */
        if (SPC_LIKELY(num_rows == num_values)) {
            if (data_type == DataType::INT32) {
                return mema::value_t{data_begin[local_row]};
            } else {
                return mema::value_t::encode_string(static_cast<int32_t>(page_num), 
                                                    static_cast<int32_t>(local_row));
            }
        } 
        
        /* sparse page handling */
        /* FIX: Use uint8_t* for correct pointer arithmetic and sizing */
        size_t bitmap_size = (num_rows + 7) / 8;
        auto *bitmap = reinterpret_cast<const uint8_t *>(
            page + PAGE_SIZE - bitmap_size);
        
        /* optimized bit check: replace div/mod with shifts */
        bool is_valid = bitmap[local_row >> 3] & (1u << (local_row & 7));

        if (!is_valid) {
            return mema::value_t{mema::value_t::NULL_VALUE};
        }

        const auto &prefix_sums = page_index.page_prefix_sums[page_num];
        size_t chunk_idx = local_row >> 6;
        size_t bit_offset = local_row & 0x3F;

        uint16_t data_idx = prefix_sums[chunk_idx];

        /* FIX: Safe read to avoid reading past PAGE_SIZE/bitmap end */
        uint64_t word = 0;
        size_t offset = chunk_idx * 8;
        size_t remaining = bitmap_size > offset ? bitmap_size - offset : 0;
        size_t bytes_to_read = std::min(remaining, size_t(8));

        if (bytes_to_read > 0) {
            std::memcpy(&word, bitmap + offset, bytes_to_read);
        }
        
        /* create mask for bits preceding current position */
        uint64_t mask = (1ULL << bit_offset) - 1;
        data_idx += __builtin_popcountll(word & mask);

        if (data_type == DataType::INT32) {
            return mema::value_t{data_begin[data_idx]};
        } else {
            return mema::value_t::encode_string(static_cast<int32_t>(page_num), 
                                                static_cast<int32_t>(data_idx));
        }
    }

    std::vector<PageIndex> build_page_indices;
    std::vector<PageIndex> probe_page_indices;
};
} // namespace Contest

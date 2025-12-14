#pragma once

#include <algorithm>
#include <intermediate.h>
#include <plan.h>
#include <vector>

namespace Contest {

/**
 *
 *  page index for efficient row lookup in ColumnarTable pages
 *  pre-computed metadata to accelerate random access during materialization
 *
 *  cumulative_rows stores running total of rows up to each page
 *  enables binary search to locate which page contains a given row_id
 *
 *  page_prefix_sums accelerates sparse page bitmap navigation
 *  sparse pages have bitmap marking valid non-NULL rows
 *  to find data index for row N requires counting valid bits before it
 *  without optimization this is O(N) bit-by-bit counting
 *  page_prefix_sums divides bitmap into 64-bit chunks with pre-computed
 *  popcounts reduces to O(1) array lookup plus one popcount on final 64-bit chunk
 *
 **/
struct PageIndex {
    std::vector<uint32_t> cumulative_rows;
    std::vector<std::vector<uint32_t>> page_prefix_sums;

    /**
     *
     *  builds page index for entire column
     *  processes all pages to construct cumulative_rows and page_prefix_sums
     *  called once per column during prepare_build or prepare_probe
     *
     **/
    inline void build(const Column &column) {
        uint32_t total = 0;
        page_prefix_sums.reserve(column.pages.size());

        for (auto *page_obj : column.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page);

            if (num_rows == 0xfffe) {
            } else if (num_rows == 0xffff) {
                total += 1;
            } else {
                total += num_rows;
            }
            cumulative_rows.push_back(total);

            auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
            std::vector<uint32_t> prefix_sums;

            if (num_rows != 0xfffe && num_rows != 0xffff &&
                num_rows != num_values) {
                auto *bitmap = reinterpret_cast<const uint64_t *>(
                    page + PAGE_SIZE - (num_rows + 7) / 8);
                size_t num_chunks = (num_rows + 63) / 64;

                prefix_sums.reserve(num_chunks);
                uint32_t sum = 0;
                for (size_t i = 0; i < num_chunks; ++i) {
                    prefix_sums.push_back(sum);
                    sum += __builtin_popcountll(bitmap[i]);
                }
            }
            page_prefix_sums.push_back(std::move(prefix_sums));
        }
    }

    /**
     *
     *  binary search to find which page contains row_id
     *  uses cumulative_rows for O(log P) lookup where P is page count
     *  returns page index that contains the requested row
     *
     **/
    inline size_t find_page(uint32_t row_id) const {
        auto it = std::upper_bound(cumulative_rows.begin(),
                                   cumulative_rows.end(), row_id);
        size_t page_idx = (it - cumulative_rows.begin());
        return page_idx;
    }

    /**
     *
     *  returns starting row number for given page
     *  used to compute local row offset within page
     *
     **/
    inline uint32_t page_start_row(size_t page_num) const {
        return page_num == 0 ? 0 : cumulative_rows[page_num - 1];
    }
};

namespace {

/**
 *
 *  reads value_t from ColumnarTable page at given row id
 *  uses page index for fast page lookup
 *  handles dense and sparse pages with bitmaps
 *  returns encoded string metadata for varchar columns
 *
 **/
inline mema::value_t read_value_from_columnar(const Column &column,
                                              const PageIndex &page_index,
                                              uint32_t row_id,
                                              DataType data_type) {
    size_t page_num = page_index.find_page(row_id);
    uint32_t page_start = page_index.page_start_row(page_num);
    uint32_t local_row = row_id - page_start;

    auto *page = column.pages[page_num]->data;
    auto num_rows = *reinterpret_cast<const uint16_t *>(page);
    auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);

    if (num_rows == 0xffff) {
        return mema::value_t::encode_string(page_num,
                                            mema::value_t::LONG_STRING_OFFSET);
    }

    auto *data_begin = reinterpret_cast<const int32_t *>(page + 4);

    if (num_rows == num_values) {
        if (data_type == DataType::INT32) {
            return mema::value_t{data_begin[local_row]};
        } else {
            return mema::value_t::encode_string(page_num, local_row);
        }
    } else {
        auto *bitmap = reinterpret_cast<const uint8_t *>(page + PAGE_SIZE -
                                                         (num_rows + 7) / 8);
        bool is_valid = bitmap[local_row / 8] & (1u << (local_row % 8));

        if (!is_valid) {
            return mema::value_t{mema::value_t::NULL_VALUE};
        }
        uint16_t data_idx = 0;
        for (uint16_t i = 0; i < local_row; ++i) {
            if (bitmap[i / 8] & (1u << (i % 8))) {
                data_idx++;
            }
        }

        if (data_type == DataType::INT32) {
            return mema::value_t{data_begin[data_idx]};
        } else {
            return mema::value_t::encode_string(page_num, data_idx);
        }
    }
}
} // anonymous namespace

/**
 *
 *  efficient reader for ColumnarTable inputs during materialization
 *  provides two-level optimization for random row access
 *
 *  pre-computed for all pages during prepare_build or prepare_probe
 *  enables fast page lookup and O(1) sparse page data index calculation
 *
 *  lazily populated during reads to exploit spatial locality
 *  when consecutive reads target same page avoids binary search entirely
 *  hash joins produce clustered matches from same hash bucket
 *  nested loops iterate sequentially through pages
 *
 *  maintains separate indices and caches for build and probe sides
 *
 **/
class ColumnarReader {
  public:
    ColumnarReader()
        : build_cached_col(~0u), build_cached_page(~0u), build_cached_start(0),
          build_cached_end(0), probe_cached_col(~0u), probe_cached_page(~0u),
          probe_cached_start(0), probe_cached_end(0) {}

    /**
     *
     *  builds PageIndex for all build side and probe side columns
     *  pre-computes cumulative_rows and page_prefix_sums
     *  resets last accessed page cache to invalid state
     *
     **/
    inline void prepare_build(const std::vector<const Column *> &columns) {
        build_page_indices.clear();
        build_page_indices.reserve(columns.size());
        for (const auto *column : columns) {
            PageIndex page_idx;
            page_idx.build(*column);
            build_page_indices.push_back(std::move(page_idx));
        }
        build_cached_col = ~0u;
        build_cached_page = ~0u;
    }

    inline void prepare_probe(const std::vector<const Column *> &columns) {
        probe_page_indices.clear();
        probe_page_indices.reserve(columns.size());
        for (const auto *column : columns) {
            PageIndex page_idx;
            page_idx.build(*column);
            probe_page_indices.push_back(std::move(page_idx));
        }
        probe_cached_col = ~0u;
        probe_cached_page = ~0u;
    }

    inline mema::value_t read_value_build(const Column &column, size_t col_idx,
                                          uint32_t row_id, DataType data_type) {
        if (col_idx == build_cached_col && row_id >= build_cached_start &&
            row_id < build_cached_end) {
            return read_from_page(column, build_page_indices[col_idx],
                                  build_cached_page,
                                  row_id - build_cached_start, data_type);
        }

        const PageIndex &page_index = build_page_indices[col_idx];
        size_t page_num = page_index.find_page(row_id);
        uint32_t page_start = page_index.page_start_row(page_num);
        uint32_t page_end = page_index.cumulative_rows[page_num];

        build_cached_col = col_idx;
        build_cached_page = page_num;
        build_cached_start = page_start;
        build_cached_end = page_end;

        return read_from_page(column, page_index, page_num, row_id - page_start,
                              data_type);
    }

    inline mema::value_t read_value_probe(const Column &column, size_t col_idx,
                                          uint32_t row_id, DataType data_type) {
        if (col_idx == probe_cached_col && row_id >= probe_cached_start &&
            row_id < probe_cached_end) {
            return read_from_page(column, probe_page_indices[col_idx],
                                  probe_cached_page,
                                  row_id - probe_cached_start, data_type);
        }

        const PageIndex &page_index = probe_page_indices[col_idx];
        size_t page_num = page_index.find_page(row_id);
        uint32_t page_start = page_index.page_start_row(page_num);
        uint32_t page_end = page_index.cumulative_rows[page_num];

        probe_cached_col = col_idx;
        probe_cached_page = page_num;
        probe_cached_start = page_start;
        probe_cached_end = page_end;

        return read_from_page(column, page_index, page_num, row_id - page_start,
                              data_type);
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

        if (num_rows == 0xffff) {
            return mema::value_t::encode_string(
                page_num, mema::value_t::LONG_STRING_OFFSET);
        }

        auto *data_begin = reinterpret_cast<const int32_t *>(page + 4);

        if (num_rows == num_values) {
            if (data_type == DataType::INT32) {
                return mema::value_t{data_begin[local_row]};
            } else {
                return mema::value_t::encode_string(page_num, local_row);
            }
        } else {
            auto *bitmap = reinterpret_cast<const uint8_t *>(
                page + PAGE_SIZE - (num_rows + 7) / 8);
            bool is_valid = bitmap[local_row / 8] & (1u << (local_row % 8));

            if (!is_valid) {
                return mema::value_t{mema::value_t::NULL_VALUE};
            }

            const auto &prefix_sums = page_index.page_prefix_sums[page_num];
            size_t chunk_idx = local_row >> 6;
            size_t bit_offset = local_row & 0x3F;

            uint16_t data_idx = prefix_sums[chunk_idx];

            auto *bitmap64 = reinterpret_cast<const uint64_t *>(bitmap);
            uint64_t mask = (1ULL << bit_offset) - 1;
            data_idx += __builtin_popcountll(bitmap64[chunk_idx] & mask);

            if (data_type == DataType::INT32) {
                return mema::value_t{data_begin[data_idx]};
            } else {
                return mema::value_t::encode_string(page_num, data_idx);
            }
        }
    }

    /* caches for build and probe sides */
    std::vector<PageIndex> build_page_indices;
    size_t build_cached_col;
    size_t build_cached_page;
    uint32_t build_cached_start;
    uint32_t build_cached_end;

    std::vector<PageIndex> probe_page_indices;
    size_t probe_cached_col;
    size_t probe_cached_page;
    uint32_t probe_cached_start;
    uint32_t probe_cached_end;
};
} // namespace Contest

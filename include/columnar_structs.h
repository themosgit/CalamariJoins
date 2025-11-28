#pragma once

#include <common.h>
#include <cstdint>
#include <plan.h>
#include <table.h>
#include <vector>

namespace mema {

/**
 *
 *  value_t struct holding necessary metadata.
 *  for INT32 we store the integer value directly
 *  For VARCHAR we pack 17 bits for page_idx and
 *  15 bits for offset_idx this fits all pages
 *  and allows the neccesary offset index
 *
 **/
struct alignas(4) value_t {
    int32_t value;

    /* encodes string metadata by packing bits */
    static inline value_t encode_string(int32_t page_idx, int32_t offset_idx) {
        return {(offset_idx << 17) | (page_idx & 0x1FFFF)};
    }

    /**
     *
     *  decodes first 17 bits for page_idx casts
     *  to prevent sign extension when shifting
     *  signed integer and decodes 15 bits for
     *  offset idx
     *
     **/
    static inline void decode_string(int32_t encoded, int32_t &page_idx,
                                     int32_t &offset_idx) {
        page_idx = encoded & 0x1FFFF;
        offset_idx = (static_cast<uint32_t>(encoded) >> 17) & 0x7FFF;
    }

    /* when we encounter long string all offset bits are set */
    static constexpr int32_t LONG_STRING_OFFSET = 0x7FFF;
};

constexpr size_t CAP_PER_PAGE = PAGE_SIZE / sizeof(value_t);

/**
 *
 *  CAP_PER_PAGE = 1024 to achieve 100% memory utilization per page
 *  a simple vector of pages with value_t append function that  writes value
 *  sequentially to the end,and also checks if new page is needed.
 *  and also an operator to read the value from the idx
 *
 **/
struct column_t {
  private:
    /* added page alignment */
    struct alignas(PAGE_SIZE) Page {
        value_t data[CAP_PER_PAGE];
    };

    std::vector<Page *> pages;
    size_t num_values = 0;
    size_t num_rows = 0;

    bool direct_access = true;

    std::vector<uint64_t> bitmap;
    std::vector<uint32_t> chunk_prefix_sum;

  public:
    uint8_t source_table = 0;
    uint8_t source_column = 0;

  private:
    inline void set_bit(size_t idx) {
        size_t target_idx = idx >> 6;
        if (target_idx >= bitmap.size()) {
            bitmap.resize((target_idx + 1) * 2, 0);
        }
        bitmap[target_idx] |= (1ULL << (idx & 0x3F));
    }

    inline bool is_bit_set(size_t idx) const {
        size_t target_idx = idx >> 6;
        if (target_idx >= bitmap.size())
            return false;
        return (bitmap[target_idx] & (1ULL << (idx & 0x3F))) != 0;
    }

    inline void backfill_previous_bits() {
        if (num_rows == 0)
            return;

        size_t last_chunk = (num_rows - 1) >> 6;
        bitmap.resize(last_chunk + 1, ~0ULL);

        size_t last_bit_offset = (num_rows - 1) & 0x3F;
        bitmap[last_chunk] = (1ULL << (last_bit_offset + 1)) - 1;
    }

  public:
    column_t() = default;

    ~column_t() {
        for (auto *page : pages)
            delete page;
    }

    /* if we know the size we can pre allocate */
    inline void reserve(size_t expected_rows) {
        pages.reserve((expected_rows + CAP_PER_PAGE - 1) / CAP_PER_PAGE);
    }

    /* appends value to page creates one if needed
     * simplified index calculation sets bitmap as well */
    inline void append(const value_t &val) {
        if (num_values % CAP_PER_PAGE == 0) {
            pages.push_back(new Page());
        }
        pages.back()->data[num_values % CAP_PER_PAGE] = val;
        if (!direct_access)
            set_bit(num_rows);
        num_values++;
        num_rows++;
    }

    /* bulk append for dense data (no nulls) */
    inline void append_bulk(const int32_t *data, size_t count) {
        size_t remaining = count;
        size_t src_idx = 0;

        size_t pages_needed =
            (num_values + count + CAP_PER_PAGE - 1) / CAP_PER_PAGE;
        while (pages.size() < pages_needed) {
            pages.push_back(new Page());
        }

        while (remaining > 0) {
            size_t page_idx = num_values / CAP_PER_PAGE;
            size_t offset = num_values % CAP_PER_PAGE;
            size_t space = CAP_PER_PAGE - offset;
            size_t batch = std::min(space, remaining);

            /* copy batch to page */
            memcpy(&pages[page_idx]->data[offset], &data[src_idx],
                   batch * sizeof(int32_t));

            /* Set bitmap if sparse */
            if (!direct_access) {
                for (size_t i = 0; i < batch; ++i) {
                    set_bit(num_rows + i);
                }
            }

            num_values += batch;
            num_rows += batch;
            src_idx += batch;
            remaining -= batch;
        }
    }

    /* bulk append for string metadata (no nulls) - generates sequential offset
     * indices */
    inline void append_bulk_string(int32_t page_idx, size_t count) {
        size_t remaining = count;
        size_t offset_idx = 0;

        size_t pages_needed =
            (num_values + count + CAP_PER_PAGE - 1) / CAP_PER_PAGE;
        while (pages.size() < pages_needed) {
            pages.push_back(new Page());
        }

        while (remaining > 0) {
            size_t pg_idx = num_values / CAP_PER_PAGE;
            size_t pg_offset = num_values % CAP_PER_PAGE;
            size_t space = CAP_PER_PAGE - pg_offset;
            size_t batch = std::min(space, remaining);

            /* generate and copy batch */
            for (size_t i = 0; i < batch; ++i) {
                pages[pg_idx]->data[pg_offset + i] =
                    value_t::encode_string(page_idx, offset_idx + i);
            }

            if (!direct_access) {
                for (size_t i = 0; i < batch; ++i) {
                    set_bit(num_rows + i);
                }
            }

            num_values += batch;
            num_rows += batch;
            offset_idx += batch;
            remaining -= batch;
        }
    }

    /* when the value is null
     * we just iterate the num rows */
    inline void append_null() {
        if (direct_access) {
            direct_access = false;
            backfill_previous_bits();
        }
        num_rows++;
    }

    const value_t &operator[](size_t idx) const {
        return pages[idx / CAP_PER_PAGE]->data[idx % CAP_PER_PAGE];
    }

    /* navigates to the correct value index by calculating
     * popcount up to that point in the bitmap */
    inline const value_t *get_by_row(size_t row_idx) const {
        if (direct_access) {
            return (row_idx < num_rows) ? &(*this)[row_idx] : nullptr;
        }

        if (!is_bit_set(row_idx))
            return nullptr;

        size_t chunk_idx = row_idx >> 6;
        uint32_t value_idx = 0;
        value_idx = chunk_prefix_sum[chunk_idx];
        size_t bit_offset = row_idx & 0x3F;
        uint64_t mask = (1ULL << bit_offset) - 1;
        value_idx += __builtin_popcountll(bitmap[chunk_idx] & mask);

        return &(*this)[value_idx];
    }

    inline void build_cache() {
        if (direct_access)
            return;

        chunk_prefix_sum.clear();
        chunk_prefix_sum.reserve(bitmap.size());

        uint32_t total_count = 0;
        for (uint64_t chunk : bitmap) {
            chunk_prefix_sum.push_back(total_count);
            total_count += __builtin_popcountll(chunk);
        }
    }

    size_t size() const { return num_values; }
    size_t row_count() const { return num_rows; }
    bool has_direct_access() const { return direct_access; }
};

using Columnar = std::vector<column_t>;
Columnar
copy_scan(const ColumnarTable &table, uint8_t table_idx,
          const std::vector<std::tuple<size_t, DataType>> &output_attrs);

ColumnarTable to_columnar(const Columnar &table, const Plan &plan);
}

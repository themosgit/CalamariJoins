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
 *  For VARCHAR we pack 19 bits for page_idx and
 *  13 bits for offset_idx this fits all pages
 *  and allows the neccesary offset index
 *
 **/
struct alignas(4) value_t {
    int32_t value;

    /* encodes string metadata by packing bits */
    static inline value_t encode_string(int32_t page_idx, int32_t offset_idx) {
        return {(offset_idx << 19) | (page_idx & 0x7FFFF)};
    }

    /**
     *
     *  decodes first 19 bits for page_idx casts
     *  to prevent sign extension when shifting
     *  signed integer and decodes 13 bits for
     *  offset idx
     *
     **/
    static inline void decode_string(int32_t encoded, int32_t &page_idx,
                                     int32_t &offset_idx) {
        page_idx = encoded & 0x7FFFF;
        offset_idx = (static_cast<uint32_t>(encoded) >> 19) & 0x1FFF;
    }

    /* when we encounter long string all offset bits are set */
    static constexpr int32_t LONG_STRING_OFFSET = 0x1FFF;

    /* sentinel value to represent NULL for both INT32 and VARCHAR */
    static constexpr int32_t NULL_VALUE = INT32_MIN;

    inline bool is_null() const { return value == NULL_VALUE; }
};

constexpr size_t CAP_PER_PAGE = PAGE_SIZE / sizeof(value_t);

/**
 *
 *  CAP_PER_PAGE = 2048 to achieve 100% memory utilization per page
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

    size_t num_values = 0;

  public:
    std::vector<Page *> pages;
    uint8_t source_table = 0;
    uint8_t source_column = 0;

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

    /* appends value to page,creates new page if current page is full */
    inline void append(const value_t &val) {
        if (num_values % CAP_PER_PAGE == 0) {
            pages.push_back(new Page());
        }
        pages.back()->data[num_values % CAP_PER_PAGE] = val;
        num_values++;
    }

    /* when the value is null
     * we append the NULL_VALUE sentinel */
    inline void append_null() {
        append(value_t{value_t::NULL_VALUE});
    }

    const value_t &operator[](size_t idx) const {
        return pages[idx / CAP_PER_PAGE]->data[idx % CAP_PER_PAGE];
    }

    size_t size() const { return num_values; }
    size_t row_count() const { return num_values; }
};

using Columnar = std::vector<column_t>;
ColumnarTable to_columnar(const Columnar &table, const Plan &plan);
} // namespace mema

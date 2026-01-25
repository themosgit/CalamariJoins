/**
 *
 * @file intermediate.h
 * @brief Intermediate join format:
 * VARCHAR as page/offset refs (no string copy).
 *
 * construct_intermediate.h
 *
 **/
#pragma once

#include <cstdint>
#include <data_access/table.h>
#include <data_model/plan.h>
#include <foundation/common.h>
#include <platform/arena.h>
#include <vector>

/**
 *
 * @namespace mema
 * @brief Compact join intermediate: value_t (4B) + column_t (16KB pages).
 *
 * value_t: INT32 direct or VARCHAR page/offset ref. column_t: arena-allocated
 * pages with write_at(). @see Contest::ExecuteResult, plan.h ColumnarTable.
 *
 **/
namespace mema {

/**
 *
 * @brief 4-byte value:
 * - INT32 direct,
 * - VARCHAR packed (19-bit page + 13-bit offset)
 *
 * Sentinel values:
 * NULL = INT32_MIN, long string offset = 0x1FFF.
 *
 **/
struct alignas(4) value_t {
    int32_t value;

    /* @brief Encode VARCHAR as page/offset. */
    static inline value_t encode_string(int32_t page_idx, int32_t offset_idx) {
        return {(offset_idx << 19) | (page_idx & 0x7FFFF)};
    }

    /* @brief Decode VARCHAR to page/offset. */
    static inline void decode_string(int32_t encoded, int32_t &page_idx,
                                     int32_t &offset_idx) {

        page_idx = encoded & 0x7FFFF;
        offset_idx = (static_cast<uint32_t>(encoded) >> 19) & 0x1FFF;
    }

    static constexpr int32_t LONG_STRING_OFFSET =
        0x1FFF; /*< Sentinel for long strings. */
    static constexpr int32_t NULL_VALUE =
        INT32_MIN; /*< NULL sentinel for both types. */

    /** @brief Check if this value represents NULL. */
    inline bool is_null() const { return value == NULL_VALUE; }
};

/** @brief Page size for intermediate results (16KB) */
constexpr size_t IR_PAGE_SIZE = 1 << 14;

/** @brief Number of value_t entries per intermediate page. */
constexpr size_t CAP_PER_PAGE = IR_PAGE_SIZE / sizeof(value_t);

/**
 *
 * @brief Intermediate column: 16KB pages of value_t, no header.
 *
 * All pages are arena-allocated (memory freed on arena reset between queries).
 * write_at() is thread-safe after pages are set up; source_table/column track
 * VARCHAR provenance. Move-only.
 *
 * @see construct_intermediate.h, materialize.h
 *
 **/
struct column_t {
  public:
    /* @brief Intermediate page: fixed array of value_t entries. */
    struct alignas(IR_PAGE_SIZE) Page {
        value_t data[CAP_PER_PAGE];
    };

  private:
    size_t num_values = 0; /**< Total value count across all pages. */

  public:
    std::vector<Page *> pages; /**< Pointers to arena-allocated pages. */
    uint8_t source_table = 0; /**< Base table index for VARCHAR dereferencing */
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

    /** 
     *
     * @brief O(1) read: idx>>12 for page, idx&0xFFF for offset.
     * No bounds check.
     *
     **/
    inline const value_t &operator[](size_t idx) const {
        return pages[idx >> 12]->data[idx & 0xFFF];
    }

    /* @brief Total value count. */
    size_t row_count() const { return num_values; }

    /* @brief Pre-allocate pages from arena. */
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

    /* @brief Set row count without allocation. */
    inline void set_row_count(size_t count) { num_values = count; }

    /* @brief Thread-safe write at idx (requires pages to be set up first). */
    inline void write_at(size_t idx, const value_t &val) {
        pages[idx >> 12]->data[idx & 0xFFF] = val;
    }
};

/* @brief Alias for a collection of intermediate columns. */
using Columnar = std::vector<column_t>;

} /* namespace mema */

/** @namespace Contest @brief Contest API. @see execute.cpp */
namespace Contest {
/** @brief Result type for non-root joins (intermediate format). */
using ExecuteResult = std::vector<mema::column_t>;
} /* namespace Contest */

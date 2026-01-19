/**
 * @file intermediate.h
 * @brief Intermediate join format: VARCHAR as page/offset refs (no string copy).
 *
 * Base tables must outlive execution. @see plan.h ColumnarTable, construct_intermediate.h
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
 * @brief Compact join intermediate: value_t (4B) + column_t (16KB pages).
 *
 * value_t: INT32 direct or VARCHAR page/offset ref. column_t: append or
 * parallel write_at. @see Contest::ExecuteResult, plan.h ColumnarTable.
 */
namespace mema {

/**
 * @brief 4-byte value: INT32 direct, VARCHAR packed (19-bit page + 13-bit offset),
 * NULL = INT32_MIN, long string offset = 0x1FFF. Refs valid only while source exists.
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
 * owns_pages=true: new/delete; owns_pages=false: BatchAllocator mmap.
 * append() NOT thread-safe; write_at() safe after pre_allocate().
 * source_table/column track VARCHAR provenance. Move-only.
 *
 * @see construct_intermediate.h, plan.h, materialize.h
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

    /** @brief Resize to count elements. Allocates pages for write_at(). */
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

    /** @brief Append value, auto-allocate. NOT thread-safe. */
    inline void append(const value_t &val) {
        if ((num_values & (CAP_PER_PAGE - 1)) == 0) {
            pages.push_back(new Page());
        }
        pages.back()->data[num_values & (CAP_PER_PAGE - 1)] = val;
        num_values++;
    }

    /** @brief Pre-allocate pages for parallel write_at(). */
    inline void pre_allocate(size_t count) {
        size_t pages_needed = (count + CAP_PER_PAGE - 1) / CAP_PER_PAGE;
        pages.reserve(pages_needed);
        for (size_t i = 0; i < pages_needed; ++i) {
            pages.push_back(new Page());
        }
        num_values = count;
    }

    /** @brief O(1) read: idx>>12 for page, idx&0xFFF for offset. No bounds check. */
    inline const value_t &operator[](size_t idx) const {
        return pages[idx >> 12]->data[idx & 0xFFF];
    }

    /** @brief Total value count. */
    size_t row_count() const { return num_values; }

    /** @brief Pre-allocate from mmap block. owns_pages=false. @see BatchAllocator. */
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

    /** @brief Thread-safe write at idx (requires pre-allocation). */
    inline void write_at(size_t idx, const value_t &val) {
        pages[idx >> 12]->data[idx & 0xFFF] = val;
    }
};

/** @brief Alias for a collection of intermediate columns. */
using Columnar = std::vector<column_t>;

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
} /* namespace Contest */

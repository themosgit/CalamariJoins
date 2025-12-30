#pragma once

#include <common.h>
#include <cstdint>
#include <plan.h>
#include <table.h>
#include <vector>
#include <memory>

namespace mema {

/**
 *
 * value_t struct holding necessary metadata.
 * optimized bit packing for VARCHAR
 *
 **/
struct alignas(4) value_t {
    int32_t value;

    /* packed: offset_idx (13 bits) | page_idx (19 bits) */
    static inline value_t encode_string(int32_t page_idx, int32_t offset_idx) {
        return { (offset_idx << 19) | (page_idx & 0x7FFFF) };
    }

    static inline void decode_string(int32_t encoded, int32_t &page_idx,
                                     int32_t &offset_idx) {
        page_idx = encoded & 0x7FFFF;
        offset_idx = (static_cast<uint32_t>(encoded) >> 19) & 0x1FFF;
    }

    static constexpr int32_t LONG_STRING_OFFSET = 0x1FFF;
    static constexpr int32_t NULL_VALUE = INT32_MIN;

    inline bool is_null() const { return value == NULL_VALUE; }
};

/* Ensuring this aligns with system page size is good for mmap */
constexpr size_t CAP_PER_PAGE = 2048; 

struct column_t {
  private:
    struct alignas(4096) Page {
        value_t data[CAP_PER_PAGE];
    };

    size_t num_values = 0;
    bool owns_pages = true;
    std::shared_ptr<void> external_memory;

  public:
    std::vector<Page *> pages;
    uint8_t source_table = 0;
    uint8_t source_column = 0;

  public:
    column_t() = default;

    column_t(column_t&& other) noexcept 
        : num_values(other.num_values), owns_pages(other.owns_pages), 
          external_memory(std::move(other.external_memory)), 
          pages(std::move(other.pages)), 
          source_table(other.source_table), source_column(other.source_column) {
        other.owns_pages = false;
        other.pages.clear();
        other.num_values = 0;
    }

    column_t& operator=(column_t&& other) noexcept {
        if (this != &other) {
            if (owns_pages) { for (auto *p : pages) delete p; }
            
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

    column_t(const column_t&) = delete;
    column_t& operator=(const column_t&) = delete;

    ~column_t() {
        if (owns_pages) {
            for (auto *page : pages) delete page;
        }
    }

    /**
     *  
     *  Resizes the column to hold `count` elements.
     *  Allocates new pages if necessary. 
     *  Crucial for parallel write_at access.
     *
     **/
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

    inline void append(const value_t &val) {
        if ((num_values & (CAP_PER_PAGE - 1)) == 0) {
             pages.push_back(new Page());
        }
        pages.back()->data[num_values & (CAP_PER_PAGE - 1)] = val;
        num_values++;
    }

    /* thread-safe random write used by parallel construct_intermediate */
    inline void write_at(size_t idx, const value_t &val) {
        pages[idx >> 11]->data[idx & 0x7FF] = val;
    }

    /* read operator */
    const value_t &operator[](size_t idx) const {
        return pages[idx >> 11]->data[idx & 0x7FF];
    }
    
    size_t row_count() const { return num_values; }

    /* pre-allocate from a contiguous memory block (batch allocation) */
    inline void pre_allocate_from_block(void* block, size_t& offset, size_t count,
                                        std::shared_ptr<void> memory_keeper) {
        size_t pages_needed = (count + CAP_PER_PAGE - 1) / CAP_PER_PAGE;
        pages.reserve(pages_needed);
        char* base = static_cast<char*>(block);
        for (size_t i = 0; i < pages_needed; ++i) {
            pages.push_back(reinterpret_cast<Page*>(base + offset));
            offset += sizeof(Page);
        }
        num_values = count;
        owns_pages = false;
        external_memory = memory_keeper;
    }
};

using Columnar = std::vector<column_t>;
ColumnarTable to_columnar(const Columnar &table, const Plan &plan);
} // namespace mema

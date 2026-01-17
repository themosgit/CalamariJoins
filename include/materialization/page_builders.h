#pragma once

#include <algorithm>
#include <cstring>
#include <data_model/intermediate.h>
#include <data_model/plan.h>
#include <functional>
#include <vector>

namespace Contest {

/**
 *
 *  helper to get string data from a varchar column page
 *  returns pointer to string start and its length
 *
 **/
inline std::pair<const char *, uint16_t>
get_string_view(const Column &src_col, int32_t page_idx, int32_t offset_idx) {
    auto *page = reinterpret_cast<uint8_t *>(src_col.pages[page_idx]->data);
    auto num_valid = *reinterpret_cast<uint16_t *>(page + 2);
    auto *offset_array = reinterpret_cast<uint16_t *>(page + 4);
    char *char_begin = reinterpret_cast<char *>(page + 4 + num_valid * 2);

    uint16_t end_off = offset_array[offset_idx];
    uint16_t start_off = (offset_idx == 0) ? 0 : offset_array[offset_idx - 1];

    return {char_begin + start_off, static_cast<uint16_t>(end_off - start_off)};
}

/**
 *
 *  accumulates validity bits for nullable columns
 *  flushes to page bitmap location when complete
 *
 **/
struct alignas(8) BitmapAccumulator {
    std::vector<uint8_t> buffer;
    uint8_t pending_bits = 0;
    int bit_count = 0;

    void reserve(size_t count) {
        buffer.clear();
        buffer.reserve((count + 7) / 8);
    }

    inline void add_bit(bool set) {
        if (set)
            pending_bits |= (1u << bit_count);
        if (++bit_count == 8) {
            buffer.push_back(pending_bits);
            pending_bits = 0;
            bit_count = 0;
        }
    }

    void flush_to_memory(uint8_t *dest_ptr) {
        if (bit_count > 0) {
            buffer.push_back(pending_bits);
            pending_bits = 0;
            bit_count = 0;
        }
        if (!buffer.empty()) {
            std::memcpy(dest_ptr, buffer.data(), buffer.size());
        }
        buffer.clear();
    }

    size_t current_byte_size() const {
        return buffer.size() + (bit_count > 0 ? 1 : 0);
    }
};

/**
 *
 *  builds pages for INT32 columns during materialization
 *  handles null bitmap and page overflow detection
 *
 **/
struct Int32PageBuilder {
    static constexpr size_t MIN_ROWS_PER_PAGE_CHECK = (PAGE_SIZE - 4 - 256) / 5;

    Page *current_page = nullptr;
    int32_t *data_ptr = nullptr;
    std::function<Page *()> alloc_page;
    BitmapAccumulator bitmap;
    uint16_t num_rows = 0;
    uint16_t valid_count = 0;

    explicit Int32PageBuilder(std::function<Page *()> alloc)
        : alloc_page(std::move(alloc)) {}

    void prepare(size_t chunk_matches) { bitmap.reserve(chunk_matches); }

    inline bool add(mema::value_t val) {
        if (!current_page) [[unlikely]] {
            if (num_rows > 0)
                save_to_page(current_page);
            current_page = alloc_page();
            data_ptr = reinterpret_cast<int32_t *>(current_page->data + 4);
        }

        if (!val.is_null()) {
            bitmap.add_bit(true);
            data_ptr[valid_count++] = val.value;
        } else {
            bitmap.add_bit(false);
        }
        num_rows++;
        return false;
    }

    bool should_check_overflow() const {
        size_t est_bitmap = (num_rows + 8) / 8;
        return (num_rows >= 65000) ||
               (4 + (valid_count + 1) * 4 + est_bitmap + 32 > PAGE_SIZE);
    }

    void save_to_page(Page *page_ptr) {
        auto *page = reinterpret_cast<uint8_t *>(page_ptr->data);
        *reinterpret_cast<uint16_t *>(page) = num_rows;
        *reinterpret_cast<uint16_t *>(page + 2) = valid_count;

        size_t bmp_size = bitmap.current_byte_size();
        bitmap.flush_to_memory(page + PAGE_SIZE - bmp_size);

        current_page = nullptr;
        num_rows = 0;
        valid_count = 0;
    }
};

/**
 *
 *  builds pages for VARCHAR columns during materialization
 *  handles variable-length strings, long strings spanning multiple pages,
 *  null bitmap, and page overflow detection
 *
 **/
struct VarcharPageBuilder {
    static constexpr size_t OFFSET_GAP_SIZE = 2048;
    static constexpr size_t MIN_ROWS_PER_PAGE_CHECK = 100;

    Page *current_page = nullptr;
    char *string_write_ptr = nullptr;
    std::function<Page *()> alloc_page;
    size_t current_gap_size = OFFSET_GAP_SIZE;

    std::vector<uint16_t> offsets;
    BitmapAccumulator bitmap;

    uint16_t num_rows = 0;
    size_t current_char_bytes = 0;

    const Column &src_col;

    VarcharPageBuilder(const Column &s, std::function<Page *()> alloc)
        : alloc_page(std::move(alloc)), src_col(s) {}

    void prepare(size_t chunk_matches) {
        offsets.reserve(chunk_matches > 1024 ? 1024 : chunk_matches);
        bitmap.reserve(chunk_matches);
    }

    bool add(mema::value_t val) {
        if (val.is_null()) {
            bitmap.add_bit(false);
            num_rows++;
            return false;
        }

        int32_t page_idx, offset_idx;
        mema::value_t::decode_string(val.value, page_idx, offset_idx);

        if (offset_idx == mema::value_t::LONG_STRING_OFFSET) {
            if (num_rows > 0)
                flush_current_page();
            copy_long_string_pages(page_idx);
            return true;
        }

        auto [str_ptr, str_len] =
            get_string_view(src_col, page_idx, offset_idx);

        if (str_len > PAGE_SIZE - 512) {
            if (num_rows > 0)
                flush_current_page();
            save_long_string_buffer(str_ptr, str_len);
            return true;
        }

        if (!current_page)
            init_new_page();

        size_t ptr_offset =
            string_write_ptr - reinterpret_cast<char *>(current_page->data);
        size_t physical_space = PAGE_SIZE - ptr_offset - 64;

        size_t needed = 4 + (offsets.size() + 1) * 2 + current_char_bytes +
                        str_len + bitmap.current_byte_size() + 10;

        bool flushed = false;
        if (num_rows == 65535 || needed > PAGE_SIZE ||
            (offsets.size() * 2 >= current_gap_size) ||
            str_len > physical_space) {
            flush_current_page();
            init_new_page();
            flushed = true;
            if (str_len > PAGE_SIZE - OFFSET_GAP_SIZE - 100) {
                size_t reduced_gap = 256;
                string_write_ptr = reinterpret_cast<char *>(current_page->data +
                                                            4 + reduced_gap);
                current_gap_size = reduced_gap;
            }
        }

        std::memcpy(string_write_ptr, str_ptr, str_len);
        string_write_ptr += str_len;
        current_char_bytes += str_len;

        offsets.push_back(static_cast<uint16_t>(current_char_bytes));
        bitmap.add_bit(true);
        num_rows++;

        return flushed;
    }

    bool should_check_overflow() const {
        if (!current_page)
            return false;
        size_t ptr_offset = reinterpret_cast<uint8_t *>(string_write_ptr) -
                            reinterpret_cast<uint8_t *>(current_page->data);
        return (num_rows >= 65000) ||
               (ptr_offset + bitmap.current_byte_size() + 100 > PAGE_SIZE);
    }

    void save_to_page(Page *page_ptr) {
        if (num_rows > 0 && current_page == page_ptr) {
            flush_current_page();
            if (current_page == nullptr) {
                num_rows = 0;
                offsets.clear();
                bitmap.buffer.clear();
                current_char_bytes = 0;
            }
        }
    }

  private:
    void init_new_page() {
        current_page = alloc_page();
        current_gap_size = OFFSET_GAP_SIZE;
        string_write_ptr =
            reinterpret_cast<char *>(current_page->data + 4 + OFFSET_GAP_SIZE);
        current_char_bytes = 0;
        num_rows = 0;
        offsets.clear();
    }

    void flush_current_page() {
        if (current_page && num_rows > 0) {
            finalize_page();
        }
        current_page = nullptr;
        offsets.clear();
        bitmap.buffer.clear();
        current_char_bytes = 0;
    }

    void finalize_page() {
        uint8_t *page_base = reinterpret_cast<uint8_t *>(current_page->data);
        size_t offsets_size = offsets.size() * 2;
        char *chars_start_actual =
            reinterpret_cast<char *>(page_base + 4 + offsets_size);
        char *chars_gap_end = string_write_ptr;

        *reinterpret_cast<uint16_t *>(page_base) = num_rows;
        *reinterpret_cast<uint16_t *>(page_base + 2) =
            static_cast<uint16_t>(offsets.size());

        std::memcpy(page_base + 4, offsets.data(), offsets_size);

        if (current_char_bytes > 0) {
            std::memmove(chars_start_actual, chars_gap_end - current_char_bytes,
                         current_char_bytes);
        }

        size_t bmp_size = bitmap.current_byte_size();
        bitmap.flush_to_memory(page_base + PAGE_SIZE - bmp_size);
    }

    void copy_long_string_pages(int32_t start_page_idx) {
        int32_t curr_idx = start_page_idx;
        while (true) {
            auto *src = src_col.pages[curr_idx]->data;
            auto *dest = alloc_page()->data;
            std::memcpy(dest, src, PAGE_SIZE);
            if (++curr_idx >= static_cast<int32_t>(src_col.pages.size()))
                break;
            if (*reinterpret_cast<uint16_t *>(src_col.pages[curr_idx]->data) !=
                0xfffe)
                break;
        }
        num_rows = 0;
        offsets.clear();
        bitmap.buffer.clear();
        current_char_bytes = 0;
        current_page = nullptr;
    }

    void save_long_string_buffer(const char *data_ptr, size_t total_len) {
        size_t offset = 0;
        bool first_page = true;
        while (offset < total_len) {
            auto *page = alloc_page()->data;
            *reinterpret_cast<uint16_t *>(page) = first_page ? 0xffff : 0xfffe;
            first_page = false;
            size_t len = std::min(total_len - offset, PAGE_SIZE - 4);
            *reinterpret_cast<uint16_t *>(page + 2) =
                static_cast<uint16_t>(len);
            std::memcpy(page + 4, data_ptr + offset, len);
            offset += len;
        }
        num_rows = 0;
        offsets.clear();
        bitmap.buffer.clear();
        current_char_bytes = 0;
        current_page = nullptr;
    }
};

} /* namespace Contest */

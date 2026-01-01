#pragma once

#include <columnar_reader.h>
#include <construct_intermediate.h>
#include <cstring>
#include <intermediate.h>
#include <join_setup.h>
#include <plan.h>
#include <vector>
#include <worker_pool.h>
#include <sys/mman.h>
#include <variant>
#include <match_collector.h>
#include <algorithm>
#include <functional>


/* work here will be done after match collector is optimized and finalized
 * we mainly want to look at the memcpy happening in string reconstruction */

namespace Contest {

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

struct BitmapAccumulator {
    std::vector<uint8_t> buffer;
    uint8_t pending_bits = 0;
    int bit_count = 0;
    void reserve(size_t count) {
        buffer.clear();
        buffer.reserve((count + 7) / 8);
    }
    inline void add_bit(bool set) {
        if (set) pending_bits |= (1u << bit_count);
        if (++bit_count == 8) {
            buffer.push_back(pending_bits);
            pending_bits = 0;
            bit_count = 0;
        }
    }
    void flush_to_memory(uint8_t* dest_ptr) {
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

struct Int32PageBuilder {
    static constexpr size_t MIN_ROWS_PER_PAGE_CHECK = (PAGE_SIZE - 4 - 256) / 5;
    
    Page* current_page = nullptr;
    int32_t* data_ptr = nullptr;
    std::function<Page*()> alloc_page;
    BitmapAccumulator bitmap;
    uint16_t num_rows = 0;
    uint16_t valid_count = 0;

    explicit Int32PageBuilder(std::function<Page*()> alloc) : alloc_page(std::move(alloc)) {}

    void prepare(size_t chunk_matches) {
        bitmap.reserve(chunk_matches);
    }

    inline bool add(mema::value_t val) {
        if (!current_page) [[unlikely]] {
            if (num_rows > 0) save_to_page(current_page);
            current_page = alloc_page();
            data_ptr = reinterpret_cast<int32_t*>(current_page->data + 4);
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
        return (num_rows >= 65000) || (4 + (valid_count + 1) * 4 + est_bitmap + 32 > PAGE_SIZE);
    }

    void save_to_page(Page* page_ptr) {
        auto *page = reinterpret_cast<uint8_t*>(page_ptr->data);
        *reinterpret_cast<uint16_t *>(page) = num_rows;
        *reinterpret_cast<uint16_t *>(page + 2) = valid_count;
        
        size_t bmp_size = bitmap.current_byte_size();
        bitmap.flush_to_memory(page + PAGE_SIZE - bmp_size);

        current_page = nullptr; 
        num_rows = 0;
        valid_count = 0;
    }
};

struct VarcharPageBuilder {
    static constexpr size_t OFFSET_GAP_SIZE = 2048; 
    static constexpr size_t MIN_ROWS_PER_PAGE_CHECK = 100;

    Page* current_page = nullptr;
    char* string_write_ptr = nullptr;
    std::function<Page*()> alloc_page;
    size_t current_gap_size = OFFSET_GAP_SIZE;
    
    std::vector<uint16_t> offsets;
    BitmapAccumulator bitmap;
    
    uint16_t num_rows = 0;
    size_t current_char_bytes = 0;
    
    const Column& src_col;

    VarcharPageBuilder(const Column& s, std::function<Page*()> alloc) : alloc_page(std::move(alloc)), src_col(s) {}

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
            if (num_rows > 0) flush_current_page();
            copy_long_string_pages(page_idx);
            return true; 
        } 

        auto [str_ptr, str_len] = get_string_view(src_col, page_idx, offset_idx);

        if (str_len > PAGE_SIZE - 512) {
            if (num_rows > 0) flush_current_page();
            save_long_string_buffer(str_ptr, str_len);
            return true;
        }

        if (!current_page) init_new_page();

        size_t ptr_offset = string_write_ptr - reinterpret_cast<char*>(current_page->data);
        size_t physical_space = PAGE_SIZE - ptr_offset - 64; 

        size_t needed = 4 + (offsets.size() + 1) * 2 + current_char_bytes + str_len + bitmap.current_byte_size() + 10;
        
        bool flushed = false;
        if (num_rows == 65535 || needed > PAGE_SIZE || (offsets.size() * 2 >= current_gap_size) || str_len > physical_space) {
            flush_current_page();
            init_new_page();
            flushed = true;
            if (str_len > PAGE_SIZE - OFFSET_GAP_SIZE - 100) {
                 size_t reduced_gap = 256;
                 string_write_ptr = reinterpret_cast<char*>(current_page->data + 4 + reduced_gap);
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
        if (!current_page) return false;
        size_t ptr_offset = reinterpret_cast<uint8_t*>(string_write_ptr) - reinterpret_cast<uint8_t*>(current_page->data);
        return (num_rows >= 65000) || (ptr_offset + bitmap.current_byte_size() + 100 > PAGE_SIZE);
    }

    void save_to_page(Page* page_ptr) {
        if (num_rows > 0 && current_page == page_ptr) {
            flush_current_page();
            if (current_page == nullptr) {
                num_rows = 0; offsets.clear(); bitmap.buffer.clear(); current_char_bytes = 0;
            }
        }
    }

private:
    void init_new_page() {
        current_page = alloc_page();
        current_gap_size = OFFSET_GAP_SIZE;
        string_write_ptr = reinterpret_cast<char*>(current_page->data + 4 + OFFSET_GAP_SIZE);
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
        uint8_t* page_base = reinterpret_cast<uint8_t*>(current_page->data);
        size_t offsets_size = offsets.size() * 2;
        char* chars_start_actual = reinterpret_cast<char*>(page_base + 4 + offsets_size);
        char* chars_gap_end = string_write_ptr;
        
        *reinterpret_cast<uint16_t *>(page_base) = num_rows;
        *reinterpret_cast<uint16_t *>(page_base + 2) = static_cast<uint16_t>(offsets.size());

        std::memcpy(page_base + 4, offsets.data(), offsets_size);

        if (current_char_bytes > 0) {
            std::memmove(chars_start_actual, chars_gap_end - current_char_bytes, current_char_bytes);
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
            if (++curr_idx >= static_cast<int32_t>(src_col.pages.size())) break;
            if (*reinterpret_cast<uint16_t *>(src_col.pages[curr_idx]->data) != 0xfffe) break;
        }
        num_rows = 0; offsets.clear(); bitmap.buffer.clear(); current_char_bytes = 0; current_page = nullptr;
    }

    void save_long_string_buffer(const char* data_ptr, size_t total_len) {
        size_t offset = 0;
        bool first_page = true;
        while (offset < total_len) {
            auto *page = alloc_page()->data;
            *reinterpret_cast<uint16_t *>(page) = first_page ? 0xffff : 0xfffe;
            first_page = false;
            size_t len = std::min(total_len - offset, PAGE_SIZE - 4);
            *reinterpret_cast<uint16_t *>(page + 2) = static_cast<uint16_t>(len);
            std::memcpy(page + 4, data_ptr + offset, len);
            offset += len;
        }
        num_rows = 0; offsets.clear(); bitmap.buffer.clear(); current_char_bytes = 0; current_page = nullptr;
    }
};

/**
 *
 *  generic parallel column materialization driver
 *  handles mmap allocation, threading, and merging logic
 *
 **/
template <typename BuilderType, typename ReaderFunc, typename InitBuilderFunc>
inline void materialize_column(Column &dest_col,
                                        const MatchCollector &collector,
                                        ReaderFunc &&read_value,
                                        InitBuilderFunc &&init_builder,
                                        bool from_build,
                                        size_t est_bytes_per_row) {
    const size_t total_matches = collector.size();
    if (total_matches == 0) return;
    const auto& matches_vec = const_cast<MatchCollector&>(collector).get_flattened_matches();
    const uint64_t *matches_ptr = matches_vec.data();

    constexpr int num_threads = SPC__CORE_COUNT;

    size_t matches_per_thread = (total_matches + num_threads - 1) / num_threads;
    size_t usable_per_page = PAGE_SIZE - 256; 
    size_t rows_per_page = std::max(1ul, usable_per_page / est_bytes_per_row);
    size_t pages_per_thread = (matches_per_thread + rows_per_page - 1) / rows_per_page + 10; 
    size_t total_pages = pages_per_thread * num_threads;

    void* page_memory = mmap(nullptr, total_pages * PAGE_SIZE,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page_memory == MAP_FAILED) throw std::bad_alloc();

    std::vector<Column> thread_columns;
    thread_columns.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_columns.emplace_back(dest_col.type);
    }

    worker_pool.execute([&](size_t t, size_t num_threads) {
        size_t start = t * total_matches / num_threads;
        size_t end = (t + 1) * total_matches / num_threads;
        if (start >= end) return;

        Column &local_col = thread_columns[t];
        
        size_t thread_page_start = t * pages_per_thread;
        size_t thread_page_limit = pages_per_thread;
        size_t used_pages = 0;

        ColumnarReader::Cursor cursor;

        auto page_allocator = [&]() -> Page* {
            Page* p;
            if (used_pages < thread_page_limit) {
                p = reinterpret_cast<Page*>(
                    static_cast<char*>(page_memory) + (thread_page_start + used_pages) * PAGE_SIZE);
                used_pages++;
            } else {
                p = new Page();
            }
            local_col.pages.push_back(p);
            return p;
        };

        BuilderType builder = init_builder(page_allocator);
        builder.prepare(end - start);

        const size_t check_interval = BuilderType::MIN_ROWS_PER_PAGE_CHECK;
        size_t rows_since_check = 0;
        
        auto get_row_id = [from_build](uint64_t m) -> uint32_t {
            return from_build ? static_cast<uint32_t>(m) 
                              : static_cast<uint32_t>(m >> 32);
        };

        for (size_t i = start; i < end; ++i) {
            uint32_t row_id = get_row_id(matches_ptr[i]);
            
            bool flushed = builder.add(read_value(row_id, cursor));
            
            if (flushed) {
                rows_since_check = 0;
            } else {
                rows_since_check++;
                if (rows_since_check >= check_interval) {
                    if (builder.should_check_overflow()) {
                        builder.save_to_page(builder.current_page);
                        rows_since_check = 0;
                    }
                    if (rows_since_check > check_interval * 2) rows_since_check = 0; 
                }
            }
        }

        if (builder.num_rows != 0) {
            builder.save_to_page(builder.current_page);
        }
    });

    for (auto &thread_col : thread_columns) {
        for (auto *page : thread_col.pages) {
            dest_col.pages.push_back(page);
        }
        thread_col.pages.clear();
    }

    auto* mapped_mem = new MappedMemory(page_memory, total_pages * PAGE_SIZE);
    dest_col.assign_mapped_memory(mapped_mem);
}
/**
 *
 *  materializes final ColumnarTable from mixed inputs
 *  dispatches to specific parallel builders based on type
 *
 **/
inline ColumnarTable materialize(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, const Plan &plan) {

    ColumnarTable result;
    result.num_rows = collector.size();
    
    if (collector.size() == 0) {
        for (auto [_, dtype] : remapped_attrs) result.columns.emplace_back(dtype);
        return result;
    }

    for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
        auto [col_idx, data_type] = remapped_attrs[out_idx];
        bool from_build = col_idx < build_size;
        size_t remapped_col_idx = from_build ? col_idx : col_idx - build_size;

        result.columns.emplace_back(data_type);
        Column &dest_col = result.columns.back();

        const Column *col_source = nullptr;
        const mema::column_t *inter_source = nullptr;
        const JoinInput& input = from_build ? build_input : probe_input;
        const PlanNode& node = from_build ? build_node : probe_node;

        if (input.is_columnar()) {
            auto *table = std::get<const ColumnarTable *>(input.data);
            auto [actual_idx, _] = node.output_attrs[remapped_col_idx];
            col_source = &table->columns[actual_idx];
        } else {
            const auto &res = std::get<ExecuteResult>(input.data);
            inter_source = &res[remapped_col_idx];
        }

        if (data_type == DataType::INT32) {
            auto reader = [&](uint32_t rid, ColumnarReader::Cursor &cursor) {
                if (col_source) {
                     return from_build 
                        ? columnar_reader.read_value_build(*col_source, remapped_col_idx, rid, DataType::INT32, cursor)
                        : columnar_reader.read_value_probe(*col_source, remapped_col_idx, rid, DataType::INT32, cursor);
                }
                return (*inter_source)[rid];
            };

            auto init = [](std::function<Page*()> alloc) { return Int32PageBuilder(std::move(alloc)); };

            materialize_column<Int32PageBuilder>(
                dest_col, collector, reader, init, from_build, 4);

        } else {
            const Column *str_src_ptr = col_source;
            if (!str_src_ptr && inter_source) {
                str_src_ptr = &plan.inputs[inter_source->source_table]
                                   .columns[inter_source->source_column];
            }

            auto reader = [&](uint32_t rid, ColumnarReader::Cursor & cursor) {
                if (col_source) {
                    return from_build
                        ? columnar_reader.read_value_build(*col_source, remapped_col_idx, rid, DataType::VARCHAR, cursor)
                        : columnar_reader.read_value_probe(*col_source, remapped_col_idx, rid, DataType::VARCHAR, cursor);
                }
                return (*inter_source)[rid];
            };

            auto init = [str_src_ptr](std::function<Page*()> alloc) { 
                return VarcharPageBuilder(*str_src_ptr, std::move(alloc)); 
            };

            materialize_column<VarcharPageBuilder>(
                dest_col, collector, reader, init, from_build, 35);
        }
    }
    return result;
}
inline ColumnarTable create_empty_result(
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs) {
    ColumnarTable empty_result;
    empty_result.num_rows = 0;
    for (auto [_, data_type] : remapped_attrs) {
        empty_result.columns.emplace_back(data_type);
    }
    return empty_result;
}

} // namespace Contest

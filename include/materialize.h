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

namespace Contest {

/**
 *
 *  extracts string view from ColumnarTable page
 *  decodes offset array to find string boundaries
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
 *  computes chunk size for materialization parallelization
 *
 **/
inline size_t compute_materialization_chunk_size(size_t total_matches) {
    constexpr int NUM_CORES = SPC__CORE_COUNT;
    constexpr size_t L2_CACHE_SIZE = SPC__LEVEL2_CACHE_SIZE;
    size_t per_core_l2 = L2_CACHE_SIZE / NUM_CORES;
    size_t target_bytes = per_core_l2 / 2;
    size_t bytes_per_match = 72;
    size_t cache_optimal = target_bytes / bytes_per_match;
    size_t min_chunk = (total_matches + NUM_CORES * 4 - 1) / (NUM_CORES * 4);
    return std::max(cache_optimal, min_chunk);
}

/**
 *
 *  helper for bitmap accumulation
 *  buffers bits until byte is full then flushes
 *
 **/
struct BitmapAccumulator {
    std::vector<uint8_t> buffer;
    uint8_t pending_bits = 0;
    int bit_count = 0;

    void reserve(size_t count) {
        buffer.reserve((count + 7) / 8);
    }

    void add_bit(bool set) {
        if (set) pending_bits |= (1u << bit_count);
        if (++bit_count == 8) {
            buffer.push_back(pending_bits);
            pending_bits = 0;
            bit_count = 0;
        }
    }

    void flush() {
        if (bit_count > 0) {
            buffer.push_back(pending_bits);
            pending_bits = 0;
            bit_count = 0;
        }
    }

    void clear() {
        buffer.clear();
        pending_bits = 0;
        bit_count = 0;
    }
};

/**
 *
 *  strategy for building int32 pages
 *  checks overflow via fixed row count heuristic
 *
 **/
struct Int32PageBuilder {
    static constexpr size_t MIN_ROWS_PER_PAGE_CHECK = (PAGE_SIZE - 4 - 256) / 5;
    
    std::vector<int32_t> data;
    BitmapAccumulator bitmap;
    uint16_t num_rows = 0;

    void prepare(size_t chunk_matches) {
        data.reserve(chunk_matches);
        bitmap.reserve(chunk_matches);
    }

    bool add(mema::value_t val) {
        if (!val.is_null()) {
            bitmap.add_bit(true);
            data.push_back(val.value);
        } else {
            bitmap.add_bit(false);
        }
        num_rows++;
        return false;
    }

    bool should_check_overflow() const {
        return 4 + (data.size() + 1) * 4 + (bitmap.buffer.size() + 2) > PAGE_SIZE;
    }

    void save_to_page(Page* page_ptr) {
        bitmap.flush();
        auto *page = page_ptr->data;
        *reinterpret_cast<uint16_t *>(page) = num_rows;
        *reinterpret_cast<uint16_t *>(page + 2) = static_cast<uint16_t>(data.size());
        
        std::memcpy(page + 4, data.data(), data.size() * 4);
        
        size_t bmp_size = bitmap.buffer.size();
        std::memcpy(page + PAGE_SIZE - bmp_size, bitmap.buffer.data(), bmp_size);

        num_rows = 0;
        data.clear();
        bitmap.clear();
    }
};

/**
 *
 *  strategy for building varchar pages
 *  handles long strings and page copying logic
 *
 **/
struct VarcharPageBuilder {
    static constexpr size_t MIN_ROWS_PER_PAGE_CHECK = 100;

    std::vector<char> char_data;
    std::vector<uint16_t> offsets;
    BitmapAccumulator bitmap;
    
    uint16_t num_rows = 0;
    size_t current_char_size = 0;

    const Column& src_col;
    Column& local_col;

    VarcharPageBuilder(const Column& s, Column& l) : src_col(s), local_col(l) {}

    void prepare(size_t chunk_matches) {
        constexpr size_t AVG_STRING_SIZE = 32;
        char_data.reserve(chunk_matches * AVG_STRING_SIZE);
        offsets.reserve(chunk_matches);
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
            if (num_rows > 0) save_to_page(local_col.new_page());
            copy_long_string_pages(page_idx);
            return true; 
        } 

        auto [str_ptr, str_len] = get_string_view(src_col, page_idx, offset_idx);

        if (str_len > PAGE_SIZE - 7) {
            if (num_rows > 0) save_to_page(local_col.new_page());
            save_long_string_buffer(std::string(str_ptr, str_len));
            return true;
        }

        size_t needed = 4 + (offsets.size() + 1) * 2 + 
                        (current_char_size + str_len) + (bitmap.buffer.size() + 2);
        
        bool flushed = false;
        if (needed > PAGE_SIZE) {
            save_to_page(local_col.new_page());
            flushed = true;
        }

        if (current_char_size + str_len > char_data.size()) {
            char_data.resize(std::max(char_data.size() * 2, current_char_size + str_len + 4096));
        }
        std::memcpy(char_data.data() + current_char_size, str_ptr, str_len);
        current_char_size += str_len;
        offsets.push_back(static_cast<uint16_t>(current_char_size));
        
        bitmap.add_bit(true);
        num_rows++;
        return flushed;
    }

    bool should_check_overflow() const {
        return 4 + offsets.size() * 2 + current_char_size + (bitmap.buffer.size() + 2) > PAGE_SIZE;
    }

    void save_to_page(Page* page_ptr) {
        bitmap.flush();
        auto *page = page_ptr->data;
        *reinterpret_cast<uint16_t *>(page) = num_rows;
        *reinterpret_cast<uint16_t *>(page + 2) = static_cast<uint16_t>(offsets.size());
        
        std::memcpy(page + 4, offsets.data(), offsets.size() * 2);
        std::memcpy(page + 4 + offsets.size() * 2, char_data.data(), current_char_size);
        
        size_t bmp_size = bitmap.buffer.size();
        std::memcpy(page + PAGE_SIZE - bmp_size, bitmap.buffer.data(), bmp_size);

        num_rows = 0;
        current_char_size = 0;
        offsets.clear();
        bitmap.clear();
    }

private:
    void copy_long_string_pages(int32_t start_page_idx) {
        int32_t curr_idx = start_page_idx;
        while (true) {
            auto *src = src_col.pages[curr_idx]->data;
            auto *dest = local_col.new_page()->data;
            std::memcpy(dest, src, PAGE_SIZE);

            if (++curr_idx >= static_cast<int32_t>(src_col.pages.size())) break;
            if (*reinterpret_cast<uint16_t *>(src_col.pages[curr_idx]->data) != 0xfffe) break;
        }
    }

    void save_long_string_buffer(const std::string &str) {
        size_t offset = 0;
        bool first_page = true;
        while (offset < str.size()) {
            auto *page = local_col.new_page()->data;
            *reinterpret_cast<uint16_t *>(page) = first_page ? 0xffff : 0xfffe;
            first_page = false;
            size_t len = std::min(str.size() - offset, PAGE_SIZE - 4);
            *reinterpret_cast<uint16_t *>(page + 2) = static_cast<uint16_t>(len);
            std::memcpy(page + 4, str.data() + offset, len);
            offset += len;
        }
    }
};

/**
 *
 *  generic parallel column materialization driver
 *  handles mmap allocation, threading, and merging logic
 *
 **/
template <typename BuilderType, typename ReaderFunc, typename InitBuilderFunc>
inline void materialize_column_parallel(Column &dest_col,
                                        const MatchCollector &collector,
                                        ReaderFunc &&read_value,
                                        InitBuilderFunc &&init_builder,
                                        bool from_build,
                                        size_t est_bytes_per_row) {
    const size_t total_matches = collector.size();
    if (total_matches == 0) return;

    const uint64_t *matches_ptr = collector.matches.data();
    constexpr int num_threads = SPC__CORE_COUNT;

    /* estimate pages per thread conservatively */
    size_t matches_per_thread = (total_matches + num_threads - 1) / num_threads;
    size_t usable_per_page = PAGE_SIZE - 256; 
    size_t rows_per_page = std::max(1ul, usable_per_page / est_bytes_per_row);
    size_t pages_per_thread = (matches_per_thread + rows_per_page - 1) / rows_per_page + 2; 
    size_t total_pages = pages_per_thread * num_threads;

    /* batch allocate all pages upfront */
    void* page_memory = mmap(nullptr, total_pages * PAGE_SIZE,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page_memory == MAP_FAILED) throw std::bad_alloc();

    std::vector<Column> thread_columns;
    thread_columns.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_columns.emplace_back(dest_col.type);
        /* pre-assign pages to each thread's column */
        for (size_t p = 0; p < pages_per_thread; ++p) {
            size_t page_idx = i * pages_per_thread + p;
            Page* page = reinterpret_cast<Page*>(
                static_cast<char*>(page_memory) + page_idx * PAGE_SIZE);
            thread_columns[i].pages.push_back(page);
        }
    }

    worker_pool.execute([&](size_t t, size_t num_threads) {
        size_t start = t * total_matches / num_threads;
        size_t end = (t + 1) * total_matches / num_threads;
        if (start >= end) return;

        Column &local_col = thread_columns[t];
        
        BuilderType builder = init_builder(local_col);
        builder.prepare(end - start);

        const size_t check_interval = BuilderType::MIN_ROWS_PER_PAGE_CHECK;
        size_t rows_since_check = 0;
        
        auto get_row_id = [from_build](uint64_t m) -> uint32_t {
            return from_build ? static_cast<uint32_t>(m) 
                              : static_cast<uint32_t>(m >> 32);
        };

        for (size_t i = start; i < end; ++i) {
            uint32_t row_id = get_row_id(matches_ptr[i]);
            
            bool flushed = builder.add(read_value(row_id));
            
            if (flushed) {
                rows_since_check = 0;
            } else {
                rows_since_check++;
                if (rows_since_check >= check_interval) {
                    if (builder.should_check_overflow()) {
                        builder.save_to_page(local_col.new_page());
                        rows_since_check = 0;
                    }
                    if (rows_since_check > check_interval * 2) rows_since_check = 0; 
                }
            }
        }

        if (builder.num_rows != 0) {
            builder.save_to_page(local_col.new_page());
        }
    });

    /* merge phase: move pages from thread-local columns to dest_col */
    for (auto &thread_col : thread_columns) {
        for (auto *page : thread_col.pages) {
            dest_col.pages.push_back(page);
        }
        thread_col.pages.clear();
    }

    /* assign mapped memory so dest_col will munmap on destruction */
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

        /* resolve source columns or intermediates */
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
            auto reader = [&](uint32_t rid) {
                if (col_source) {
                     return from_build 
                        ? columnar_reader.read_value_build(*col_source, remapped_col_idx, rid, DataType::INT32)
                        : columnar_reader.read_value_probe(*col_source, remapped_col_idx, rid, DataType::INT32);
                }
                return (*inter_source)[rid];
            };

            auto init = [](Column&) { return Int32PageBuilder{}; };

            materialize_column_parallel<Int32PageBuilder>(
                dest_col, collector, reader, init, from_build, 4);

        } else {
            const Column *str_src_ptr = col_source;
            if (!str_src_ptr && inter_source) {
                str_src_ptr = &plan.inputs[inter_source->source_table]
                                   .columns[inter_source->source_column];
            }

            auto reader = [&](uint32_t rid) {
                if (col_source) {
                    return from_build
                        ? columnar_reader.read_value_build(*col_source, remapped_col_idx, rid, DataType::VARCHAR)
                        : columnar_reader.read_value_probe(*col_source, remapped_col_idx, rid, DataType::VARCHAR);
                }
                return (*inter_source)[rid];
            };

            auto init = [str_src_ptr](Column& local) { 
                return VarcharPageBuilder(*str_src_ptr, local); 
            };

            materialize_column_parallel<VarcharPageBuilder>(
                dest_col, collector, reader, init, from_build, 35);
        }
    }
    return result;
}

/**
 *
 *  creates empty ColumnarTable with correct column types
 *  used when join produces no matches at root node
 *
 **/
inline ColumnarTable create_empty_result(
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs) {
    ColumnarTable empty_result;
    empty_result.num_rows = 0;
    for (auto [_, data_type] : remapped_attrs) {
        empty_result.columns.emplace_back(data_type);
    }
    return empty_result;
}

/**
 *
 *  dispatches materialization based on is_root flag
 *  root nodes produce ColumnarTable, others produce intermediate
 *
 **/
inline std::variant<ExecuteResult, ColumnarTable> materialize_join_results(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input, const BuildProbeConfig &config,
    const PlanNode &build_node, const PlanNode &probe_node, JoinSetup &setup,
    const Plan &plan, bool is_root) {

    if (is_root) {
        if (collector.size() == 0) {
            return create_empty_result(config.remapped_attrs);
        }
        return materialize(collector, build_input, probe_input,
                           config.remapped_attrs, build_node, probe_node,
                           build_input.output_size(), setup.columnar_reader, plan);
    } else {
        if (collector.size() == 0) return std::move(setup.results);
        
        construct_intermediate(collector, build_input, probe_input,
                               config.remapped_attrs, build_node, probe_node,
                               build_input.output_size(), setup.columnar_reader,
                               setup.results);
        return std::move(setup.results);
    }
}

} // namespace Contest

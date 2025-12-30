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
#include <string_view>
#include <algorithm>

namespace Contest {

/**
 *
 * helper to centralize mmap allocation
 * handles error checking and throws bad_alloc
 *
 **/
inline void* allocate_pages(size_t total_pages) {
    void* ptr = mmap(nullptr, total_pages * PAGE_SIZE,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::bad_alloc();
    }
    return ptr;
}

/**
 *
 * helper for building sparse page bitmaps
 * handles buffering and flushing partial bytes
 * OPTIMIZED: Branchless bit setting
 *
 **/
struct BitmapBuilder {
    std::vector<uint8_t>& bitmap;
    uint8_t pending_bits = 0;
    int bit_count = 0;

    explicit BitmapBuilder(std::vector<uint8_t>& bm) : bitmap(bm) {}

    /* branchless bit setting */
    inline void add_bit(bool is_valid) {
        pending_bits |= (static_cast<uint8_t>(is_valid) << bit_count);
        bit_count++;
        if (bit_count == 8) {
            bitmap.push_back(pending_bits);
            pending_bits = 0;
            bit_count = 0;
        }
    }

    inline void flush() {
        if (bit_count > 0) {
            bitmap.push_back(pending_bits);
            pending_bits = 0;
            bit_count = 0;
        }
    }

    inline void reset() {
        pending_bits = 0;
        bit_count = 0;
    }
};

/**
 *
 * extracts string view from ColumnarTable page
 * decodes offset array to find string boundaries
 * returns string_view wrapper around data
 * works directly on page layout without copy
 *
 **/
inline std::string_view
get_string_view(const Column &src_col, int32_t page_idx, int32_t offset_idx) {
    auto *page = reinterpret_cast<uint8_t *>(src_col.pages[page_idx]->data);
    auto num_valid = *reinterpret_cast<uint16_t *>(page + 2);
    auto *offset_array = reinterpret_cast<uint16_t *>(page + 4);
    char *char_begin = reinterpret_cast<char *>(page + 4 + num_valid * 2);

    uint16_t end_off = offset_array[offset_idx];
    uint16_t start_off = (offset_idx == 0) ? 0 : offset_array[offset_idx - 1];

    return {char_begin + start_off, static_cast<size_t>(end_off - start_off)};
}

/**
 *
 * computes chunk size for materialization parallelization
 * balances cache-optimal size with enough tasks per thread
 *
 **/
inline size_t compute_materialization_chunk_size(size_t total_matches) {
    constexpr int NUM_CORES = SPC__CORE_COUNT;

    /* calculate per-core L2 cache optimal size */
    constexpr size_t L2_CACHE_SIZE = SPC__LEVEL2_CACHE_SIZE;
    size_t per_core_l2 = L2_CACHE_SIZE / NUM_CORES;
    size_t target_bytes = per_core_l2 / 2;

    /* estimate bytes per match (conservative for VARCHAR) */
    size_t bytes_per_match = 8 + 64;  // match + avg string overhead
    size_t cache_optimal = target_bytes / bytes_per_match;

    /* ensure at least 4 chunks per thread */
    size_t min_chunk = (total_matches + NUM_CORES * 4 - 1) / (NUM_CORES * 4);

    return std::max(cache_optimal, min_chunk);
}

/**
 *
 * parallel int32 materialization
 * each thread builds local pages then merges at end
 * OPTIMIZED: Uses "Safe Batch" calculation to remove per-row page checks
 *
 **/
template <typename ReaderFunc>
inline void materialize_int32_column(Column &dest_col,
                                     const MatchCollector &collector,
                                     ReaderFunc &&read_value, bool from_build) {
    const size_t total_matches = collector.size();
    if (total_matches == 0) return;

    const uint64_t *matches_ptr = collector.matches.data();
    constexpr int num_threads = SPC__CORE_COUNT;

    size_t matches_per_thread = (total_matches + num_threads - 1) / num_threads;
    size_t max_rows_per_page = (PAGE_SIZE - 4 - 256) / 4;
    size_t pages_per_thread = (matches_per_thread + max_rows_per_page - 1) / max_rows_per_page + 1;
    size_t total_pages = pages_per_thread * num_threads;

    void* page_memory = allocate_pages(total_pages);

    std::vector<Column> thread_columns;
    thread_columns.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_columns.emplace_back(DataType::INT32);
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
        size_t chunk_matches = end - start;

        uint16_t num_rows = 0;
        std::vector<int32_t> data;
        std::vector<uint8_t> bitmap;
        
        data.reserve(chunk_matches);
        bitmap.reserve((chunk_matches + 7) / 8);

        /* 5 bytes per row conservative estimate (4 data + 1 bitmap overhead) */
        constexpr size_t BYTES_PER_ROW = 5;
        constexpr size_t HEADER_SIZE = 4; // NumRows + NumValues
        /* Leave some buffer for safety */
        constexpr size_t SAFE_PAGE_CAPACITY = PAGE_SIZE - HEADER_SIZE - 64; 

        BitmapBuilder bm_builder(bitmap);

        auto save_page = [&]() {
            bm_builder.flush();
            auto *page = local_col.new_page()->data;
            *reinterpret_cast<uint16_t *>(page) = num_rows;
            *reinterpret_cast<uint16_t *>(page + 2) = static_cast<uint16_t>(data.size());
            std::memcpy(page + 4, data.data(), data.size() * 4);
            std::memcpy(page + PAGE_SIZE - bitmap.size(), bitmap.data(), bitmap.size());
            
            num_rows = 0;
            data.clear();
            bitmap.clear();
            bm_builder.reset();
        };

        auto run_loop = [&](auto get_row_id) {
            size_t current = start;
            while (current < end) {
                /* * Calculate Safe Batch Size
                 * Determine how many rows definitely fit in the current page
                 * so we can run the inner loop without checks.
                 */
                size_t current_bytes = (data.size() * 4) + bitmap.size();
                
                if (current_bytes >= SAFE_PAGE_CAPACITY) {
                    save_page();
                    current_bytes = 0;
                }

                size_t available_bytes = SAFE_PAGE_CAPACITY - current_bytes;
                size_t max_rows_in_batch = available_bytes / BYTES_PER_ROW;
                size_t batch_size = std::min(end - current, max_rows_in_batch);
                
                /* fallback to ensure progress */
                if (batch_size == 0) batch_size = 1;

                size_t batch_end = current + batch_size;

                /* * Inner Tight Loop
                 * No page full checks here. Just read/write.
                 */
                for (size_t i = current; i < batch_end; ++i) {
                    uint32_t row_id = get_row_id(matches_ptr[i]);
                    mema::value_t val = read_value(row_id);
                    
                    bool is_valid = !val.is_null();
                    if (is_valid) {
                        data.push_back(val.value);
                    }
                    bm_builder.add_bit(is_valid);
                }
                
                num_rows += static_cast<uint16_t>(batch_size);
                current += batch_size;
            }
        };

        if (from_build) {
            run_loop([](uint64_t m) { return static_cast<uint32_t>(m & 0xFFFFFFFF); });
        } else {
            run_loop([](uint64_t m) { return static_cast<uint32_t>(m >> 32); });
        }

        if (num_rows != 0) save_page();
    });

    for (auto &thread_col : thread_columns) {
        for (auto *page : thread_col.pages) dest_col.pages.push_back(page);
        thread_col.pages.clear();
    }

    auto* mapped_mem = new MappedMemory(page_memory, total_pages * PAGE_SIZE);
    dest_col.assign_mapped_memory(mapped_mem);
}

/**
 *
 * parallel varchar column materialization
 * each thread builds its own Column with local pages
 * OPTIMIZED: Uses reserve() instead of resize() and aggressive buffer growth
 *
 **/
template <typename ReaderFunc>
inline void materialize_varchar_column(Column &dest_col,
                                       const MatchCollector &collector,
                                       ReaderFunc &&read_value,
                                       const Column &src_col, bool from_build) {
    const size_t total_matches = collector.size();
    if (total_matches == 0) return;

    const uint64_t *matches_ptr = collector.matches.data();
    constexpr int num_threads = SPC__CORE_COUNT;

    size_t matches_per_thread = (total_matches + num_threads - 1) / num_threads;
    constexpr size_t AVG_STRING_LEN = 32;
    size_t rows_per_page = (PAGE_SIZE - 256) / (AVG_STRING_LEN + 3);
    size_t pages_per_thread = (matches_per_thread + rows_per_page - 1) / rows_per_page + 2;
    size_t total_pages = pages_per_thread * num_threads;

    void* page_memory = allocate_pages(total_pages);

    std::vector<Column> thread_columns;
    thread_columns.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_columns.emplace_back(DataType::VARCHAR);
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
        size_t chunk_matches = end - start;

        uint16_t num_rows = 0;
        std::vector<char> char_data;
        std::vector<uint16_t> offsets;
        std::vector<uint8_t> bitmap;
        size_t current_char_size = 0;

        char_data.reserve(chunk_matches * 32);
        offsets.reserve(chunk_matches);
        bitmap.reserve((chunk_matches + 7) / 8);

        /* use block checking to reduce overflow overhead */
        constexpr size_t CHECK_INTERVAL = 32; 

        BitmapBuilder bm_builder(bitmap);

        auto save_long_string_buffer = [&](std::string_view str) {
            size_t offset = 0;
            bool first_page = true;
            while (offset < str.size()) {
                auto *page = local_col.new_page()->data;
                *reinterpret_cast<uint16_t *>(page) = first_page ? 0xffff : 0xfffe;
                first_page = false;
                size_t len = std::min(str.size() - offset, static_cast<size_t>(PAGE_SIZE - 4));
                *reinterpret_cast<uint16_t *>(page + 2) = static_cast<uint16_t>(len);
                std::memcpy(page + 4, str.data() + offset, len);
                offset += len;
            }
        };

        auto copy_long_string_pages = [&](int32_t start_page_idx) {
            int32_t curr_idx = start_page_idx;
            while (true) {
                auto *src_page_data = src_col.pages[curr_idx]->data;
                auto *dest_page_data = local_col.new_page()->data;
                std::memcpy(dest_page_data, src_page_data, PAGE_SIZE);
                curr_idx++;
                if (curr_idx >= static_cast<int32_t>(src_col.pages.size())) break;
                auto *next_p = src_col.pages[curr_idx]->data;
                if (*reinterpret_cast<uint16_t *>(next_p) != 0xfffe) break;
            }
        };

        auto save_page = [&]() {
            bm_builder.flush();
            auto *page = local_col.new_page()->data;
            *reinterpret_cast<uint16_t *>(page) = num_rows;
            *reinterpret_cast<uint16_t *>(page + 2) = static_cast<uint16_t>(offsets.size());
            std::memcpy(page + 4, offsets.data(), offsets.size() * 2);
            std::memcpy(page + 4 + offsets.size() * 2, char_data.data(), current_char_size);
            std::memcpy(page + PAGE_SIZE - bitmap.size(), bitmap.data(), bitmap.size());
            
            num_rows = 0;
            current_char_size = 0;
            offsets.clear();
            bitmap.clear();
            /* Keep char_data capacity, just reset usage */
            bm_builder.reset();
        };

        auto add_normal_string = [&](int32_t page_idx, int32_t offset_idx) -> bool {
            auto sv = get_string_view(src_col, page_idx, offset_idx);

            if (sv.length() > PAGE_SIZE - 7) {
                if (num_rows > 0) save_page();
                save_long_string_buffer(sv);
                return false;
            }

            /* conservative check */
            size_t needed = 4 + (offsets.size() + 1) * 2 +
                            (current_char_size + sv.length()) +
                            (bitmap.size() + 2);
            if (needed > PAGE_SIZE) save_page();

            /* aggressive growth strategy: avoid realloc */
            if (current_char_size + sv.length() > char_data.capacity()) {
                size_t new_cap = std::max(char_data.capacity() * 2, 
                                          current_char_size + sv.length() + 8192);
                char_data.reserve(new_cap);
            }

            /* write directly to buffer without zero-init */
            std::memcpy(char_data.data() + current_char_size, sv.data(), sv.length());
            current_char_size += sv.length();
            offsets.push_back(current_char_size); 
            return true;
        };

        auto run_loop = [&](auto get_row_id) {
            size_t rows_since_check = 0;
            for (size_t i = start; i < end; ++i) {
                uint32_t row_id = get_row_id(matches_ptr[i]);
                mema::value_t val = read_value(row_id);

                if (val.is_null()) {
                    /* check page overflow periodically */
                    if (++rows_since_check >= CHECK_INTERVAL) {
                        if (4 + offsets.size() * 2 + current_char_size +
                                (bitmap.size() + 2) > PAGE_SIZE) {
                            save_page();
                        }
                        rows_since_check = 0;
                    }
                    bm_builder.add_bit(false);
                    num_rows++;
                } else {
                    int32_t page_idx, offset_idx;
                    mema::value_t::decode_string(val.value, page_idx, offset_idx);

                    if (offset_idx == mema::value_t::LONG_STRING_OFFSET) {
                        if (num_rows > 0) save_page();
                        copy_long_string_pages(page_idx);
                        rows_since_check = 0;
                    } else {
                        if (add_normal_string(page_idx, offset_idx)) {
                            bm_builder.add_bit(true);
                            num_rows++;
                            rows_since_check++; /* incremented inside add_normal too implicitly by size check */
                        }
                    }
                }
            }
        };

        if (from_build) {
            run_loop([](uint64_t m) { return static_cast<uint32_t>(m & 0xFFFFFFFF); });
        } else {
            run_loop([](uint64_t m) { return static_cast<uint32_t>(m >> 32); });
        }

        if (num_rows != 0) save_page();
    });

    for (auto &thread_col : thread_columns) {
        for (auto *page : thread_col.pages) dest_col.pages.push_back(page);
        thread_col.pages.clear();
    }

    auto* mapped_mem = new MappedMemory(page_memory, total_pages * PAGE_SIZE);
    dest_col.assign_mapped_memory(mapped_mem);
}

/**
 *
 * helper to determine source column pointer
 * handles both columnar and intermediate inputs
 *
 **/
inline std::pair<const Column*, const mema::column_t*>
resolve_column_source(const JoinInput& input, const PlanNode& node, size_t col_idx) {
    if (input.is_columnar()) {
        auto* table = std::get<const ColumnarTable*>(input.data);
        auto [actual_idx, _] = node.output_attrs[col_idx];
        return { &table->columns[actual_idx], nullptr };
    } else {
        const auto& result = std::get<ExecuteResult>(input.data);
        return { nullptr, &result[col_idx] };
    }
}

/**
 *
 * creates empty ColumnarTable with correct column types
 * used when join produces no matches at root node
 *
 **/
inline ColumnarTable create_empty_result(
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs) {
    ColumnarTable empty_result;
    empty_result.num_rows = 0;
    for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
        auto [_, data_type] = remapped_attrs[out_idx];
        empty_result.columns.emplace_back(data_type);
    }
    return empty_result;
}

/**
 *
 * materializes final ColumnarTable from mixed inputs
 * produces result from columnar + column_t intermediate combinations
 *
 **/
inline ColumnarTable materialize(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, const Plan &plan) {

    if (collector.size() == 0) {
        return create_empty_result(remapped_attrs);
    }

    ColumnarTable result;
    result.num_rows = collector.size();

    for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
        auto [col_idx, data_type] = remapped_attrs[out_idx];
        bool from_build = col_idx < build_size;
        size_t remapped_col_idx = from_build ? col_idx : col_idx - build_size;

        result.columns.emplace_back(data_type);
        Column &dest_col = result.columns.back();

        /* use helper to resolve source without duplicating logic */
        auto [columnar_src_col, intermediate_src_col] = from_build
            ? resolve_column_source(build_input, build_node, remapped_col_idx)
            : resolve_column_source(probe_input, probe_node, remapped_col_idx);

        if (data_type == DataType::INT32) {
            if (columnar_src_col) {
                auto &col = *columnar_src_col;
                if (from_build) {
                    materialize_int32_column(dest_col, collector,
                        [&](uint32_t rid) { return columnar_reader.read_value_build(col, remapped_col_idx, rid, DataType::INT32); },
                        from_build);
                } else {
                    materialize_int32_column(dest_col, collector,
                        [&](uint32_t rid) { return columnar_reader.read_value_probe(col, remapped_col_idx, rid, DataType::INT32); },
                        from_build);
                }
            } else {
                auto &col = *intermediate_src_col;
                materialize_int32_column(dest_col, collector,
                    [&](uint32_t rid) { return col[rid]; }, from_build);
            }
        } else {
            const Column *string_source_col = columnar_src_col;
            if (!string_source_col && intermediate_src_col) {
                const auto &src_table = plan.inputs[intermediate_src_col->source_table];
                string_source_col = &src_table.columns[intermediate_src_col->source_column];
            }

            if (columnar_src_col) {
                auto &col = *columnar_src_col;
                if (from_build) {
                    materialize_varchar_column(dest_col, collector,
                        [&](uint32_t rid) { return columnar_reader.read_value_build(col, remapped_col_idx, rid, DataType::VARCHAR); },
                        *string_source_col, from_build);
                } else {
                    materialize_varchar_column(dest_col, collector,
                        [&](uint32_t rid) { return columnar_reader.read_value_probe(col, remapped_col_idx, rid, DataType::VARCHAR); },
                        *string_source_col, from_build);
                }
            } else {
                auto &col = *intermediate_src_col;
                materialize_varchar_column(dest_col, collector,
                    [&](uint32_t rid) { return col[rid]; }, *string_source_col, from_build);
            }
        }
    }
    return result;
}

/**
 *
 * dispatches materialization based on is_root flag and input types
 * root nodes produce ColumnarTable via materialize_* functions
 * intermediate nodes produce column_t via construct_intermediate_* functions
 * handles empty collector case for both root and intermediate
 *
 **/
inline std::variant<ExecuteResult, ColumnarTable> materialize_join_results(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input, const BuildProbeConfig &config,
    const PlanNode &build_node, const PlanNode &probe_node, JoinSetup &setup,
    const Plan &plan, bool is_root) {

    if (collector.size() == 0) {
        if (is_root) {
            return create_empty_result(config.remapped_attrs);
        }
        return std::move(setup.results);
    }

    if (is_root) {
        return std::move(materialize(
            collector, build_input, probe_input, config.remapped_attrs,
            build_node, probe_node, build_input.output_size(),
            setup.columnar_reader, plan));
    } else {
        construct_intermediate(collector, build_input, probe_input,
                               config.remapped_attrs, build_node, probe_node,
                               build_input.output_size(), setup.columnar_reader,
                               setup.results);

        return std::move(setup.results);
    }
}

} // namespace Contest

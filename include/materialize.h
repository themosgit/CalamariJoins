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

namespace Contest {

/**
 *
 *  extracts string view from ColumnarTable page
 *  decodes offset array to find string boundaries
 *  returns pointer to string data and length
 *  works directly on page layout without copy
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
 *  balances cache-optimal size with enough tasks per thread
 *
 **/
inline size_t compute_materialization_chunk_size(size_t total_matches) {
    constexpr int NUM_CORES = SPC__CORE_COUNT;
    constexpr size_t L2_CACHE_SIZE = SPC__LEVEL2_CACHE_SIZE;
    size_t per_core_l2 = L2_CACHE_SIZE / NUM_CORES;
    size_t target_bytes = per_core_l2 / 2;
    size_t bytes_per_match = 8 + 64;
    size_t cache_optimal = target_bytes / bytes_per_match;
    size_t min_chunk = (total_matches + NUM_CORES * 4 - 1) / (NUM_CORES * 4);
    return std::max(cache_optimal, min_chunk);
}
/**
 *
 *  parallel int32 materialization
 *  each thread builds local pages then merges at end
 *
 **/
template <typename ReaderFunc>
inline void materialize_int32_column(Column &dest_col,
                                     const MatchCollector &collector,
                                     ReaderFunc &&read_value, bool from_build) {
    const size_t total_matches = collector.size();
    if (total_matches == 0) return;

    const uint64_t *matches_ptr = collector.matches.data();
    const size_t chunk_size = compute_materialization_chunk_size(total_matches);
    constexpr int num_threads = SPC__CORE_COUNT;

    /* estimate pages per thread conservatively */
    size_t matches_per_thread = (total_matches + num_threads - 1) / num_threads;
    size_t max_rows_per_page = (PAGE_SIZE - 4 - 256) / 4;  // header + bitmap + data
    size_t pages_per_thread = (matches_per_thread + max_rows_per_page - 1) / max_rows_per_page + 1;
    size_t total_pages = pages_per_thread * num_threads;

    /* batch allocate all pages upfront */
    void* page_memory = mmap(nullptr, total_pages * PAGE_SIZE,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page_memory == MAP_FAILED) {
        throw std::bad_alloc();
    }

    std::vector<Column> thread_columns;
    thread_columns.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_columns.emplace_back(DataType::INT32);
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
        if (start >= total_matches)
            return;

        Column &local_col = thread_columns[t];
        size_t chunk_matches = end - start;

        /* pre-size vectors to exact capacity */
        uint16_t num_rows = 0;
        std::vector<int32_t> data;
        std::vector<uint8_t> bitmap;
        data.reserve(chunk_matches);
        bitmap.reserve((chunk_matches + 7) / 8);

        /* calculate conservative min rows per page to reduce overflow checks */
        constexpr size_t MIN_ROWS_PER_PAGE = (PAGE_SIZE - 4 - 256) / (4 + 1);

        uint8_t pending_bits = 0;
        int bit_count = 0;

        auto flush_bitmap = [&]() {
            if (bit_count > 0) {
                bitmap.push_back(pending_bits);
                pending_bits = 0;
                bit_count = 0;
            }
        };

        auto save_page = [&]() {
            flush_bitmap();
            auto *page = local_col.new_page()->data;
            *reinterpret_cast<uint16_t *>(page) = num_rows;
            *reinterpret_cast<uint16_t *>(page + 2) =
                static_cast<uint16_t>(data.size());
            std::memcpy(page + 4, data.data(), data.size() * 4);
            std::memcpy(page + PAGE_SIZE - bitmap.size(), bitmap.data(),
                        bitmap.size());
            num_rows = 0;
            data.clear();
            bitmap.clear();
            pending_bits = 0;
            bit_count = 0;
        };

        auto run_loop = [&](auto get_row_id) {
            size_t rows_since_check = 0;
            for (size_t i = start; i < end; ++i) {
                uint32_t row_id = get_row_id(matches_ptr[i]);
                mema::value_t val = read_value(row_id);

                /* check overflow only every MIN_ROWS_PER_PAGE rows */
                if (rows_since_check >= MIN_ROWS_PER_PAGE) {
                    if (4 + (data.size() + 1) * 4 + (bitmap.size() + 2) > PAGE_SIZE) {
                        save_page();
                    }
                    rows_since_check = 0;
                }

                if (!val.is_null()) {
                    pending_bits |= (1u << bit_count);
                    data.push_back(val.value);
                }
                bit_count++;
                if (bit_count == 8) {
                    bitmap.push_back(pending_bits);
                    pending_bits = 0;
                    bit_count = 0;
                }
                ++num_rows;
                ++rows_since_check;
            }
        };

        if (from_build) {
            run_loop(
                [](uint64_t m) { return static_cast<uint32_t>(m & 0xFFFFFFFF); });
        } else {
            run_loop([](uint64_t m) { return static_cast<uint32_t>(m >> 32); });
        }

        if (num_rows != 0)
            save_page();
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
 *  parallel varchar column materialization
 *  each thread builds its own Column with local pages
 *  merge pages at end via pointer moves
 *
 **/
template <typename ReaderFunc>
inline void materialize_varchar_column(Column &dest_col,
                                       const MatchCollector &collector,
                                       ReaderFunc &&read_value,
                                       const Column &src_col, bool from_build) {
    const size_t total_matches = collector.size();
    if (total_matches == 0)
        return;

    const uint64_t *matches_ptr = collector.matches.data();
    const size_t chunk_size = compute_materialization_chunk_size(total_matches);
    constexpr int num_threads = SPC__CORE_COUNT;

    /* estimate pages per thread conservatively for VARCHAR */
    size_t matches_per_thread = (total_matches + num_threads - 1) / num_threads;
    constexpr size_t AVG_STRING_LEN = 32;
    size_t bytes_per_row = AVG_STRING_LEN + 2 + 1;
    size_t usable_per_page = PAGE_SIZE - 256;
    size_t rows_per_page = usable_per_page / bytes_per_row;
    size_t pages_per_thread = (matches_per_thread + rows_per_page - 1) / rows_per_page + 2;
    size_t total_pages = pages_per_thread * num_threads;

    /* batch allocate all pages upfront */
    void* page_memory = mmap(nullptr, total_pages * PAGE_SIZE,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page_memory == MAP_FAILED) {
        throw std::bad_alloc();
    }

    std::vector<Column> thread_columns;
    thread_columns.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_columns.emplace_back(DataType::VARCHAR);
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
        if (start >= total_matches)
            return;

        Column &local_col = thread_columns[t];
        size_t chunk_matches = end - start;

        uint16_t num_rows = 0;
        std::vector<char> char_data;
        std::vector<uint16_t> offsets;
        std::vector<uint8_t> bitmap;
        size_t current_char_size = 0;

        /* pre-size vectors with better estimates */
        constexpr size_t AVG_STRING_SIZE = 32;
        char_data.reserve(chunk_matches * AVG_STRING_SIZE);
        offsets.reserve(chunk_matches);
        bitmap.reserve((chunk_matches + 7) / 8);

        /* calculate conservative min rows per page for VARCHAR */
        constexpr size_t MIN_ROWS_PER_PAGE = 100;

        uint8_t pending_bits = 0;
        int bit_count = 0;

        auto flush_bitmap = [&]() {
            if (bit_count > 0) {
                bitmap.push_back(pending_bits);
                pending_bits = 0;
                bit_count = 0;
            }
        };

        auto save_long_string_buffer = [&](const std::string &str) {
            size_t offset = 0;
            bool first_page = true;
            while (offset < str.size()) {
                auto *page = local_col.new_page()->data;
                *reinterpret_cast<uint16_t *>(page) =
                    first_page ? 0xffff : 0xfffe;
                first_page = false;
                size_t len = std::min(str.size() - offset, PAGE_SIZE - 4);
                *reinterpret_cast<uint16_t *>(page + 2) =
                    static_cast<uint16_t>(len);
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
                if (curr_idx >= static_cast<int32_t>(src_col.pages.size()))
                    break;
                auto *next_p = src_col.pages[curr_idx]->data;
                if (*reinterpret_cast<uint16_t *>(next_p) != 0xfffe)
                    break;
            }
        };

        size_t rows_since_check = 0;

        auto save_page = [&]() {
            flush_bitmap();
            auto *page = local_col.new_page()->data;
            *reinterpret_cast<uint16_t *>(page) = num_rows;
            *reinterpret_cast<uint16_t *>(page + 2) =
                static_cast<uint16_t>(offsets.size());
            std::memcpy(page + 4, offsets.data(), offsets.size() * 2);
            std::memcpy(page + 4 + offsets.size() * 2, char_data.data(),
                        current_char_size);
            std::memcpy(page + PAGE_SIZE - bitmap.size(), bitmap.data(),
                        bitmap.size());
            num_rows = 0;
            current_char_size = 0;
            offsets.clear();
            bitmap.clear();
            pending_bits = 0;
            bit_count = 0;
            rows_since_check = 0;
        };

        auto add_normal_string =
            [&](int32_t page_idx, int32_t offset_idx) -> bool {
            auto [str_ptr, str_len] =
                get_string_view(src_col, page_idx, offset_idx);

            if (str_len > PAGE_SIZE - 7) {
                if (num_rows > 0)
                    save_page();
                save_long_string_buffer(std::string(str_ptr, str_len));
                return false;
            }

            size_t needed = 4 + (offsets.size() + 1) * 2 +
                            (current_char_size + str_len) +
                            (bitmap.size() + 2);
            if (needed > PAGE_SIZE)
                save_page();

            if (current_char_size + str_len > char_data.size()) {
                char_data.resize(std::max(char_data.size() * 2,
                                          current_char_size + str_len + 4096));
            }

            std::memcpy(char_data.data() + current_char_size, str_ptr, str_len);
            current_char_size += str_len;
            offsets.push_back(current_char_size);
            return true;
        };

        auto run_loop = [&](auto get_row_id) {
            for (size_t i = start; i < end; ++i) {
                uint32_t row_id = get_row_id(matches_ptr[i]);
                mema::value_t val = read_value(row_id);

                if (val.is_null()) {
                    /* reduce overflow checks for nulls */
                    if (rows_since_check >= MIN_ROWS_PER_PAGE) {
                        if (4 + offsets.size() * 2 + current_char_size +
                                (bitmap.size() + 2) > PAGE_SIZE) {
                            save_page();
                        }
                        rows_since_check = 0;
                    }
                    bit_count++;
                    if (bit_count == 8) {
                        bitmap.push_back(pending_bits);
                        pending_bits = 0;
                        bit_count = 0;
                    }
                    num_rows++;
                    rows_since_check++;
                } else {
                    int32_t page_idx, offset_idx;
                    mema::value_t::decode_string(val.value, page_idx,
                                                  offset_idx);

                    if (offset_idx == mema::value_t::LONG_STRING_OFFSET) {
                        if (num_rows > 0)
                            save_page();
                        copy_long_string_pages(page_idx);
                        rows_since_check = 0;
                    } else {
                        if (add_normal_string(page_idx, offset_idx)) {
                            pending_bits |= (1u << bit_count);
                            bit_count++;
                            if (bit_count == 8) {
                                bitmap.push_back(pending_bits);
                                pending_bits = 0;
                                bit_count = 0;
                            }
                            num_rows++;
                            rows_since_check++;
                        }
                    }
                }
            }
        };

        if (from_build) {
            run_loop(
                [](uint64_t m) { return static_cast<uint32_t>(m & 0xFFFFFFFF); });
        } else {
            run_loop(
                [](uint64_t m) { return static_cast<uint32_t>(m >> 32); });
        }

        if (num_rows != 0)
            save_page();
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
 *  produces result from columnar + column_t intermediate combinations
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
        for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
            auto [_, data_type] = remapped_attrs[out_idx];
            result.columns.emplace_back(data_type);
        }
        return result;
    }

    auto read_from_intermediate = [](const mema::column_t &column,
                                     uint32_t row_id) {
        return column[row_id];
    };

    for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
        auto [col_idx, data_type] = remapped_attrs[out_idx];
        bool from_build = col_idx < build_size;
        size_t remapped_col_idx = from_build ? col_idx : col_idx - build_size;

        result.columns.emplace_back(data_type);
        Column &dest_col = result.columns.back();

        const Column *columnar_src_col = nullptr;
        const mema::column_t *intermediate_src_col = nullptr;

        if (from_build) {
            if (build_input.is_columnar()) {
                auto *build_table =
                    std::get<const ColumnarTable *>(build_input.data);
                auto [actual_col_idx, _] =
                    build_node.output_attrs[remapped_col_idx];
                columnar_src_col = &build_table->columns[actual_col_idx];
            } else {
                const auto &build_result =
                    std::get<ExecuteResult>(build_input.data);
                intermediate_src_col = &build_result[remapped_col_idx];
            }
        } else {
            if (probe_input.is_columnar()) {
                auto *probe_table =
                    std::get<const ColumnarTable *>(probe_input.data);
                auto [actual_col_idx, _] =
                    probe_node.output_attrs[remapped_col_idx];
                columnar_src_col = &probe_table->columns[actual_col_idx];
            } else {
                const auto &probe_result =
                    std::get<ExecuteResult>(probe_input.data);
                intermediate_src_col = &probe_result[remapped_col_idx];
            }
        }

        if (data_type == DataType::INT32) {
            if (columnar_src_col) {
                auto &col = *columnar_src_col;
                if (from_build) {
                    materialize_int32_column(
                        dest_col, collector,
                        [&](uint32_t rid) {
                            return columnar_reader.read_value_build(
                                col, remapped_col_idx, rid, DataType::INT32);
                        },
                        from_build);
                } else {
                    materialize_int32_column(
                        dest_col, collector,
                        [&](uint32_t rid) {
                            return columnar_reader.read_value_probe(
                                col, remapped_col_idx, rid, DataType::INT32);
                        },
                        from_build);
                }
            } else {
                auto &col = *intermediate_src_col;
                materialize_int32_column(
                    dest_col, collector,
                    [&](uint32_t rid) {
                        return read_from_intermediate(col, rid);
                    },
                    from_build);
            }
        } else {
            const Column *string_source_col = columnar_src_col;
            if (!string_source_col && intermediate_src_col) {
                const auto &src_table =
                    plan.inputs[intermediate_src_col->source_table];
                string_source_col =
                    &src_table.columns[intermediate_src_col->source_column];
            }

            if (columnar_src_col) {
                auto &col = *columnar_src_col;
                if (from_build) {
                    materialize_varchar_column(
                        dest_col, collector,
                        [&](uint32_t rid) {
                            return columnar_reader.read_value_build(
                                col, remapped_col_idx, rid, DataType::VARCHAR);
                        },
                        *string_source_col, from_build);
                } else {
                    materialize_varchar_column(
                        dest_col, collector,
                        [&](uint32_t rid) {
                            return columnar_reader.read_value_probe(
                                col, remapped_col_idx, rid, DataType::VARCHAR);
                        },
                        *string_source_col, from_build);
                }
            } else {
                auto &col = *intermediate_src_col;
                materialize_varchar_column(
                    dest_col, collector,
                    [&](uint32_t rid) {
                        return read_from_intermediate(col, rid);
                    },
                    *string_source_col, from_build);
            }
        }
    }
    return result;
}

inline void ensure_bitmap_size(std::vector<uint8_t> &bitmap, size_t byte_idx) {
    if (bitmap.size() <= byte_idx) {
        bitmap.resize(byte_idx + 1, 0);
    }
}
inline void set_bitmap_bit(std::vector<uint8_t> &bitmap, size_t byte_idx,
                           uint8_t bit_idx) {
    ensure_bitmap_size(bitmap, byte_idx);
    bitmap[byte_idx] |= (1u << bit_idx);
}

/**
 *
 *  creates empty ColumnarTable with correct column types
 *  used when join produces no matches at root node
 *  initializes columns without any pages
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
 *  dispatches materialization based on is_root flag and input types
 *  root nodes produce ColumnarTable via materialize_* functions
 *  intermediate nodes produce column_t via construct_intermediate_* functions
 *  handles empty collector case for both root and intermediate
 *
 **/
inline std::variant<ExecuteResult, ColumnarTable> materialize_join_results(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input, const BuildProbeConfig &config,
    const PlanNode &build_node, const PlanNode &probe_node, JoinSetup &setup,
    const Plan &plan, bool is_root) {

    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();

    if (is_root) {
        if (collector.size() == 0) {
            return create_empty_result(config.remapped_attrs);
        }
            return materialize(
                collector, build_input, probe_input, config.remapped_attrs,
                build_node, probe_node, build_input.output_size(),
                setup.columnar_reader, plan);
        } else {
        if (collector.size() == 0) {
            return std::move(setup.results);
        }

        construct_intermediate(collector, build_input, probe_input,
                              config.remapped_attrs, build_node, probe_node,
                              build_input.output_size(), setup.columnar_reader,
                              setup.results);
        return std::move(setup.results);
    }
}

} // namespace Contest

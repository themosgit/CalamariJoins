/**
 *
 * @file nested_loop.h
 * @brief Nested loop join for small build tables (<8 rows).
 *
 * Fallback when build fits vector register. 
 * Parallel work-stealing probe.
 *
 * Outperforms hash join for tiny tables (duh).
 *
 * Templated on MatchCollectionMode for zero-overhead mode selection.
 *
 * @see execute.cpp HASH_TABLE_THRESHOLD = 8
 *
 **/
#pragma once

#include <atomic>
#include <cstdint>
#include <data_model/intermediate.h>
#include <data_model/plan.h>
#include <join_execution/join_setup.h>
#include <join_execution/match_collector.h>
#include <join_execution/simd_compare.h>
#include <platform/arena_vector.h>
#include <platform/worker_pool.h>
#include <vector>

/**
 *
 * @namespace Contest::join
 * @brief visit_rows() iterator, nested_loop_join() for tiny build tables.
 *
 **/
namespace Contest::join {

using Contest::ExecuteResult;
using Contest::platform::THREAD_COUNT;
using Contest::platform::worker_pool;

/**
 *
 * @brief Iterates over non-NULL values in a join input column.
 *
 * Abstracts columnar vs intermediate input. Handles NULL bitmaps.
 *
 * @tparam Func void(uint32_t row_id, int32_t value).
 *
 **/
template <typename Func>
inline void visit_rows(const JoinInput &input, size_t attr_idx,
                       Func &&visitor) {
    if (input.is_columnar()) {
        auto *table = std::get<const ColumnarTable *>(input.data);
        auto [col_idx, _] = input.node->output_attrs[attr_idx];
        const Column &col = table->columns[col_idx];

        uint32_t row_id = 0;
        for (auto *page_obj : col.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<uint16_t *>(page);
            auto num_values = *reinterpret_cast<uint16_t *>(page + 2);
            auto *data = reinterpret_cast<const int32_t *>(page + 4);

            uint16_t val_idx = 0;
            for (uint16_t i = 0; i < num_rows; i++) {
                if (num_rows == num_values) {
                    visitor(row_id++, data[i]);
                } else {
                    auto *bitmap = reinterpret_cast<const uint8_t *>(
                        page + PAGE_SIZE - (num_rows + 7) / 8);
                    if (bitmap[i / 8] & (1u << (i % 8))) {
                        visitor(row_id, data[val_idx++]);
                    }
                    row_id++;
                }
            }
        }
    } else {
        const auto &res = std::get<ExecuteResult>(input.data);
        const mema::column_t &col = res[attr_idx];
        size_t count = col.row_count();
        for (size_t i = 0; i < count; i++) {
            const mema::value_t &val = col[i];
            if (!val.is_null()) {
                visitor(static_cast<uint32_t>(i), val.value);
            }
        }
    }
}

/**
 *
 * @brief Nested loop join for small build tables (<=8 rows).
 *
 * Build keys/IDs in SIMD registers (AVX2/NEON).
 * Parallel probe via work-stealing.
 *
 * if i had a machine with AVX512 i would support it :)
 *
 *
 * @tparam Mode Collection mode (BOTH, LEFT_ONLY, RIGHT_ONLY) for compile-time
 *              specialization of match buffer operations.
 * @return Thread-local match buffers for direct iteration.
 *
 **/
template <MatchCollectionMode Mode>
inline std::vector<ThreadLocalMatchBuffer<Mode>>
nested_loop_join(const JoinInput &build_input, const JoinInput &probe_input,
                 size_t build_attr, size_t probe_attr) {
    size_t build_rows = build_input.row_count(build_attr);
    size_t probe_rows = probe_input.row_count(probe_attr);

    if (build_rows == 0 || probe_rows == 0)
        return {};

    size_t num_threads = THREAD_COUNT;
    std::vector<ThreadLocalMatchBuffer<Mode>> buffers(num_threads);

    constexpr size_t MAX_BUILD_SIZE = 8;
    alignas(32) int32_t b_vals[MAX_BUILD_SIZE];
    alignas(16) uint32_t b_ids[MAX_BUILD_SIZE];
    size_t b_count = 0;

    auto collect_build = [&](uint32_t id, int32_t val) {
        if (b_count < MAX_BUILD_SIZE) {
            b_ids[b_count] = id;
            b_vals[b_count] = val;
            b_count++;
        }
    };

    visit_rows(build_input, build_attr, collect_build);

    for (size_t i = b_count; i < MAX_BUILD_SIZE; ++i) {
        b_vals[i] = INT32_MIN;
    }

    const Column *probe_col = nullptr;
    platform::ArenaVector<uint32_t> page_offsets(
        Contest::platform::get_arena(0));
    if (probe_input.is_columnar()) {
        auto *table = std::get<const ColumnarTable *>(probe_input.data);
        auto [col_idx, _] = probe_input.node->output_attrs[probe_attr];
        probe_col = &table->columns[col_idx];

        page_offsets.reserve(probe_col->pages.size() + 1);
        uint32_t current = 0;
        for (auto *p : probe_col->pages) {
            page_offsets.push_back(current);
            current += *reinterpret_cast<const uint16_t *>(p->data);
        }
        page_offsets.push_back(current);
    }
    std::atomic<size_t> probe_page_counter{0};

    worker_pool().execute([&](size_t t_id) {
        buffers[t_id] =
            ThreadLocalMatchBuffer<Mode>(Contest::platform::get_arena(t_id));
        auto &local_buffer = buffers[t_id];

        auto process_value = [&](uint32_t p_id, int32_t p_val) {
            simd::eq_scan_build<Mode>(p_id, p_val, b_vals, b_ids, b_count,
                                      local_buffer);
        };

        if (probe_input.is_columnar()) {
            size_t num_pages = probe_col->pages.size();

            while (true) {
                size_t i =
                    probe_page_counter.fetch_add(1, std::memory_order_relaxed);

                if (i >= num_pages)
                    break;
                auto *page = probe_col->pages[i]->data;
                auto num_rows = *reinterpret_cast<const uint16_t *>(page);
                auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
                auto *data = reinterpret_cast<const int32_t *>(page + 4);
                uint32_t row_id = page_offsets[i];

                if (num_rows == num_values) {
                    uint16_t j = simd::eq_batch_columnar<Mode>(
                        data, num_rows, row_id, b_vals, b_ids, b_count,
                        local_buffer);
                    row_id += j;
                    for (; j < num_rows; j++) {
                        process_value(row_id++, data[j]);
                    }
                } else {
                    auto *bitmap = reinterpret_cast<const uint8_t *>(
                        page + PAGE_SIZE - (num_rows + 7) / 8);
                    uint16_t val_idx = 0;
                    for (uint16_t j = 0; j < num_rows; j++) {
                        if (bitmap[j / 8] & (1u << (j % 8))) {
                            process_value(row_id, data[val_idx++]);
                        }
                        row_id++;
                    }
                }
            }
        } else {
            const auto &res = std::get<ExecuteResult>(probe_input.data);
            const mema::column_t &col = res[probe_attr];
            size_t count = col.row_count();
            size_t start = (t_id * count) / THREAD_COUNT;
            size_t end = ((t_id + 1) * count) / THREAD_COUNT;

            constexpr size_t BATCH_SIZE = simd::INTERMEDIATE_BATCH_SIZE;
            size_t i = start;

            if constexpr (BATCH_SIZE > 0) {
                for (; i + BATCH_SIZE <= end; i += BATCH_SIZE) {
                    size_t page_idx = i >> 12;
                    size_t offset = i & 0xFFF;

                    if (offset + BATCH_SIZE <= mema::CAP_PER_PAGE) {
                        const int32_t *vals = reinterpret_cast<const int32_t *>(
                            &col.pages[page_idx]->data[offset]);
                        simd::eq_batch_intermediate<Mode>(
                            vals, i, b_vals, b_ids, b_count, local_buffer);
                    } else {
                        for (size_t j = i; j < i + BATCH_SIZE; j++) {
                            const mema::value_t &val = col[j];
                            if (!val.is_null()) {
                                process_value(static_cast<uint32_t>(j),
                                              val.value);
                            }
                        }
                    }
                }
            }

            for (; i < end; i++) {
                const mema::value_t &val = col[i];
                if (!val.is_null()) {
                    process_value(static_cast<uint32_t>(i), val.value);
                }
            }
        }
    });

    return buffers;
}

} // namespace Contest::join

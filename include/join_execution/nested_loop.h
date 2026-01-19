/**
 * @file nested_loop.h
 * @brief Nested loop join for small build tables (<8 rows).
 *
 * Fallback when build fits in L1 cache. Parallel work-stealing probe.
 * Outperforms hash join for tiny tables due to cache locality.
 *
 * @see execute.cpp HASH_TABLE_THRESHOLD = 8
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <data_model/intermediate.h>
#include <data_model/plan.h>
#include <join_execution/join_setup.h>
#include <join_execution/match_collector.h>
#include <materialization/construct_intermediate.h>
#include <platform/worker_pool.h>
#include <vector>

/**
 * @namespace Contest::join
 * @brief visit_rows() iterator, nested_loop_join() for tiny build tables.
 */
namespace Contest::join {

using Contest::ExecuteResult;
using Contest::platform::worker_pool;

/**
 * @brief Iterates over non-NULL values in a join input column.
 *
 * Abstracts columnar vs intermediate input. Handles NULL bitmaps.
 *
 * @tparam Func void(uint32_t row_id, int32_t value).
 */
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
 * @brief Nested loop join for small build tables (<=64 rows).
 *
 * Build keys/IDs in stack arrays (512 bytes, L1-resident). Parallel probe
 * via work-stealing. Beats hash join for <8 rows due to no hash overhead.
 *
 * @param mode BOTH, LEFT_ONLY, or RIGHT_ONLY.
 */
inline void
nested_loop_join(const JoinInput &build_input, const JoinInput &probe_input,
                 size_t build_attr, size_t probe_attr,
                 MatchCollector &collector,
                 MatchCollectionMode mode = MatchCollectionMode::BOTH) {
    size_t build_rows = build_input.row_count(build_attr);
    size_t probe_rows = probe_input.row_count(probe_attr);

    if (build_rows == 0 || probe_rows == 0)
        return;

    /**
     * MAX_BUILD_SIZE = 64 fits in L1 (512 bytes).
     */
    constexpr size_t MAX_BUILD_SIZE = 64;
    int32_t b_vals[MAX_BUILD_SIZE];
    uint32_t b_ids[MAX_BUILD_SIZE];
    size_t b_count = 0;

    auto collect_build = [&](uint32_t id, int32_t val) {
        if (b_count < MAX_BUILD_SIZE) {
            b_ids[b_count] = id;
            b_vals[b_count] = val;
            b_count++;
        }
    };

    visit_rows(build_input, build_attr, collect_build);

    size_t num_threads = worker_pool.thread_count();
    auto buffers = create_thread_local_buffers(num_threads, mode);

    const Column *probe_col = nullptr;
    std::vector<uint32_t> page_offsets;
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

    worker_pool.execute([&](size_t t_id) {
        auto &local_buffer = buffers[t_id];

        auto process_value = [&](uint32_t p_id, int32_t p_val) {
            for (size_t k = 0; k < b_count; ++k) {
                if (b_vals[k] == p_val) {
                    local_buffer.add_match(b_ids[k], p_id);
                }
            }
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
                    for (uint16_t j = 0; j < num_rows; j++) {
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
            size_t start = (t_id * count) / worker_pool.thread_count();
            size_t end = ((t_id + 1) * count) / worker_pool.thread_count();

            for (size_t i = start; i < end; i++) {
                const mema::value_t &val = col[i];
                if (!val.is_null()) {
                    process_value((uint32_t)i, val.value);
                }
            }
        }
    });
    for (auto &buf : buffers) {
        collector.merge_thread_buffer(buf);
    }
}
} // namespace Contest::join

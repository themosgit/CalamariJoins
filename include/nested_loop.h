/*nested_loop.h*/
#pragma once

#include <columnar_reader.h>
#include <construct_intermediate.h>
#include <cstdint>
#include <intermediate.h>
#include <join_setup.h>
#include <plan.h>
#include <vector>
#include <atomic>
#include "worker_pool.h"
#include "match_collector.h"

namespace Contest {

/**
 *
 *  Nested Loop Join Implementation
 *   Optimized for small build sides (broadcast join).
 *  Uses stack-allocated arrays for build data to maximize register usage
 *  and SIMD auto-vectorization during the probe phase.
 *
 **/
inline void nested_loop_join(const JoinInput &build_input,
                             const JoinInput &probe_input, size_t build_attr,
                             size_t probe_attr, ColumnarReader &columnar_reader,
                             MatchCollector &collector) {
    size_t build_rows = build_input.row_count(build_attr);
    size_t probe_rows = probe_input.row_count(probe_attr);

    if (build_rows == 0 || probe_rows == 0) return;

    // -------------------------------------------------------------
    // 1. FAST PATH: Materialize Build Side into Stack Arrays
    // -------------------------------------------------------------
    // Split values and IDs for better cache locality (Structure of Arrays)
    // Using fixed size arrays since build is known to be small (typically < 64)
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

    // Serial scan of tiny build side
    if (build_input.is_columnar()) {
        auto *table = std::get<const ColumnarTable *>(build_input.data);
        auto [col_idx, _] = build_input.node->output_attrs[build_attr];
        const Column &col = table->columns[col_idx];
        const auto &prefix = columnar_reader.get_build_page_index(build_attr);

        uint32_t row_id = 0;
        size_t page_idx = 0;
        for (auto *page_obj : col.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<uint16_t *>(page);
            auto num_values = *reinterpret_cast<uint16_t *>(page + 2);
            auto *data = reinterpret_cast<const int32_t *>(page + 4);
            const auto &page_prefix = prefix.page_prefix_sums[page_idx];

            for (uint16_t i = 0; i < num_rows; i++) {
                if (num_rows == num_values) {
                    collect_build(row_id++, data[i]);
                } else {
                    auto *bitmap = reinterpret_cast<const uint8_t *>(page + PAGE_SIZE - (num_rows + 7) / 8);
                    if (bitmap[i / 8] & (1u << (i % 8))) {
                        size_t chunk = i >> 6;
                        uint16_t idx = page_prefix[chunk] + __builtin_popcountll(((const uint64_t*)bitmap)[chunk] & ((1ULL << (i & 0x3F)) - 1));
                        collect_build(row_id, data[idx]);
                    }
                    row_id++;
                }
            }
            page_idx++;
        }
    } else {
        const auto &res = std::get<ExecuteResult>(build_input.data);
        const mema::column_t &col = res[build_attr];
        size_t count = col.row_count();
        for (size_t i = 0; i < count; i++) {
            const mema::value_t &val = col[i];
            if (!val.is_null()) collect_build((uint32_t)i, val.value);
        }
    }

    // -------------------------------------------------------------
    // 2. PARALLEL PROBE: Scan Probe Once, Check against Buffer
    // -------------------------------------------------------------
    
    // Use the new slab-based thread local buffers
    int num_threads = worker_pool.thread_count();
    std::vector<ThreadLocalMatchBuffer> buffers(num_threads);

    // Prepare Probe Metadata for random access
    const Column *probe_col = nullptr;
    std::vector<uint32_t> page_offsets;
    if (probe_input.is_columnar()) {
        auto *table = std::get<const ColumnarTable *>(probe_input.data);
        auto [col_idx, _] = probe_input.node->output_attrs[probe_attr];
        probe_col = &table->columns[col_idx];
        
        // Map global row IDs to pages
        page_offsets.reserve(probe_col->pages.size() + 1);
        uint32_t current = 0;
        for (auto *p : probe_col->pages) {
            page_offsets.push_back(current);
            current += *reinterpret_cast<const uint16_t *>(p->data);
        }
        page_offsets.push_back(current);
    }

    worker_pool.execute([&](size_t t_id, size_t total_threads) {
        auto &local_buffer = buffers[t_id];

        // INLINED INNER LOOP: Compare one probe val against all build vals
        // This encourages the compiler to use SIMD/Registers for b_vals
        auto process_value = [&](uint32_t p_id, int32_t p_val) {
            for (size_t k = 0; k < b_count; ++k) {
                if (b_vals[k] == p_val) {
                    local_buffer.add_match(b_ids[k], p_id);
                }
            }
        };

        if (probe_input.is_columnar()) {
            size_t num_pages = probe_col->pages.size();
            size_t start = (t_id * num_pages) / total_threads;
            size_t end = ((t_id + 1) * num_pages) / total_threads;

            for (size_t i = start; i < end; ++i) {
                auto *page = probe_col->pages[i]->data;
                auto num_rows = *reinterpret_cast<const uint16_t *>(page);
                auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
                auto *data = reinterpret_cast<const int32_t *>(page + 4);
                uint32_t row_id = page_offsets[i];

                if (num_rows == num_values) {
                    // Fast path: No nulls
                    for (uint16_t j = 0; j < num_rows; j++) {
                        process_value(row_id++, data[j]);
                    }
                } else {
                    // Bitmap path
                    auto *bitmap = reinterpret_cast<const uint8_t *>(page + PAGE_SIZE - (num_rows + 7) / 8);
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
            size_t start = (t_id * count) / total_threads;
            size_t end = ((t_id + 1) * count) / total_threads;

            for (size_t i = start; i < end; i++) {
                const mema::value_t &val = col[i];
                if (!val.is_null()) {
                    process_value((uint32_t)i, val.value);
                }
            }
        }
    });

    // -------------------------------------------------------------
    // 3. MERGE (Zero-Copy Pointer Swapping)
    // -------------------------------------------------------------
    for (auto &buf : buffers) {
        collector.merge_thread_buffer(buf);
    }
}

} // namespace Contest

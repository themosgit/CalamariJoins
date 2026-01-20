#pragma once

#include <data_model/intermediate.h>
#include <join_execution/hashtable.h>
#include <join_execution/join_setup.h>
#include <join_execution/match_collector.h>
#include <platform/worker_pool.h>

/**
 * @file hash_join.h
 * @brief Hash join build and probe operations.
 *
 * Supports ColumnarTable and intermediate (column_t) inputs. Probe uses
 * parallel work-stealing; thread-local match buffers merged after processing.
 *
 * @see hashtable.h, match_collector.h
 */

/**
 * @namespace Contest::join
 * @brief Hash join build/probe: build_from_columnar(), probe_intermediate(),
 * probe_columnar().
 */
namespace Contest::join {

using Contest::ExecuteResult;
using Contest::platform::THREAD_COUNT;
using Contest::platform::worker_pool;

/**
 * @brief Build hash table from ColumnarTable input.
 *
 * Maps logical attr index to physical column, delegates to build_columnar().
 */
inline UnchainedHashtable build_from_columnar(const JoinInput &input,
                                              size_t attr_idx) {
    auto *table = std::get<const ColumnarTable *>(input.data);
    auto [actual_col_idx, _] = input.node->output_attrs[attr_idx];
    const Column &column = table->columns[actual_col_idx];

    size_t row_count = input.row_count(attr_idx);
    UnchainedHashtable hash_table(row_count);
    hash_table.build_columnar(column, 8);

    return hash_table;
}

/**
 * @brief Build hash table from intermediate results (column_t).
 *
 * Uses join key column from ExecuteResult produced by prior pipeline stages.
 */
inline UnchainedHashtable build_from_intermediate(const JoinInput &input,
                                                  size_t attr_idx) {
    const auto &result = std::get<ExecuteResult>(input.data);
    const auto &column = result[attr_idx];

    size_t row_count = input.row_count(attr_idx);
    UnchainedHashtable hash_table(row_count);
    hash_table.build_intermediate(column, 8);

    return hash_table;
}

/**
 * @brief Probe hash table with intermediate input, returning thread-local
 * buffers.
 *
 * Each thread keeps its buffer for direct iteration without merge overhead.
 */
inline std::vector<ThreadLocalMatchBuffer>
probe_intermediate(const UnchainedHashtable &hash_table,
                   const mema::column_t &probe_column,
                   MatchCollectionMode mode = MatchCollectionMode::BOTH) {
    const auto *keys = hash_table.keys();
    const auto *row_ids = hash_table.row_ids();

    size_t pool_size = THREAD_COUNT;
    std::vector<ThreadLocalMatchBuffer> local_buffers(pool_size);

    const size_t num_pages = probe_column.pages.size();
    const size_t probe_count = probe_column.row_count();
    std::atomic<size_t> page_counter(0);

    worker_pool().execute([&](size_t thread_id) {
        local_buffers[thread_id] = ThreadLocalMatchBuffer(
            Contest::platform::get_arena(thread_id), mode);
        auto &local_buf = local_buffers[thread_id];

        while (true) {
            size_t page_idx = page_counter.fetch_add(1);
            if (page_idx >= num_pages)
                break;

            size_t base_row = page_idx * mema::CAP_PER_PAGE;
            size_t page_end =
                std::min(base_row + mema::CAP_PER_PAGE, probe_count);

            constexpr size_t PREFETCH_DIST = 8;
            for (size_t idx = base_row; idx < page_end; ++idx) {
                if (idx + PREFETCH_DIST < page_end) {
                    const mema::value_t &future_key =
                        probe_column[idx + PREFETCH_DIST];
                    if (!future_key.is_null()) {
                        hash_table.prefetch_slot(future_key.value);
                    }
                }

                const mema::value_t &key = probe_column[idx];

                if (!key.is_null()) {
                    int32_t key_val = key.value;
                    auto [start_idx, end_idx] =
                        hash_table.find_indices(key_val);

                    for (uint64_t i = start_idx; i < end_idx; ++i) {
                        if (keys[i] == key_val) {
                            local_buf.add_match(row_ids[i],
                                                static_cast<uint32_t>(idx));
                        }
                    }
                }
            }
        }
    });

    return local_buffers;
}

/**
 * @brief Probe hash table with ColumnarTable input, returning thread-local
 * buffers.
 *
 * Each thread keeps its buffer for direct iteration without merge overhead.
 */
inline std::vector<ThreadLocalMatchBuffer>
probe_columnar(const UnchainedHashtable &hash_table,
               const JoinInput &probe_input, size_t probe_attr,
               MatchCollectionMode mode = MatchCollectionMode::BOTH) {

    const auto *keys = hash_table.keys();
    const auto *row_ids = hash_table.row_ids();

    auto *table = std::get<const ColumnarTable *>(probe_input.data);
    auto [actual_idx_col, _] = probe_input.node->output_attrs[probe_attr];
    const Column &probe_col = table->columns[actual_idx_col];
    size_t num_pages = probe_col.pages.size();

    std::vector<uint32_t> page_offsets;
    page_offsets.reserve(num_pages);
    uint32_t running_offset = 0;

    for (const auto *page_obj : probe_col.pages) {
        page_offsets.push_back(running_offset);
        auto num_rows = *reinterpret_cast<const uint16_t *>(page_obj->data);
        running_offset += num_rows;
    }

    size_t pool_size = THREAD_COUNT;
    std::vector<ThreadLocalMatchBuffer> local_buffers(pool_size);

    std::atomic<size_t> page_counter(0);
    worker_pool().execute([&](size_t thread_id) {
        local_buffers[thread_id] = ThreadLocalMatchBuffer(
            Contest::platform::get_arena(thread_id), mode);
        auto &local_buf = local_buffers[thread_id];

        while (true) {
            size_t page_idx = page_counter.fetch_add(1);
            if (page_idx >= num_pages)
                break;

            auto *page = probe_col.pages[page_idx]->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page);
            auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
            auto *data_begin = reinterpret_cast<const int32_t *>(page + 4);
            uint32_t probe_row_id = page_offsets[page_idx];

            if (num_rows == num_values) {
                constexpr int PREFETCH_DIST = 32;
                for (uint16_t i = 0; i < num_rows; ++i) {
                    hash_table.prefetch_slot(data_begin[i + PREFETCH_DIST]);

                    int32_t key_val = data_begin[i];
                    auto [start_idx, end_idx] =
                        hash_table.find_indices(key_val);

                    for (uint64_t j = start_idx; j < end_idx; ++j) {
                        if (keys[j] == key_val) {
                            local_buf.add_match(row_ids[j], probe_row_id);
                        }
                    }
                    probe_row_id++;
                }
            } else {
                auto *bitmap = reinterpret_cast<const uint8_t *>(
                    page + PAGE_SIZE - (num_rows + 7) / 8);
                constexpr int PREFETCH_DIST = 32;
                uint16_t data_idx = 0;
                for (uint16_t i = 0; i < num_rows; ++i) {
                    bool is_valid = bitmap[i / 8] & (1u << (i % 8));
                    if (is_valid) {
                        hash_table.prefetch_slot(
                            data_begin[data_idx + PREFETCH_DIST]);

                        int32_t key_val = data_begin[data_idx++];
                        auto [start_idx, end_idx] =
                            hash_table.find_indices(key_val);

                        for (uint64_t j = start_idx; j < end_idx; ++j) {
                            if (keys[j] == key_val) {
                                local_buf.add_match(row_ids[j], probe_row_id);
                            }
                        }
                    }
                    probe_row_id++;
                }
            }
        }
    });

    return local_buffers;
}

} // namespace Contest::join

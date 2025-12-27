#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <intermediate.h>
#include <vector>
#include <thread>
#include <algorithm>
#include <atomic>
#include <hardware.h>
#include <inner_column.h>

enum class BuildType { DENSE, SPARSE, COLUMN };

#if defined(__x86_64__)
    #include  <immintrin.h>
#endif

class UnchainedHashtable {
  public:
    struct Tuple {
        int32_t key;
        uint32_t row_id;
    };

  private:
    std::vector<uint64_t> directory;

    /* structure of srrays layout for better cache locality */
    std::vector<int32_t> keys;
    std::vector<uint32_t> row_ids;

    int shift;


    static inline uint64_t hash_key(uint32_t key) noexcept {
        uint64_t k = 0x8648DBDB;
        #if defined(__aarch64__)
            uint32_t crc = __builtin_arm_crc32w(0, key);
        #else
            uint32_t crc = __builtin_ia32_crc32si(0, key);
        #endif
        return crc * ((k << 32) + 1);
    }

    /**
     *
     *  a clever methods that sets
     *  sets of 4 bits with different
     *  parts of the hash allowing
     *  for more sparsity and
     *  fewer false positives
     *
     **/
    static inline uint16_t compute_bloom(uint64_t hash) {
        uint16_t mask = 0;
        mask |= (1 << (hash & 0xF));
        mask |= (1 << ((hash >> 4) & 0xF));
        mask |= (1 << ((hash >> 8) & 0xF));
        mask |= (1 << ((hash >> 12) & 0xF));
        return mask;
    }

  public:
    UnchainedHashtable(size_t build_size) {
        size_t dir_size = build_size > 0 ? build_size : 1;
        size_t pow2 = 1ULL << (64 - __builtin_clzll(dir_size - 1));

        if (pow2 < 2048)
            pow2 = 2048;

        directory.resize(pow2, 0);

        /**
         *
         *  shift determines the bits used
         *  to select a bucket be using
         *  trailing zeros etc.
         *
         *  this is the exact amout of bits
         *  we need to reference its slot
         *  in the directory
         *
         **/
        shift = 64 - __builtin_ctzll(pow2);
        keys.reserve(build_size);
        row_ids.reserve(build_size);
    }

    void build(const mema::column_t &column) {
        if (column.has_direct_access()) {
            build_dense_parallel(column);
        } else {
            build_sparse_parallel(column);
        }
    }

    void build(const Column &column) { build_from_column_parallel(column); }

    void build_parallel(const mema::column_t &column) {
        if (column.has_direct_access()) {
            build_dense_parallel(column);
        } else {
            build_sparse_parallel(column);
        }
    }

    void build_parallel(const Column &column) { build_from_column_parallel(column); }

  private:
    /* Helper methods for unified build implementation */

    // Count phase helpers
    void count_dense(const mema::column_t &column, size_t thread_id, size_t num_threads,
                     size_t chunk_size, std::vector<std::vector<uint32_t>> &thread_counts) {
        size_t rows = column.row_count();
        size_t start = thread_id * chunk_size;
        size_t end_idx = std::min(start + chunk_size, rows);
        for (size_t i = start; i < end_idx; ++i) {
            uint64_t h = hash_key(column[i].value);
            thread_counts[thread_id][h >> shift]++;
        }
    }

    void count_sparse(const mema::column_t &column, size_t thread_id, size_t num_threads,
                      size_t chunk_size, std::vector<std::vector<uint32_t>> &thread_counts) {
        size_t rows = column.row_count();
        size_t start = thread_id * chunk_size;
        size_t end_idx = std::min(start + chunk_size, rows);
        for (size_t i = start; i < end_idx; ++i) {
            const mema::value_t *val = column.get_by_row(i);
            if (val) {
                uint64_t h = hash_key(val->value);
                thread_counts[thread_id][h >> shift]++;
            }
        }
    }

    void count_from_column(const Column &column, size_t thread_id, size_t num_threads,
                           size_t rows_per_thread, size_t total_rows,
                           std::vector<std::vector<uint32_t>> &thread_counts) {
        size_t thread_start_row = thread_id * rows_per_thread;
        size_t thread_end_row = std::min(thread_start_row + rows_per_thread, total_rows);
        size_t current_row = 0;

        for (auto *page_obj : column.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page);

            if (current_row >= thread_end_row) break;
            if (current_row + num_rows <= thread_start_row) {
                current_row += num_rows;
                continue;
            }

            auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
            auto *data_begin = reinterpret_cast<const int32_t *>(page + 4);

            size_t page_start = current_row;
            size_t page_end = current_row + num_rows;
            size_t local_start = std::max(thread_start_row, page_start) - page_start;
            size_t local_end = std::min(thread_end_row, page_end) - page_start;

            if (num_rows == num_values) {
                for (size_t i = local_start; i < local_end; ++i) {
                    uint64_t h = hash_key(data_begin[i]);
                    thread_counts[thread_id][h >> shift]++;
                }
            } else {
                auto *bitmap = reinterpret_cast<const uint8_t *>(
                    page + PAGE_SIZE - (num_rows + 7) / 8);
                uint16_t data_idx = 0;
                for (uint16_t i = 0; i < num_rows; ++i) {
                    if (i >= local_start && i < local_end) {
                        bool is_valid = bitmap[i / 8] & (1u << (i % 8));
                        if (is_valid) {
                            uint64_t h = hash_key(data_begin[data_idx]);
                            thread_counts[thread_id][h >> shift]++;
                            data_idx++;
                        }
                    } else if (i < local_start) {
                        bool is_valid = bitmap[i / 8] & (1u << (i % 8));
                        if (is_valid) data_idx++;
                    }
                }
            }

            current_row += num_rows;
        }
    }

    // Scatter phase helpers
    void scatter_dense(const mema::column_t &column, size_t thread_id, size_t num_threads,
                       size_t chunk_size, std::vector<std::vector<uint32_t>> &thread_offsets,
                       std::vector<std::vector<uint16_t>> &thread_blooms) {
        size_t rows = column.row_count();
        size_t start = thread_id * chunk_size;
        size_t end_idx = std::min(start + chunk_size, rows);
        for (size_t i = start; i < end_idx; ++i) {
            int32_t val = column[i].value;
            uint64_t h = hash_key(val);
            size_t slot = h >> shift;
            uint32_t pos = thread_offsets[thread_id][slot]++;
            keys[pos] = val;
            row_ids[pos] = (uint32_t)i;
            thread_blooms[thread_id][slot] |= compute_bloom(h);
        }
    }

    void scatter_sparse(const mema::column_t &column, size_t thread_id, size_t num_threads,
                        size_t chunk_size, std::vector<std::vector<uint32_t>> &thread_offsets,
                        std::vector<std::vector<uint16_t>> &thread_blooms) {
        size_t rows = column.row_count();
        size_t start = thread_id * chunk_size;
        size_t end_idx = std::min(start + chunk_size, rows);
        for (size_t i = start; i < end_idx; ++i) {
            const mema::value_t *val = column.get_by_row(i);
            if (val) {
                int32_t key_val = val->value;
                uint64_t h = hash_key(key_val);
                size_t slot = h >> shift;
                uint32_t pos = thread_offsets[thread_id][slot]++;
                keys[pos] = key_val;
                row_ids[pos] = (uint32_t)i;
                thread_blooms[thread_id][slot] |= compute_bloom(h);
            }
        }
    }

    void scatter_from_column(const Column &column, size_t thread_id, size_t num_threads,
                             size_t rows_per_thread, size_t total_rows,
                             std::vector<std::vector<uint32_t>> &thread_offsets,
                             std::vector<std::vector<uint16_t>> &thread_blooms) {
        size_t thread_start_row = thread_id * rows_per_thread;
        size_t thread_end_row = std::min(thread_start_row + rows_per_thread, total_rows);
        size_t current_row = 0;
        uint32_t global_row_id = 0;

        for (auto *page_obj : column.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page);

            if (current_row >= thread_end_row) break;
            if (current_row + num_rows <= thread_start_row) {
                current_row += num_rows;
                global_row_id += num_rows;
                continue;
            }

            auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
            auto *data_begin = reinterpret_cast<const int32_t *>(page + 4);

            size_t page_start = current_row;
            size_t page_end = current_row + num_rows;
            size_t local_start = std::max(thread_start_row, page_start) - page_start;
            size_t local_end = std::min(thread_end_row, page_end) - page_start;

            if (num_rows == num_values) {
                for (size_t i = local_start; i < local_end; ++i) {
                    int32_t val = data_begin[i];
                    uint64_t h = hash_key(val);
                    size_t slot = h >> shift;
                    uint32_t pos = thread_offsets[thread_id][slot]++;
                    keys[pos] = val;
                    row_ids[pos] = global_row_id + i;
                    thread_blooms[thread_id][slot] |= compute_bloom(h);
                }
            } else {
                auto *bitmap = reinterpret_cast<const uint8_t *>(
                    page + PAGE_SIZE - (num_rows + 7) / 8);
                uint16_t data_idx = 0;
                // Skip to local_start
                for (uint16_t i = 0; i < local_start; ++i) {
                    bool is_valid = bitmap[i / 8] & (1u << (i % 8));
                    if (is_valid) data_idx++;
                }
                for (uint16_t i = local_start; i < local_end && i < num_rows; ++i) {
                    bool is_valid = bitmap[i / 8] & (1u << (i % 8));
                    if (is_valid) {
                        int32_t val = data_begin[data_idx++];
                        uint64_t h = hash_key(val);
                        size_t slot = h >> shift;
                        uint32_t pos = thread_offsets[thread_id][slot]++;
                        keys[pos] = val;
                        row_ids[pos] = global_row_id + i;
                        thread_blooms[thread_id][slot] |= compute_bloom(h);
                    }
                }
            }

            current_row += num_rows;
            global_row_id += num_rows;
        }
    }

    // Shared merge operations
    void merge_and_prepare_offsets(const std::vector<std::vector<uint32_t>> &thread_counts,
                                    size_t num_threads, std::vector<uint32_t> &counts,
                                    std::vector<std::vector<uint32_t>> &thread_offsets) {
        // Serial count merge (faster than parallel for small work)
        counts.resize(directory.size(), 0);
        for (size_t i = 0; i < directory.size(); ++i) {
            for (size_t t = 0; t < num_threads; ++t) {
                counts[i] += thread_counts[t][i];
            }
        }

        // Prefix sum
        uint32_t current_offset = 0;
        for (size_t i = 0; i < counts.size(); ++i) {
            uint32_t count = counts[i];
            counts[i] = current_offset;
            current_offset += count;
        }

        keys.resize(current_offset);
        row_ids.resize(current_offset);

        // Compute per-thread starting offsets for each bucket
        thread_offsets.resize(num_threads);
        for (size_t t = 0; t < num_threads; ++t) {
            thread_offsets[t].resize(directory.size(), 0);
        }

        for (size_t slot = 0; slot < directory.size(); ++slot) {
            uint32_t offset = counts[slot];
            for (size_t t = 0; t < num_threads; ++t) {
                thread_offsets[t][slot] = offset;
                offset += thread_counts[t][slot];
            }
        }
    }

    void merge_blooms_and_finalize(const std::vector<std::vector<uint16_t>> &thread_blooms,
                                    size_t num_threads, std::vector<uint32_t> &counts,
                                    const std::vector<std::vector<uint32_t>> &thread_offsets) {
        // Serial bloom merge (faster than parallel for small work)
        for (size_t i = 0; i < directory.size(); ++i) {
            for (size_t t = 0; t < num_threads; ++t) {
                directory[i] |= thread_blooms[t][i];
            }
        }

        // Reconstruct final counts from thread_offsets
        for (size_t i = 0; i < directory.size(); ++i) {
            counts[i] = thread_offsets[num_threads - 1][i];
        }

        finalize_directory(counts);
    }

    /* parallel build for dense column - simplified using helper methods */
    void build_dense_parallel(const mema::column_t &column) {
        const size_t rows = column.row_count();
        size_t num_threads = SPC__THREAD_COUNT;

        // Ensure thread count is power of 2
        if (num_threads == 0) num_threads = 1;
        num_threads = 1ULL << (64 - __builtin_clzll(num_threads - 1));
        if (num_threads > rows) num_threads = 1;

        const size_t chunk_size = (rows + num_threads - 1) / num_threads;

        // Thread-local counts to avoid contention
        std::vector<std::vector<uint32_t>> thread_counts(num_threads, std::vector<uint32_t>(directory.size(), 0));

        // Phase 1: Parallel counting
        {
            std::vector<std::thread> threads;
            for (size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    count_dense(column, t, num_threads, chunk_size, thread_counts);
                });
            }
            for (auto &thread : threads) {
                thread.join();
            }
        }

        // Serial merge + prefix sum + thread offset computation
        std::vector<uint32_t> counts;
        std::vector<std::vector<uint32_t>> thread_offsets;
        merge_and_prepare_offsets(thread_counts, num_threads, counts, thread_offsets);

        // Phase 2: Parallel scatter
        std::vector<std::vector<uint16_t>> thread_blooms(num_threads, std::vector<uint16_t>(directory.size(), 0));

        {
            std::vector<std::thread> threads;
            for (size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    scatter_dense(column, t, num_threads, chunk_size, thread_offsets, thread_blooms);
                });
            }
            for (auto &thread : threads) {
                thread.join();
            }
        }

        // Serial bloom merge + finalize
        merge_blooms_and_finalize(thread_blooms, num_threads, counts, thread_offsets);
    }

    /* parallel build for sparse column - simplified using helper methods */
    void build_sparse_parallel(const mema::column_t &column) {
        const size_t rows = column.row_count();
        size_t num_threads = SPC__THREAD_COUNT;

        // Ensure thread count is power of 2
        if (num_threads == 0) num_threads = 1;
        num_threads = 1ULL << (64 - __builtin_clzll(num_threads - 1));
        if (num_threads > rows) num_threads = 1;

        const size_t chunk_size = (rows + num_threads - 1) / num_threads;

        // Thread-local counts to avoid contention
        std::vector<std::vector<uint32_t>> thread_counts(num_threads, std::vector<uint32_t>(directory.size(), 0));

        // Phase 1: Parallel counting
        {
            std::vector<std::thread> threads;
            for (size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    count_sparse(column, t, num_threads, chunk_size, thread_counts);
                });
            }
            for (auto &thread : threads) {
                thread.join();
            }
        }

        // Serial merge + prefix sum + thread offset computation
        std::vector<uint32_t> counts;
        std::vector<std::vector<uint32_t>> thread_offsets;
        merge_and_prepare_offsets(thread_counts, num_threads, counts, thread_offsets);

        // Phase 2: Parallel scatter
        std::vector<std::vector<uint16_t>> thread_blooms(num_threads, std::vector<uint16_t>(directory.size(), 0));

        {
            std::vector<std::thread> threads;
            for (size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    scatter_sparse(column, t, num_threads, chunk_size, thread_offsets, thread_blooms);
                });
            }
            for (auto &thread : threads) {
                thread.join();
            }
        }

        // Serial bloom merge + finalize
        merge_blooms_and_finalize(thread_blooms, num_threads, counts, thread_offsets);
    }

    /* parallel build from Column format - simplified using helper methods */
    void build_from_column_parallel(const Column &column) {
        // First pass: count total rows
        size_t total_rows = 0;
        for (auto *page_obj : column.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page);
            total_rows += num_rows;
        }

        size_t num_threads = SPC__THREAD_COUNT;
        if (num_threads == 0) num_threads = 1;
        num_threads = 1ULL << (64 - __builtin_clzll(num_threads - 1));
        if (num_threads > total_rows) num_threads = 1;

        size_t rows_per_thread = (total_rows + num_threads - 1) / num_threads;

        // Thread-local counts to avoid contention
        std::vector<std::vector<uint32_t>> thread_counts(num_threads, std::vector<uint32_t>(directory.size(), 0));

        // Phase 1: Parallel counting
        {
            std::vector<std::thread> threads;
            for (size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    count_from_column(column, t, num_threads, rows_per_thread, total_rows, thread_counts);
                });
            }
            for (auto &thread : threads) {
                thread.join();
            }
        }

        // Serial merge + prefix sum + thread offset computation
        std::vector<uint32_t> counts;
        std::vector<std::vector<uint32_t>> thread_offsets;
        merge_and_prepare_offsets(thread_counts, num_threads, counts, thread_offsets);

        // Phase 2: Parallel scatter
        std::vector<std::vector<uint16_t>> thread_blooms(num_threads, std::vector<uint16_t>(directory.size(), 0));

        {
            std::vector<std::thread> threads;
            for (size_t t = 0; t < num_threads; ++t) {
                threads.emplace_back([&, t]() {
                    scatter_from_column(column, t, num_threads, rows_per_thread, total_rows, thread_offsets, thread_blooms);
                });
            }
            for (auto &thread : threads) {
                thread.join();
            }
        }

        // Serial bloom merge + finalize
        merge_blooms_and_finalize(thread_blooms, num_threads, counts, thread_offsets);
    }

    /**
     *
     *  packs blooms filter and value store index
     *  upper 48 bits are the index in value store lower 16 bits
     *  are the bloom filter
     *
     **/
    inline void finalize_directory(const std::vector<uint32_t> &final_offsets) {
        // Serial execution - this operation is too lightweight to benefit from threading
        for (size_t i = 0; i < directory.size(); ++i) {
            uint64_t end_idx = final_offsets[i];
            uint64_t bloom = directory[i] & 0xFFFF;
            directory[i] = (end_idx << 16) | bloom;
        }
    }

  public:
    /* returns index range into keys/row_ids arrays for a given key */
    inline std::pair<uint64_t, uint64_t> find_indices(int32_t key) const {
        uint64_t h = hash_key(key);
        size_t slot = h >> shift;
        uint64_t entry = directory[slot];

        uint16_t filter = (uint16_t)entry;
        uint16_t key_mask = compute_bloom(h);

        if ((filter & key_mask) != key_mask) {
            return {0, 0};
        }

        uint64_t end_idx = entry >> 16;
        uint64_t start_idx = (slot == 0) ? 0 : (directory[slot - 1] >> 16);

        if (start_idx >= end_idx) {
            return {0, 0};
        }

        return {start_idx, end_idx};
    }

    inline const int32_t *get_keys() const { return keys.data(); }
    inline const uint32_t *get_row_ids() const { return row_ids.data(); }
};

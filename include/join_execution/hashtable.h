/**
 * @file hashtable.h
 * @brief Unchained hash table with bloom filter for parallel equi-joins.
 *
 * Keys/row_ids in contiguous arrays (not linked). Directory entry: bits 16-63
 * = end offset, bits 0-15 = bloom tag. Radix-partitioned parallel build sizes
 * partitions to fit LLC.
 *
 * @note Join keys are INT32 (contest invariant).
 * @see hash_join.h, bloom_tags.h
 */
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <data_model/intermediate.h>
#include <join_execution/bloom_tags.h>
#include <platform/arena.h>
#include <platform/arena_vector.h>
#include <platform/worker_pool.h>
#include <vector>

#if defined(__APPLE__) && defined(__aarch64__)
#include <platform/hardware_darwin.h>
#elif defined(SPC__USE_BENCHMARKVM_HARDWARE)
#include <platform/hardware_benchmarkvm.h>
#else
#include <platform/hardware.h>
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_acle.h>
#endif

// Use L3 if available, otherwise L2 as last-level cache
#if defined(SPC__LEVEL3_CACHE_SIZE) && SPC__LEVEL3_CACHE_SIZE > 0
static constexpr size_t LAST_LEVEL_CACHE = SPC__LEVEL3_CACHE_SIZE;
#else
static constexpr size_t LAST_LEVEL_CACHE = SPC__LEVEL2_CACHE_SIZE;
#endif

static constexpr size_t L2_CACHE = SPC__LEVEL2_CACHE_SIZE;
static constexpr size_t CACHE_LINE = SPC__LEVEL1_DCACHE_LINESIZE;

using Contest::join::BLOOM_TAGS;

/**
 * @brief High-performance unchained hash table for parallel equi-joins.
 *
 * Keys/row_ids in contiguous arrays (not linked lists) for cache-friendly
 * sequential probe. Bloom filter tags enable early rejection. Hardware CRC32
 * hash with multiplicative mixing.
 *
 * @note Build O(n), probe O(1) avg. Thread-safe build via lock-free partitions.
 */
class UnchainedHashtable {
  public:
    /** @brief Key-rowid pair for hash table entries. */
    struct alignas(4) Tuple {
        int32_t key;     /**< Join key value. */
        uint32_t row_id; /**< Row index in source table. */
    };

    /** @brief L2-sized chunk for partition buffers. */
    static constexpr size_t CHUNK_SIZE = 4096;
    static constexpr size_t CHUNK_HEADER = 16;
    static constexpr size_t TUPLES_PER_CHUNK =
        (CHUNK_SIZE - CHUNK_HEADER) / sizeof(Tuple);

    /** @brief Linked chunk for partition buffers during radix build. */
    struct alignas(8) Chunk {
        Chunk *next;                  /**< Next chunk in partition chain. */
        size_t count;                 /**< Number of tuples in this chunk. */
        Tuple data[TUPLES_PER_CHUNK]; /**< Tuple storage. */
    };

    /** @brief Thread-local chunk allocator using global arena. */
    class ChunkAllocator {
        Contest::platform::ThreadArena *arena_ = nullptr;

      public:
        ChunkAllocator() = default;
        explicit ChunkAllocator(Contest::platform::ThreadArena &arena)
            : arena_(&arena) {}

        void set_arena(Contest::platform::ThreadArena &arena) {
            arena_ = &arena;
        }

        Chunk *alloc() {
            void *ptr =
                arena_->alloc_chunk<Contest::platform::ChunkType::HASH_CHUNK>();
            Chunk *c = static_cast<Chunk *>(ptr);
            c->next = nullptr;
            c->count = 0;
            return c;
        }
    };

    /**
     * @brief Partition of directory entries for lock-free parallel build.
     */
    struct alignas(8) Partition {
        Chunk *head = nullptr;
        Chunk *tail = nullptr;
        size_t total_count = 0;

        void append(ChunkAllocator &alloc, Tuple t) {
            if (!tail || tail->count == TUPLES_PER_CHUNK) {
                Chunk *c = alloc.alloc();
                if (tail)
                    tail->next = c;
                else
                    head = c;
                tail = c;
            }
            tail->data[tail->count++] = t;
            ++total_count;
        }
    };

  private:
    Contest::platform::ThreadArena *arena_ =
        nullptr; /**< Arena for hash table allocations. */
    Contest::platform::ArenaVector<uint64_t>
        directory; /**< Slot entries: (end_offset << 16) | bloom_tag. */
    Contest::platform::ArenaVector<int32_t>
        keys_; /**< Contiguous key storage, indexed by directory. */
    Contest::platform::ArenaVector<uint32_t>
        row_ids_; /**< Parallel row_id storage, same indexing. */
    int shift =
        0; /**< Bit shift for slot calculation: slot = hash >> (64-shift). */

    /**
     * @brief CRC32-based hash with multiplicative mixing.
     * @param key INT32 join key.
     * @return 64-bit hash (upper bits index directory slot).
     */
    static uint64_t hash_key(int32_t key) noexcept {
        constexpr uint64_t k = 0x8648DBDB;
#if defined(__aarch64__)
        uint32_t crc = __crc32w(0, static_cast<uint32_t>(key));
#else
        uint32_t crc = _mm_crc32_u32(0, static_cast<uint32_t>(key));
#endif
        return crc * ((k << 32) + 1);
    }

    /**
     * @brief Returns bloom tag from hash. Uses bits 32-42 to index BLOOM_TAGS.
     * @see bloom_tags.h
     */
    static uint16_t bloom_tag(uint64_t h) noexcept {
        return BLOOM_TAGS[(h >> 32) & 0x7FF];
    }

    inline size_t slot_for(uint64_t h) const noexcept {
        return h >> (64 - shift);
    }

    /**
     * @brief Computes partition count to fit each in per-core LLC share.
     *
     * Targets 50% of per-core LLC. Power-of-2 for fast modulo via bit shift.
     */
    size_t compute_num_partitions(size_t tuple_count, int num_threads) const {
        size_t per_core_cache =
            LAST_LEVEL_CACHE / Contest::platform::worker_pool().thread_count();
        size_t target_bytes = per_core_cache / 2;
        size_t bytes_per_tuple = sizeof(Tuple);
        size_t tuples_per_partition = target_bytes / bytes_per_tuple;

        size_t min_partitions =
            (tuple_count + tuples_per_partition - 1) / tuples_per_partition;
        min_partitions =
            std::max(min_partitions, static_cast<size_t>(num_threads));

        size_t p = 1;
        while (p < min_partitions)
            p <<= 1;
        return std::min(p, directory.size());
    }

    /**
     * @brief Merges and scatters a single partition from thread-local buffers.
     *
     * Count phase → prefix sum → scatter keys/row_ids and OR bloom tags.
     * Lock-free: each thread handles disjoint partitions.
     */
    void
    build_partition(const std::vector<std::vector<Partition>> &thread_parts,
                    size_t p, size_t slots_per_partition, size_t base_offset,
                    size_t partition_size, int num_threads, size_t thread_id) {

        const size_t slot_start = p * slots_per_partition;
        // Write offset for empty partitions
        if (partition_size == 0) {
            for (size_t s = 0; s < slots_per_partition; ++s) {
                directory[slot_start + s] = base_offset << 16;
            }
            return;
        }
        // Count keys per slot within partition - use thread-local arena
        auto &arena = Contest::platform::get_arena(thread_id);
        Contest::platform::ArenaVector<uint32_t> counts(arena);
        counts.resize(slots_per_partition);
        std::memset(counts.data(), 0, slots_per_partition * sizeof(uint32_t));
        for (int t = 0; t < num_threads; ++t) {
            for (Chunk *c = thread_parts[t][p].head; c; c = c->next) {
                for (size_t i = 0; i < c->count; ++i) {
                    counts[slot_for(hash_key(c->data[i].key)) - slot_start]++;
                }
            }
        }
        // Prefix sum for write offsets
        Contest::platform::ArenaVector<uint32_t> offsets(arena);
        offsets.resize(slots_per_partition);
        uint32_t running = base_offset;
        for (size_t s = 0; s < slots_per_partition; ++s) {
            offsets[s] = running;
            running += counts[s];
            counts[s] = 0;
        }
        // Scatter values and update bloom filter
        for (int t = 0; t < num_threads; ++t) {
            for (Chunk *c = thread_parts[t][p].head; c; c = c->next) {
                for (size_t i = 0; i < c->count; ++i) {
                    const Tuple &tup = c->data[i];
                    uint64_t h = hash_key(tup.key);
                    size_t local_slot = slot_for(h) - slot_start;
                    uint32_t idx = offsets[local_slot] + counts[local_slot]++;
                    keys_[idx] = tup.key;
                    row_ids_[idx] = tup.row_id;
                    directory[slot_start + local_slot] |= bloom_tag(h);
                }
            }
        }
        // Finalize directory entries with end pointer and bloom bits
        for (size_t s = 0; s < slots_per_partition; ++s) {
            uint64_t end = offsets[s] + counts[s];
            directory[slot_start + s] =
                (end << 16) | (directory[slot_start + s] & 0xFFFF);
        }
    }

  public:
    /**
     * @brief Construct hash table sized for expected build-side row count.
     *
     * Directory size rounded to power of 2 (min 2048) for fast modulo.
     */
    explicit UnchainedHashtable(size_t build_size)
        : arena_(&Contest::platform::get_arena(0)), directory(*arena_),
          keys_(*arena_), row_ids_(*arena_) {
        size_t pow2 = 2048;
        while (pow2 < build_size)
            pow2 <<= 1;
        directory.resize(pow2);
        std::memset(directory.data(), 0, pow2 * sizeof(uint64_t));
        shift = __builtin_ctzll(pow2);
    }

    /** @brief Number of keys in the hash table. */
    size_t size() const noexcept { return keys_.size(); }

    /** @brief True if hash table is empty. */
    bool empty() const noexcept { return keys_.empty(); }

    /** @brief Direct access to key array for probe. */
    const int32_t *keys() const noexcept { return keys_.data(); }

    /** @brief Direct access to row_id array for probe. */
    const uint32_t *row_ids() const noexcept { return row_ids_.data(); }

    /**
     * @brief Prefetch directory slot for a key to hide memory latency.
     *
     * Call N iterations ahead in probe loop to overlap directory fetch
     * with current iteration's work. Typical PREFETCH_DIST: 8-16.
     */
    void prefetch_slot(int32_t key) const noexcept {
        uint64_t h = hash_key(key);
        size_t slot = slot_for(h);
        __builtin_prefetch(&directory[slot], 0, 2);
    }

    /**
     * @brief Find index range for keys matching probe key.
     *
     * @return [start, end) into keys_/row_ids_; (0,0) if bloom rejects.
     */
    std::pair<uint64_t, uint64_t> find_indices(int32_t key) const noexcept {
        if (keys_.empty())
            return {0, 0};

        uint64_t h = hash_key(key);
        size_t slot = slot_for(h);
        uint64_t entry = directory[slot];
        uint16_t tag = bloom_tag(h);

        if ((entry & tag) != tag)
            return {0, 0};

        uint64_t end = entry >> 16;
        uint64_t start = (slot == 0) ? 0 : (directory[slot - 1] >> 16);
        return {start, end};
    }

    /**
     * @brief Build hash table from intermediate column_t.
     *
     * Radix-partitioned parallel build when row count > 10K threshold.
     * Thread-local partition buffers avoid contention.
     *
     * @param column Intermediate column with INT32 join keys.
     * @param num_threads Thread count hint.
     */
    void build_intermediate(const mema::column_t &column, int num_threads = 4) {
        const size_t row_count = column.row_count();
        if (row_count == 0)
            return;

        /**
         * @brief Below 10K rows, single-threaded build is faster.
         */
        static constexpr size_t PARALLEL_BUILD_THRESHOLD = 10000;
        num_threads = Contest::platform::worker_pool().thread_count();
        if (row_count < PARALLEL_BUILD_THRESHOLD)
            num_threads = 1;

        const size_t num_slots = directory.size();
        const size_t num_partitions =
            compute_num_partitions(row_count, num_threads);
        const int partition_bits = __builtin_ctzll(num_partitions);
        const size_t slots_per_partition = num_slots / num_partitions;

        // Thread-local partitions for lock-free parallel partitioning
        std::vector<ChunkAllocator> allocators(num_threads);
        for (int t = 0; t < num_threads; ++t)
            allocators[t].set_arena(Contest::platform::get_arena(t));
        std::vector<std::vector<Partition>> thread_parts(num_threads);
        for (auto &tp : thread_parts)
            tp.resize(num_partitions);

        // Partition data by hash
        size_t batch = (row_count + num_threads - 1) / num_threads;
        Contest::platform::worker_pool().execute([&, partition_bits](size_t t) {
            size_t start = t * batch;
            size_t end = std::min(start + batch, row_count);
            if (start >= end)
                return;
            const int shift = 64 - partition_bits;
            for (size_t i = start; i < end; ++i) {
                int32_t val = column[i].value;
                uint64_t h = hash_key(val);
                size_t p = (partition_bits == 0) ? 0 : (h >> shift);
                thread_parts[t][p].append(allocators[t],
                                          {val, static_cast<uint32_t>(i)});
            }
        });

        // Compute global offsets from per-thread counts
        Contest::platform::ArenaVector<size_t> global_offsets(*arena_);
        global_offsets.resize(num_partitions + 1);
        std::memset(global_offsets.data(), 0,
                    (num_partitions + 1) * sizeof(size_t));
        for (size_t p = 0; p < num_partitions; ++p) {
            for (size_t t = 0; t < num_threads; ++t) {
                global_offsets[p + 1] += thread_parts[t][p].total_count;
            }
            global_offsets[p + 1] += global_offsets[p];
        }

        size_t total = global_offsets[num_partitions];
        if (total == 0)
            return;
        keys_.resize(total);
        row_ids_.resize(total);

        // Build partitions in parallel
        const int nt = num_threads;
        Contest::platform::worker_pool().execute([&, nt](size_t t) {
            for (size_t p = t; p < num_partitions; p += nt) {
                build_partition(
                    thread_parts, p, slots_per_partition, global_offsets[p],
                    global_offsets[p + 1] - global_offsets[p], nt, t);
            }
        });
    }

    /**
     * @brief Build hash table from ColumnarTable Column.
     *
     * Handles dense (no NULLs) and sparse (with bitmap) pages.
     * Same radix-partitioned parallel strategy as build_intermediate().
     *
     * @param column Paged INT32 column. Page header: (n_rows, n_vals).
     * @param num_threads Thread count hint; falls back to single-threaded <16
     * pages.
     */
    void build_columnar(const Column &column, int num_threads = 4) {
        if (column.pages.empty())
            return;

        Contest::platform::ArenaVector<uint32_t> page_offsets(*arena_);
        page_offsets.reserve(column.pages.size());
        size_t total_rows = 0;
        for (const auto &page : column.pages) {
            page_offsets.push_back(static_cast<uint32_t>(total_rows));
            total_rows += *reinterpret_cast<const uint16_t *>(page->data);
        }
        if (total_rows == 0)
            return;

        num_threads = std::clamp(
            num_threads, 1, Contest::platform::worker_pool().thread_count());
        if (column.pages.size() < 16)
            num_threads = 1;

        const size_t num_pages = column.pages.size();
        const size_t num_slots = directory.size();
        const size_t num_partitions =
            compute_num_partitions(total_rows, num_threads);
        const int partition_bits = __builtin_ctzll(num_partitions);
        const size_t slots_per_partition = num_slots / num_partitions;

        std::vector<ChunkAllocator> allocators(num_threads);
        for (int t = 0; t < num_threads; ++t)
            allocators[t].set_arena(Contest::platform::get_arena(t));
        std::vector<std::vector<Partition>> thread_parts(num_threads);
        for (auto &tp : thread_parts)
            tp.resize(num_partitions);

        size_t batch = (num_pages + num_threads - 1) / num_threads;
        Contest::platform::worker_pool().execute([&, partition_bits](size_t t) {
            size_t pg_start = t * batch;
            size_t pg_end = std::min(pg_start + batch, num_pages);
            if (pg_start >= pg_end)
                return;

            const int shift = 64 - partition_bits;
            for (size_t pg = pg_start; pg < pg_end; ++pg) {
                const auto *page =
                    reinterpret_cast<const uint8_t *>(column.pages[pg]->data);
                uint16_t n_rows = *reinterpret_cast<const uint16_t *>(page);
                uint16_t n_vals = *reinterpret_cast<const uint16_t *>(page + 2);
                const int32_t *vals =
                    reinterpret_cast<const int32_t *>(page + 4);
                uint32_t base = page_offsets[pg];

                if (n_rows == n_vals) {
                    for (uint16_t r = 0; r < n_rows; ++r) {
                        int32_t val = vals[r];
                        uint64_t h = hash_key(val);
                        size_t p = (partition_bits == 0) ? 0 : (h >> shift);
                        thread_parts[t][p].append(allocators[t],
                                                  {val, base + r});
                    }
                } else {
                    const uint8_t *bitmap =
                        page + PAGE_SIZE - ((n_rows + 7) / 8);
                    uint16_t vi = 0;
                    for (uint16_t r = 0; r < n_rows; ++r) {
                        if (bitmap[r / 8] & (1u << (r % 8))) {
                            int32_t val = vals[vi++];
                            uint64_t h = hash_key(val);
                            size_t p = (partition_bits == 0) ? 0 : (h >> shift);
                            thread_parts[t][p].append(allocators[t],
                                                      {val, base + r});
                        }
                    }
                }
            }
        });

        Contest::platform::ArenaVector<size_t> global_offsets(*arena_);
        global_offsets.resize(num_partitions + 1);
        std::memset(global_offsets.data(), 0,
                    (num_partitions + 1) * sizeof(size_t));
        for (size_t p = 0; p < num_partitions; ++p) {
            for (int t = 0; t < num_threads; ++t) {
                global_offsets[p + 1] += thread_parts[t][p].total_count;
            }
            global_offsets[p + 1] += global_offsets[p];
        }

        size_t total = global_offsets[num_partitions];
        if (total == 0)
            return;
        keys_.resize(total);
        row_ids_.resize(total);

        const int nt = num_threads;
        Contest::platform::worker_pool().execute([&, nt](size_t t) {
            for (size_t p = t; p < num_partitions; p += nt) {
                build_partition(
                    thread_parts, p, slots_per_partition, global_offsets[p],
                    global_offsets[p + 1] - global_offsets[p], nt, t);
            }
        });
    }
};

/**
 * @file hashtable.h
 * @brief Unchained hash table with bloom filter and radix-partitioned build.
 *
 * Optimized for equi-joins on INT32 keys with parallel build and probe.
 * Keys and row_ids stored in contiguous arrays (not linked lists) for
 * cache-friendly sequential access during probe.
 *
 * **Directory entry layout (64 bits):**
 * - Bits 16-63: End offset in keys_/row_ids_ arrays
 * - Bits 0-15: Bloom filter tag (OR of all keys hashing to this slot)
 *
 * **Radix-partitioned parallel build:**
 * 1. Partition phase: Threads hash keys to thread-local partition buffers
 * 2. Scatter phase: Partitions processed in parallel, each thread handles
 *    disjoint partitions. Keys scattered to final positions in arrays.
 *
 * Partition count chosen so each fits in per-core LLC share, ensuring
 * working set stays cache-resident during scatter.
 *
 * @note Join keys are always INT32 (contest invariant).
 * @see hash_join.h for build/probe entry points.
 * @see bloom_tags.h for precomputed filter tags.
 */
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <data_model/intermediate.h>
#include <deque>
#include <join_execution/bloom_tags.h>
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
using Contest::platform::worker_pool;

/**
 * @brief High-performance hash table for parallel equi-joins.
 *
 * **Unchained design:** Keys and row_ids stored in separate contiguous arrays,
 * not linked lists. Directory entries point to ranges in these arrays,
 * enabling:
 * - Cache-friendly sequential access during probe
 * - SIMD-friendly memory layout
 * - No pointer chasing
 *
 * **Why unchained beats chained:** Chained hash tables suffer pointer-chasing
 * stalls (20-30 cycles per cache miss) and unpredictable memory access. This
 * design enables hardware prefetching and 4-8x faster probe due to sequential
 * reads. Build cost is similar (single pass + partition overhead), but probe
 * throughput dominates join performance on large tables.
 *
 * **Bloom filter acceleration:** Precomputed 16-bit tags enable early
 * rejection. If `(directory_entry & probe_tag) != probe_tag`, no matches exist
 * in that slot.
 *
 * **Hash function:** Hardware CRC32 (ARM/x86 intrinsics) with multiplicative
 * mixing for good distribution across directory slots.
 *
 * @note Build is O(n), probe is O(1) average per key.
 * @note Thread-safety: build_*() methods are parallel-safe via lock-free
 *       partitioning. Probe (find_indices) is read-only and lock-free.
 */
class UnchainedHashtable {
  public:
    /** @brief Key-rowid pair for hash table entries. */
    struct alignas(4) Tuple {
        int32_t key;     /**< Join key value. */
        uint32_t row_id; /**< Row index in source table. */
    };

    /** @brief L2-sized chunk for cache-friendly temporary storage during build.
     */
    static constexpr size_t CHUNK_SIZE = 4096;
    static constexpr size_t CHUNK_HEADER = 16;
    static constexpr size_t TUPLES_PER_CHUNK =
        (CHUNK_SIZE - CHUNK_HEADER) / sizeof(Tuple);

    /** @brief Linked chunk for partition buffers during radix-partitioned
     * build. */
    struct alignas(8) Chunk {
        Chunk *next;                  /**< Next chunk in partition chain. */
        size_t count;                 /**< Number of tuples in this chunk. */
        Tuple data[TUPLES_PER_CHUNK]; /**< Tuple storage. */
    };

    /** @brief Thread-local chunk allocator for lock-free partition building. */
    class ChunkAllocator {
        std::deque<Chunk> storage;

      public:
        Chunk *alloc() {
            storage.emplace_back();
            Chunk *c = &storage.back();
            c->next = nullptr;
            c->count = 0;
            return c;
        }
    };

    /**
     * @brief Partition of directory entries for lock-free parallel build.
     *
     * Contains a linked list of chunks, designed to fit in LLC.
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
    std::vector<uint64_t>
        directory; /**< Slot entries: (end_offset << 16) | bloom_tag. */
    std::vector<int32_t>
        keys_; /**< Contiguous key storage, indexed by directory. */
    std::vector<uint32_t>
        row_ids_; /**< Parallel row_id storage, same indexing. */
    int shift =
        0; /**< Bit shift for slot calculation: slot = hash >> (64-shift). */

    /**
     * @brief CRC32-based hash with multiplicative mixing.
     *
     * CRC32 chosen over MurmurHash3 or xxHash because it's hardware-accelerated
     * on both ARM (via __crc32w) and x86 (via _mm_crc32_u32), providing
     * single-cycle latency. Multiplicative constant 0x8648DBDB provides
     * avalanche effect, spreading bits across the full 64-bit range for
     * uniform slot distribution.
     *
     * @param key INT32 join key to hash.
     * @return 64-bit hash value (upper bits index directory slot).
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
     * @brief Returns precomputed bloom filter tag from hash.
     *
     * Uses upper 11 bits of hash (2048 entries) to index into BLOOM_TAGS.
     * Precomputed tags avoid runtime bit manipulation during probe, trading
     * 4KB of L1-resident data for faster membership tests.
     *
     * @param h 64-bit hash from hash_key().
     * @return 16-bit bloom tag (OR'd into directory entry during build).
     * @see bloom_tags.h for tag generation and false positive analysis.
     */
    static uint16_t bloom_tag(uint64_t h) noexcept {
        return BLOOM_TAGS[(h >> 32) & 0x7FF];
    }

    size_t slot_for(uint64_t h) const noexcept { return h >> (64 - shift); }

    /**
     * @brief Computes partition count to fit each partition in per-core LLC.
     *
     * Sizes partitions so each thread's working set fits in its LLC share
     * (typically L3 on Intel, L2 on Apple Silicon). Targets 50% of per-core
     * LLC to leave headroom for directory access and other data. Power-of-2
     * sizing enables fast modulo via bit shift (h >> (64 - log2(partitions))).
     *
     * Cache residency during scatter phase is critical because each partition
     * is processed independently - keeping it L3-resident avoids DRAM stalls
     * and maintains ~4-5 cycles/key throughput.
     *
     * @param tuple_count Total number of tuples to partition.
     * @param num_threads Thread count (ensures >= 1 partition per thread).
     * @return Power-of-2 partition count, capped at directory size.
     */
    size_t compute_num_partitions(size_t tuple_count, int num_threads) const {
        size_t per_core_cache = LAST_LEVEL_CACHE / worker_pool.thread_count();
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
     * Three-phase scatter algorithm:
     * 1. Count phase: Histogram keys per slot within partition (merges all
     *    thread chunks for this partition).
     * 2. Prefix sum: Compute final write offsets for each slot (ensures
     *    contiguous key placement).
     * 3. Scatter phase: Write keys/row_ids to final positions and OR bloom
     *    tags into directory.
     *
     * Partition-local processing enables lock-free parallelism - each thread
     * handles disjoint partitions, avoiding atomic ops or barriers during
     * scatter. This is faster than global scatter with locking or atomic CAS.
     *
     * @param thread_parts Per-thread partition buffers (thread_parts[t][p]).
     * @param p Partition index to build.
     * @param slots_per_partition Number of directory slots in this partition.
     * @param base_offset Global write offset for this partition's first key.
     * @param partition_size Total tuples in this partition (across all
     * threads).
     * @param num_threads Total thread count (for merging thread-local data).
     */
    void
    build_partition(const std::vector<std::vector<Partition>> &thread_parts,
                    size_t p, size_t slots_per_partition, size_t base_offset,
                    size_t partition_size, int num_threads) {

        const size_t slot_start = p * slots_per_partition;
        // Write offset for empty partitions
        if (partition_size == 0) {
            for (size_t s = 0; s < slots_per_partition; ++s) {
                directory[slot_start + s] = base_offset << 16;
            }
            return;
        }
        // Count keys per slot within partition
        std::vector<uint32_t> counts(slots_per_partition, 0);
        for (int t = 0; t < num_threads; ++t) {
            for (Chunk *c = thread_parts[t][p].head; c; c = c->next) {
                for (size_t i = 0; i < c->count; ++i) {
                    counts[slot_for(hash_key(c->data[i].key)) - slot_start]++;
                }
            }
        }
        // Prefix sum for write offsets
        std::vector<uint32_t> offsets(slots_per_partition);
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
     * Directory size is rounded up to power of 2 for fast modulo via bit shift.
     * Minimum size is 2048 slots.
     */
    explicit UnchainedHashtable(size_t build_size) {
        size_t pow2 = 2048;
        while (pow2 < build_size)
            pow2 <<= 1;
        directory.resize(pow2, 0);
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
     * @brief Find index range for keys matching the probe key.
     *
     * Returns [start, end) indices into keys_/row_ids_ arrays where
     * potential matches exist. Caller must verify actual key equality.
     *
     * @param key Probe key to search for.
     * @return Pair of (start, end) indices; (0,0) if bloom filter rejects.
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
     * Uses radix-partitioned parallel build when row count exceeds threshold.
     * Thread-local partition buffers avoid contention during partitioning,
     * then partitions are scattered in parallel to final positions.
     *
     * @param column Intermediate column containing INT32 join keys. Values
     *               accessed via column[i].value; row_ids are implicit indices.
     * @param num_threads Thread count hint (clamped to worker_pool size).
     *                    Falls back to single-threaded below threshold.
     */
    void build_intermediate(const mema::column_t &column, int num_threads = 4) {
        const size_t row_count = column.row_count();
        if (row_count == 0)
            return;

        /**
         * @brief Threshold for enabling parallel build.
         *
         * Below 10K rows, partitioning overhead (chunk allocation, prefix sums)
         * exceeds benefits of parallelism. Single-threaded build is faster due
         * to simpler code path and avoiding thread spawn costs. Measured on
         * 8-core M1 Pro: breakeven at ~8-10K rows, chosen conservatively.
         */
        static constexpr size_t PARALLEL_BUILD_THRESHOLD = 10000;
        num_threads = worker_pool.thread_count();
        if (row_count < PARALLEL_BUILD_THRESHOLD)
            num_threads = 1;

        const size_t num_slots = directory.size();
        const size_t num_partitions =
            compute_num_partitions(row_count, num_threads);
        const int partition_bits = __builtin_ctzll(num_partitions);
        const size_t slots_per_partition = num_slots / num_partitions;

        // Thread-local partitions for lock-free parallel partitioning
        std::vector<ChunkAllocator> allocators(num_threads);
        std::vector<std::vector<Partition>> thread_parts(num_threads);
        for (auto &tp : thread_parts)
            tp.resize(num_partitions);

        // Partition data by hash
        size_t batch = (row_count + num_threads - 1) / num_threads;
        worker_pool.execute([&, partition_bits](size_t t) {
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
        std::vector<size_t> global_offsets(num_partitions + 1, 0);
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
        worker_pool.execute([&, nt](size_t t) {
            for (size_t p = t; p < num_partitions; p += nt) {
                build_partition(thread_parts, p, slots_per_partition,
                                global_offsets[p],
                                global_offsets[p + 1] - global_offsets[p], nt);
            }
        });
    }

    /**
     * @brief Build hash table from ColumnarTable Column.
     *
     * Handles both dense pages (no NULLs) and sparse pages (with bitmap).
     * Uses same radix-partitioned parallel strategy as build_intermediate().
     *
     * @param column ColumnarTable column with paged INT32 data. Each page
     *               header contains (n_rows, n_vals) and optional NULL bitmap
     *               at end. Dense pages (n_rows == n_vals) use direct indexing;
     *               sparse pages require bitmap decoding.
     * @param num_threads Thread count hint; clamped to worker_pool size.
     *                    Falls back to single-threaded if <16 pages.
     */
    void build_columnar(const Column &column, int num_threads = 4) {
        if (column.pages.empty())
            return;

        std::vector<uint32_t> page_offsets;
        page_offsets.reserve(column.pages.size());
        size_t total_rows = 0;
        for (const auto &page : column.pages) {
            page_offsets.push_back(static_cast<uint32_t>(total_rows));
            total_rows += *reinterpret_cast<const uint16_t *>(page->data);
        }
        if (total_rows == 0)
            return;

        num_threads = std::clamp(num_threads, 1, worker_pool.thread_count());
        if (column.pages.size() < 16)
            num_threads = 1;

        const size_t num_pages = column.pages.size();
        const size_t num_slots = directory.size();
        const size_t num_partitions =
            compute_num_partitions(total_rows, num_threads);
        const int partition_bits = __builtin_ctzll(num_partitions);
        const size_t slots_per_partition = num_slots / num_partitions;

        std::vector<ChunkAllocator> allocators(num_threads);
        std::vector<std::vector<Partition>> thread_parts(num_threads);
        for (auto &tp : thread_parts)
            tp.resize(num_partitions);

        size_t batch = (num_pages + num_threads - 1) / num_threads;
        worker_pool.execute([&, partition_bits](size_t t) {
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

        std::vector<size_t> global_offsets(num_partitions + 1, 0);
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
        worker_pool.execute([&, nt](size_t t) {
            for (size_t p = t; p < num_partitions; p += nt) {
                build_partition(thread_parts, p, slots_per_partition,
                                global_offsets[p],
                                global_offsets[p + 1] - global_offsets[p], nt);
            }
        });
    }
};

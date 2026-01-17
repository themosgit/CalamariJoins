#pragma once
#include <algorithm>
#include <bloom_tags.h>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <intermediate.h>
#include <vector>
#include <worker_pool.h>

#if defined(__APPLE__) && defined(__aarch64__)
#include <hardware_darwin.h>
#elif defined(SPC__USE_BENCHMARKVM_HARDWARE)
#include <hardware_benchmarkvm.h>
#else
#include <hardware.h>
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_acle.h>
#endif

/* some systems do not have L3 cache ao L2 is set as LLC */
#if defined(SPC__LEVEL3_CACHE_SIZE) && SPC__LEVEL3_CACHE_SIZE > 0
static constexpr size_t LAST_LEVEL_CACHE = SPC__LEVEL3_CACHE_SIZE;
#else
static constexpr size_t LAST_LEVEL_CACHE = SPC__LEVEL2_CACHE_SIZE;
#endif

static constexpr size_t L2_CACHE = SPC__LEVEL2_CACHE_SIZE;
static constexpr size_t CACHE_LINE = SPC__LEVEL1_DCACHE_LINESIZE;

using Contest::worker_pool;

class UnchainedHashtable {
  public:
    struct alignas(4) Tuple {
        int32_t key;
        uint32_t row_id;
    };

    /**
     *
     *  Chunks are the smallest possible
     *  memory allocation unit, designed
     *  to fit in L2 cache.
     *
     **/
    static constexpr size_t CHUNK_SIZE = 4096;
    static constexpr size_t CHUNK_HEADER = 16;
    static constexpr size_t TUPLES_PER_CHUNK =
        (CHUNK_SIZE - CHUNK_HEADER) / sizeof(Tuple);
    struct alignas(8) Chunk {
        Chunk *next;
        size_t count;
        Tuple data[TUPLES_PER_CHUNK];
    };
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
     *
     *  A partition is a set of continuous directory
     *  entries of the hashtable used to enable lock-free
     *  parallel execution they consist of a linked list
     *  of chunks for data storage enabling optimized
     *  memory layout and allocations per thread.
     *
     *  Designed to fit in LLC.
     *
     **/
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
    std::vector<uint64_t> directory;
    std::vector<int32_t> keys_;
    std::vector<uint32_t> row_ids_;
    int shift = 0;

    /* crc32 hash function with large mix value */
    static uint64_t hash_key(int32_t key) noexcept {
        constexpr uint64_t k = 0x8648DBDB;
#if defined(__aarch64__)
        uint32_t crc = __crc32w(0, static_cast<uint32_t>(key));
#else
        uint32_t crc = _mm_crc32_u32(0, static_cast<uint32_t>(key));
#endif
        return crc * ((k << 32) + 1);
    }

    /* tags can be found at bloom_tags.h */
    static uint16_t bloom_tag(uint64_t h) noexcept {
        return BLOOM_TAGS[(h >> 32) & 0x7FF];
    }

    size_t slot_for(uint64_t h) const noexcept { return h >> (64 - shift); }

    /* compute number of partitions required */
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
     *
     * processes the same partition from
     * all threads follows the same workflow
     * as the single threaded build
     *
     **/
    void
    build_partition(const std::vector<std::vector<Partition>> &thread_parts,
                    size_t p, size_t slots_per_partition, size_t base_offset,
                    size_t partition_size, int num_threads) {

        const size_t slot_start = p * slots_per_partition;
        /* write offset to empty partitions */
        if (partition_size == 0) {
            for (size_t s = 0; s < slots_per_partition; ++s) {
                directory[slot_start + s] = base_offset << 16;
            }
            return;
        }
        /* hash all keys and find counts for partition slots */
        std::vector<uint32_t> counts(slots_per_partition, 0);
        for (int t = 0; t < num_threads; ++t) {
            for (Chunk *c = thread_parts[t][p].head; c; c = c->next) {
                for (size_t i = 0; i < c->count; ++i) {
                    counts[slot_for(hash_key(c->data[i].key)) - slot_start]++;
                }
            }
        }
        /* prefix sum of counts in partition */
        std::vector<uint32_t> offsets(slots_per_partition);
        uint32_t running = base_offset;
        for (size_t s = 0; s < slots_per_partition; ++s) {
            offsets[s] = running;
            running += counts[s];
            counts[s] = 0;
        }
        /* write values to correct slots update bloom filter */
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
        /* finalize directory write pointer and bloom filter */
        for (size_t s = 0; s < slots_per_partition; ++s) {
            uint64_t end = offsets[s] + counts[s];
            directory[slot_start + s] =
                (end << 16) | (directory[slot_start + s] & 0xFFFF);
        }
    }

  public:
    explicit UnchainedHashtable(size_t build_size) {
        size_t pow2 = 2048;
        while (pow2 < build_size)
            pow2 <<= 1;
        directory.resize(pow2, 0);
        shift = __builtin_ctzll(pow2);
    }

    size_t size() const noexcept { return keys_.size(); }
    bool empty() const noexcept { return keys_.empty(); }
    const int32_t *keys() const noexcept { return keys_.data(); }
    const uint32_t *row_ids() const noexcept { return row_ids_.data(); }

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

    void build_intermediate(const mema::column_t &column, int num_threads = 4) {
        const size_t row_count = column.row_count();
        if (row_count == 0)
            return;

        static constexpr size_t PARALLEL_BUILD_THRESHOLD = 10000;
        num_threads = worker_pool.thread_count();
        if (row_count < PARALLEL_BUILD_THRESHOLD)
            num_threads = 1;

        const size_t num_slots = directory.size();
        const size_t num_partitions =
            compute_num_partitions(row_count, num_threads);
        const int partition_bits = __builtin_ctzll(num_partitions);
        const size_t slots_per_partition = num_slots / num_partitions;

        /* all threads have a private instance of all partitions */
        std::vector<ChunkAllocator> allocators(num_threads);
        std::vector<std::vector<Partition>> thread_parts(num_threads);
        for (auto &tp : thread_parts)
            tp.resize(num_partitions);

        /* partitions data to every thread based on hash */
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

        /* compute offsets partition data from every thread */
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

        /* accumulates and builds partitions from all threads */
        const int nt = num_threads;
        worker_pool.execute([&, nt](size_t t) {
            for (size_t p = t; p < num_partitions; p += nt) {
                build_partition(thread_parts, p, slots_per_partition,
                                global_offsets[p],
                                global_offsets[p + 1] - global_offsets[p], nt);
            }
        });
    }

    /* same for ColumnarTable */
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

/**
 * @file match_collector.h
 * @brief Lock-free parallel match collection for joins.
 *
 * Collects (build_row_id, probe_row_id) pairs from parallel probe workers.
 * Uses chunk-chains for O(1) merge: each thread accumulates to thread-local
 * buffers, then chains are linked together without copying.
 *
 * **Ordering semantics:** Matches appear in thread-id order (thread 0, 1, 2,
 * ...), with each thread's matches in discovery order. This order is preserved
 * through materialization. The ordering is non-deterministic across runs but
 * semantically correct since SQL joins don't guarantee order.
 *
 * **Typical usage pattern:**
 * @code
 * MatchCollector collector(MatchCollectionMode::BOTH);
 * std::vector<ThreadLocalMatchBuffer> buffers =
 * create_thread_local_buffers(num_threads, mode);
 * worker_pool.execute([&](size_t t) {
 *     buffers[t].add_match(build_row, probe_row);
 * });
 * merge_local_collectors(buffers, collector);
 * collector.finalize_parallel();
 * for (uint32_t id : collector.get_left_range(0, collector.size())) {
 *     // process
 * }
 * @endcode
 *
 * @see ThreadLocalMatchBuffer for per-thread accumulation.
 * @see materialize.h and construct_intermediate.h for consumers.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <platform/worker_pool.h>
#include <sys/mman.h>
#include <vector>

/**
 * @namespace Contest::join
 * @brief Parallel hash join implementation for the SIGMOD contest.
 *
 * Key components in this file:
 * - MatchCollectionMode: Controls which row IDs to collect (memory
 * optimization)
 * - IndexChunk: Fixed-size chunk for lock-free linked-list match storage
 * - ThreadLocalMatchBuffer: Per-thread accumulator avoiding contention
 * - MatchCollector: Global aggregator with O(1) chain merge
 * - ChunkIterator/ChunkRange: Zero-copy iteration over match chains
 */
namespace Contest::join {

// Types from Contest::platform:: namespace
using Contest::platform::worker_pool;

/**
 * @brief Specifies which side's row IDs to collect during join.
 *
 * Memory optimization: when output only needs columns from one join side,
 * skip collecting the other side's IDs (saves 50% match storage).
 * Determined by join_setup.h:determine_collection_mode based on output_attrs.
 */
enum class MatchCollectionMode : uint8_t {
    BOTH = 0,      /**< Collect both left and right row IDs. */
    LEFT_ONLY = 1, /**< Only left (build) side IDs needed. */
    RIGHT_ONLY = 2 /**< Only right (probe) side IDs needed. */
};

/** @brief Capacity per IndexChunk; sized for L2 cache efficiency. */
static constexpr size_t MATCH_CHUNK_CAP = 8192;

/**
 * @brief Fixed-size chunk of row IDs forming a singly-linked list.
 *
 * Building block for lock-free match collection. New chunks are allocated
 * when current fills, linked via `next` pointer for O(1) merge of chains.
 */
struct IndexChunk {
    uint32_t ids[MATCH_CHUNK_CAP]; /**< Row ID storage. */
    uint32_t count = 0;            /**< Valid entries in ids[0..count-1]. */
    IndexChunk *next = nullptr; /**< Next chunk in chain (nullptr if tail). */
};

class MatchCollector;

/**
 * @brief Per-thread buffer for collecting join matches without contention.
 *
 * Each probe worker maintains its own buffer, appending matches via
 * add_match(). Maintains separate left/right chunk chains based on
 * MatchCollectionMode. After probe completes, chains are transferred to
 * MatchCollector via merge_local_collectors() - O(1) pointer linking with no
 * copying.
 *
 * @note Allocated chunks are owned by this buffer until merge, then ownership
 *       transfers to MatchCollector. Destructor only frees unrelinquished
 * chunks.
 */
class ThreadLocalMatchBuffer {
    friend void merge_local_collectors(std::vector<ThreadLocalMatchBuffer> &,
                                       MatchCollector &);
    friend class MatchCollector;

    IndexChunk *left_head = nullptr;
    IndexChunk *left_tail = nullptr;
    IndexChunk *right_head = nullptr;
    IndexChunk *right_tail = nullptr;
    MatchCollectionMode mode = MatchCollectionMode::BOTH;

  public:
    size_t total_count = 0;

    ThreadLocalMatchBuffer(MatchCollectionMode mode = MatchCollectionMode::BOTH)
        : mode(mode) {
        if (mode != MatchCollectionMode::RIGHT_ONLY) {
            left_tail = left_head = new IndexChunk();
        }
        if (mode != MatchCollectionMode::LEFT_ONLY) {
            right_tail = right_head = new IndexChunk();
        }
    }

    ThreadLocalMatchBuffer(ThreadLocalMatchBuffer &&other) noexcept
        : left_head(other.left_head), left_tail(other.left_tail),
          right_head(other.right_head), right_tail(other.right_tail),
          mode(other.mode), total_count(other.total_count) {
        other.left_head = other.left_tail = nullptr;
        other.right_head = other.right_tail = nullptr;
        other.total_count = 0;
    }

    ~ThreadLocalMatchBuffer() {
        while (left_head) {
            IndexChunk *temp = left_head;
            left_head = left_head->next;
            delete temp;
        }
        while (right_head) {
            IndexChunk *temp = right_head;
            right_head = right_head->next;
            delete temp;
        }
    }

    /**
     * @brief Records a match between build and probe row IDs.
     *
     * Hot path - inlined, branch-predicted for chunk-not-full case.
     * Allocates new chunks in pairs (BOTH mode) to keep left/right aligned.
     *
     * @param left Build side row ID
     * @param right Probe side row ID
     */
    inline void add_match(uint32_t left, uint32_t right) {
        if (mode == MatchCollectionMode::BOTH) {
            if (left_tail->count == MATCH_CHUNK_CAP) [[unlikely]] {
                IndexChunk *new_left = new IndexChunk();
                IndexChunk *new_right = new IndexChunk();
                left_tail->next = new_left;
                right_tail->next = new_right;
                left_tail = new_left;
                right_tail = new_right;
            }
            left_tail->ids[left_tail->count] = left;
            right_tail->ids[right_tail->count] = right;
            left_tail->count++;
            right_tail->count++;
        } else if (mode == MatchCollectionMode::LEFT_ONLY) {
            if (left_tail->count == MATCH_CHUNK_CAP) [[unlikely]] {
                IndexChunk *new_left = new IndexChunk();
                left_tail->next = new_left;
                left_tail = new_left;
            }
            left_tail->ids[left_tail->count] = left;
            left_tail->count++;
        } else { // RIGHT_ONLY
            if (right_tail->count == MATCH_CHUNK_CAP) [[unlikely]] {
                IndexChunk *new_right = new IndexChunk();
                right_tail->next = new_right;
                right_tail = new_right;
            }
            right_tail->ids[right_tail->count] = right;
            right_tail->count++;
        }
        total_count++;
    }
};

/**
 * @brief Global collector aggregating matches from all probe workers.
 *
 * Receives chunk chains from ThreadLocalMatchBuffer via merge_thread_buffer()
 * or merge_local_collectors(). Uses atomic counter for total_matches_count;
 * chain linking is single-threaded (called after parallel probe completes).
 *
 * After finalize_parallel(), provides ChunkRange iterators for materialization
 * to traverse matches. The all_chunks vector consolidates ownership for
 * cleanup.
 *
 * @see hash_join.h probe functions which populate this collector.
 * @see materialize.h which consumes get_left_range/get_right_range.
 */
class MatchCollector {
    IndexChunk *left_chain_head = nullptr;
    IndexChunk *left_chain_tail = nullptr;
    IndexChunk *right_chain_head = nullptr;
    IndexChunk *right_chain_tail = nullptr;

    std::atomic<size_t> total_matches_count{0};
    bool is_finalized = false;
    MatchCollectionMode collection_mode = MatchCollectionMode::BOTH;

    std::vector<IndexChunk *> all_chunks;

  public:
    /**
     * @brief Forward iterator over row IDs across chunk chains.
     *
     * Tracks current chunk, offset within chunk, and remaining count.
     * Seamlessly traverses chunk boundaries. Used by materialization to
     * iterate matches without copying to contiguous storage.
     */
    class ChunkIterator {
        IndexChunk *current_chunk;
        uint32_t offset;
        size_t remaining;

      public:
        ChunkIterator(IndexChunk *start, size_t count)
            : current_chunk(start), offset(0), remaining(count) {
            if (!current_chunk) {
                remaining = 0;
            }
        }

        inline uint32_t operator*() const { return current_chunk->ids[offset]; }

        inline ChunkIterator &operator++() {
            if (remaining == 0 || !current_chunk)
                return *this;

            offset++;
            remaining--;

            if (offset >= current_chunk->count) [[unlikely]] {
                if (current_chunk->next) {
                    current_chunk = current_chunk->next;
                    offset = 0;
                }
            }
            return *this;
        }

        inline bool operator!=(const ChunkIterator &other) const {
            return remaining != other.remaining;
        }

        /**
         * @brief Skip forward n elements without dereferencing.
         *
         * Efficiently advances iterator across chunk boundaries for range
         * slicing. Used by ChunkRange::begin() to position iterator at
         * start_offset, enabling parallel workers to process non-overlapping
         * match ranges.
         *
         * @param n Number of elements to skip forward
         */
        void advance(size_t n) {
            while (n > 0 && current_chunk) {
                size_t available = current_chunk->count - offset;
                if (n <= available) {
                    offset += n;
                    if (offset >= current_chunk->count && current_chunk->next) {
                        current_chunk = current_chunk->next;
                        offset = 0;
                    }
                    return;
                }
                n -= available;
                current_chunk = current_chunk->next;
                offset = 0;
            }
        }
    };

    /**
     * @brief Range adapter for iterating a slice of the chunk chain.
     *
     * Returned by get_left_range/get_right_range for parallel materialization.
     * Supports range-for loops: `for (uint32_t id :
     * collector.get_left_range(start, count))`.
     */
    class ChunkRange {
        IndexChunk *head;
        size_t start_offset;
        size_t count;

      public:
        ChunkRange(IndexChunk *h, size_t start, size_t cnt)
            : head(h), start_offset(start), count(cnt) {}

        ChunkIterator begin() const {
            ChunkIterator it(head, count);
            it.advance(start_offset);
            return it;
        }

        ChunkIterator end() const { return ChunkIterator(nullptr, 0); }
    };

    explicit MatchCollector(
        MatchCollectionMode mode = MatchCollectionMode::BOTH)
        : collection_mode(mode) {}
    ~MatchCollector() {
        if (!is_finalized) {
            while (left_chain_head) {
                IndexChunk *temp = left_chain_head;
                left_chain_head = left_chain_head->next;
                delete temp;
            }
            while (right_chain_head) {
                IndexChunk *temp = right_chain_head;
                right_chain_head = right_chain_head->next;
                delete temp;
            }
        } else {
            for (auto *chunk : all_chunks)
                delete chunk;
        }
    }

    void reserve(size_t) {}

    /**
     * @brief Optimized merge takes linked lists of chunks via pointer ops.
     *
     * @param left_head Head of left chunk chain (build side IDs)
     * @param left_tail Tail of left chunk chain
     * @param right_head Head of right chunk chain (probe side IDs)
     * @param right_tail Tail of right chunk chain
     * @param batch_count Total matches in the chains
     */
    void append_batch_chains(IndexChunk *left_head, IndexChunk *left_tail,
                             IndexChunk *right_head, IndexChunk *right_tail,
                             size_t batch_count) {

        if (collection_mode == MatchCollectionMode::LEFT_ONLY) {
            if (!left_head)
                return;
        } else if (collection_mode == MatchCollectionMode::RIGHT_ONLY) {
            if (!right_head)
                return;
        } else {
            if (!left_head || !right_head)
                return;
        }

        total_matches_count.fetch_add(batch_count, std::memory_order_relaxed);

        if (collection_mode != MatchCollectionMode::RIGHT_ONLY) {
            if (left_chain_tail) {
                left_chain_tail->next = left_head;
            } else {
                left_chain_head = left_head;
            }
            left_chain_tail = left_tail;
        }

        if (collection_mode != MatchCollectionMode::LEFT_ONLY) {
            if (right_chain_tail) {
                right_chain_tail->next = right_head;
            } else {
                right_chain_head = right_head;
            }
            right_chain_tail = right_tail;
        }
    }

    /**
     * @brief Merges a single thread's buffer; transfers chunk ownership.
     *
     * Ownership transfer protocol: After append_batch_chains() links the
     * chunks into this collector, pointers are nulled to prevent double-free.
     * The buffer's destructor will see nullptr and skip deletion, making this
     * collector the sole owner responsible for cleanup via all_chunks.
     *
     * @param tlb Thread-local buffer to merge and clear
     */
    void merge_thread_buffer(ThreadLocalMatchBuffer &tlb) {
        if (tlb.total_count == 0)
            return;
        append_batch_chains(tlb.left_head, tlb.left_tail, tlb.right_head,
                            tlb.right_tail, tlb.total_count);
        tlb.left_head = tlb.left_tail = nullptr;
        tlb.right_head = tlb.right_tail = nullptr;
        tlb.total_count = 0;
    }

    /**
     * @brief Consolidates chunk ownership for cleanup; call after all merges.
     *
     * Collects all chunk pointers into all_chunks for unified deletion.
     * Must be called before accessing matches via get_*_range().
     */
    void finalize_parallel() {
        if (is_finalized)
            return;

        size_t total = total_matches_count.load();
        if (total == 0) {
            is_finalized = true;
            return;
        }

        IndexChunk *curr = left_chain_head;
        while (curr) {
            all_chunks.push_back(curr);
            curr = curr->next;
        }

        curr = right_chain_head;
        while (curr) {
            all_chunks.push_back(curr);
            curr = curr->next;
        }

        is_finalized = true;
    }

    /** @brief Ensures finalize_parallel() has been called. */
    void ensure_finalized() {
        if (!is_finalized)
            finalize_parallel();
    }

    /** @brief Returns total match count across all merged buffers. */
    inline size_t size() const { return total_matches_count; }

    /**
     * @brief Returns range for iterating left (build) row IDs.
     *
     * @param start Offset to begin iteration
     * @param count Number of elements to iterate
     * @return ChunkRange for range-based for loops
     */
    ChunkRange get_left_range(size_t start, size_t count) const {
        return ChunkRange(left_chain_head, start, count);
    }

    /**
     * @brief Returns range for iterating right (probe) row IDs.
     *
     * @param start Offset to begin iteration
     * @param count Number of elements to iterate
     * @return ChunkRange for range-based for loops
     */
    ChunkRange get_right_range(size_t start, size_t count) const {
        return ChunkRange(right_chain_head, start, count);
    }
};

/**
 * @brief Creates thread-local match buffers for parallel join processing.
 *
 * @param thread_count Number of worker threads
 * @param mode Collection mode (BOTH, LEFT_ONLY, or RIGHT_ONLY)
 * @return Vector of initialized buffers, one per thread
 */
inline std::vector<ThreadLocalMatchBuffer>
create_thread_local_buffers(size_t thread_count, MatchCollectionMode mode) {
    std::vector<ThreadLocalMatchBuffer> buffers;
    buffers.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        buffers.emplace_back(mode);
    }
    return buffers;
}

/**
 * @brief Merges all thread-local buffers into the global collector.
 *
 * Performs single-pass batch merge by linking chains from all non-empty
 * buffers, then submits consolidated chains to global_collector via one
 * append_batch_chains() call. This reduces atomic operations compared to
 * merging buffers individually.
 *
 * @param local_buffers Thread-local buffers to merge (will be cleared)
 * @param global_collector Destination collector receiving all matches
 */
inline void
merge_local_collectors(std::vector<ThreadLocalMatchBuffer> &local_buffers,
                       MatchCollector &global_collector) {

    IndexChunk *left_batch_head = nullptr;
    IndexChunk *left_batch_tail = nullptr;
    IndexChunk *right_batch_head = nullptr;
    IndexChunk *right_batch_tail = nullptr;
    size_t batch_total = 0;
    bool first_buffer = true;

    for (auto &buf : local_buffers) {
        if (buf.total_count == 0)
            continue;

        batch_total += buf.total_count;

        if (first_buffer) {
            first_buffer = false;
            if (buf.mode != MatchCollectionMode::RIGHT_ONLY) {
                left_batch_head = buf.left_head;
                left_batch_tail = buf.left_tail;
            }
            if (buf.mode != MatchCollectionMode::LEFT_ONLY) {
                right_batch_head = buf.right_head;
                right_batch_tail = buf.right_tail;
            }
        } else {
            if (buf.mode != MatchCollectionMode::RIGHT_ONLY) {
                if (left_batch_tail) {
                    left_batch_tail->next = buf.left_head;
                    left_batch_tail = buf.left_tail;
                } else {
                    left_batch_head = buf.left_head;
                    left_batch_tail = buf.left_tail;
                }
            }
            if (buf.mode != MatchCollectionMode::LEFT_ONLY) {
                if (right_batch_tail) {
                    right_batch_tail->next = buf.right_head;
                    right_batch_tail = buf.right_tail;
                } else {
                    right_batch_head = buf.right_head;
                    right_batch_tail = buf.right_tail;
                }
            }
        }
        buf.left_head = buf.left_tail = nullptr;
        buf.right_head = buf.right_tail = nullptr;
        buf.total_count = 0;
    }

    global_collector.append_batch_chains(left_batch_head, left_batch_tail,
                                         right_batch_head, right_batch_tail,
                                         batch_total);
}

} // namespace Contest::join

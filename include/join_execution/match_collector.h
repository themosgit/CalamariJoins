/**
 * @file match_collector.h
 * @brief Lock-free parallel match collection for joins.
 *
 * Collects (build_row_id, probe_row_id) pairs using chunk-chains for O(1)
 * merge. Thread-local buffers avoid contention; chains linked without copying.
 *
 * @see ThreadLocalMatchBuffer, materialize.h, construct_intermediate.h
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <platform/arena.h>
#include <platform/worker_pool.h>
#include <vector>

/** @namespace Contest::join @brief Parallel hash join implementation. */
namespace Contest::join {

using Contest::platform::worker_pool;

/** @brief Specifies which side's row IDs to collect. Saves 50% when one side
 * unused. */
enum class MatchCollectionMode : uint8_t {
    BOTH = 0,      /**< Collect both left and right row IDs. */
    LEFT_ONLY = 1, /**< Only left (build) side IDs needed. */
    RIGHT_ONLY = 2 /**< Only right (probe) side IDs needed. */
};

/** @brief Capacity per IndexChunk; sized to fit in INDEX_CHUNK arena region. */
static constexpr size_t MATCH_CHUNK_CAP = 8184;

/** @brief Fixed-size chunk of row IDs forming a singly-linked list. */
struct IndexChunk {
    uint32_t ids[MATCH_CHUNK_CAP]; /**< Row ID storage. */
    uint32_t count = 0;            /**< Valid entries in ids[0..count-1]. */
    IndexChunk *next = nullptr; /**< Next chunk in chain (nullptr if tail). */
};

static_assert(sizeof(IndexChunk) <=
                  Contest::platform::ChunkSize<
                      Contest::platform::ChunkType::INDEX_CHUNK>::value,
              "IndexChunk too large for INDEX_CHUNK region");

class MatchCollector;

/**
 * @brief Per-thread buffer for collecting join matches without contention.
 *
 * Maintains separate left/right chunk chains. After probe, chains transfer to
 * MatchCollector via O(1) pointer linking. Uses arena allocation for chunks.
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
    Contest::platform::ThreadArena *arena_ = nullptr;

    /** @brief Allocate IndexChunk from arena. */
    IndexChunk *alloc_chunk() {
        void *ptr =
            arena_->alloc_chunk<Contest::platform::ChunkType::INDEX_CHUNK>();
        IndexChunk *c = static_cast<IndexChunk *>(ptr);
        c->count = 0;
        c->next = nullptr;
        return c;
    }

  public:
    size_t total_count = 0;

    ThreadLocalMatchBuffer(Contest::platform::ThreadArena &arena,
                           MatchCollectionMode m = MatchCollectionMode::BOTH)
        : mode(m), arena_(&arena) {
        if (mode != MatchCollectionMode::RIGHT_ONLY) {
            left_tail = left_head = alloc_chunk();
        }
        if (mode != MatchCollectionMode::LEFT_ONLY) {
            right_tail = right_head = alloc_chunk();
        }
    }

    ThreadLocalMatchBuffer(ThreadLocalMatchBuffer &&other) noexcept
        : left_head(other.left_head), left_tail(other.left_tail),
          right_head(other.right_head), right_tail(other.right_tail),
          mode(other.mode), arena_(other.arena_),
          total_count(other.total_count) {
        other.left_head = other.left_tail = nullptr;
        other.right_head = other.right_tail = nullptr;
        other.total_count = 0;
    }

    // No destructor needed - arena manages memory

    /** @brief Records a match. Allocates new chunks in pairs for BOTH mode. */
    inline void add_match(uint32_t left, uint32_t right) {
        if (mode == MatchCollectionMode::BOTH) {
            if (left_tail->count == MATCH_CHUNK_CAP) [[unlikely]] {
                IndexChunk *new_left = alloc_chunk();
                IndexChunk *new_right = alloc_chunk();
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
                IndexChunk *new_left = alloc_chunk();
                left_tail->next = new_left;
                left_tail = new_left;
            }
            left_tail->ids[left_tail->count] = left;
            left_tail->count++;
        } else { // RIGHT_ONLY
            if (right_tail->count == MATCH_CHUNK_CAP) [[unlikely]] {
                IndexChunk *new_right = alloc_chunk();
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
 * Chain linking is single-threaded after probe completes. After
 * finalize_parallel(), provides ChunkRange iterators for materialization.
 *
 * @see hash_join.h, materialize.h
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
    /** @brief Forward iterator over row IDs across chunk chains. */
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

        /** @brief Skip forward n elements across chunk boundaries. */
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

    /** @brief Range adapter for iterating a slice of the chunk chain. */
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

    // No destructor needed - arena manages memory

    void reserve(size_t) {}

    /** @brief Merges chunk chains via pointer ops (O(1)). */
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

    /** @brief Merges a thread's buffer; transfers chunk ownership. */
    void merge_thread_buffer(ThreadLocalMatchBuffer &tlb) {
        if (tlb.total_count == 0)
            return;
        append_batch_chains(tlb.left_head, tlb.left_tail, tlb.right_head,
                            tlb.right_tail, tlb.total_count);
        tlb.left_head = tlb.left_tail = nullptr;
        tlb.right_head = tlb.right_tail = nullptr;
        tlb.total_count = 0;
    }

    /** @brief Consolidates chunk ownership; call after all merges. */
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

    /** @brief Returns range for iterating left (build) row IDs. */
    ChunkRange get_left_range(size_t start, size_t count) const {
        return ChunkRange(left_chain_head, start, count);
    }

    /** @brief Returns range for iterating right (probe) row IDs. */
    ChunkRange get_right_range(size_t start, size_t count) const {
        return ChunkRange(right_chain_head, start, count);
    }
};

/** @brief Creates thread-local match buffers, one per thread. */
inline std::vector<ThreadLocalMatchBuffer>
create_thread_local_buffers(size_t thread_count, MatchCollectionMode mode) {
    std::vector<ThreadLocalMatchBuffer> buffers;
    buffers.reserve(thread_count);
    for (size_t i = 0; i < thread_count; ++i) {
        buffers.emplace_back(Contest::platform::get_arena(i), mode);
    }
    return buffers;
}

/** @brief Merges all thread-local buffers into global collector via batch
 * chain linking. */
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

/**
 * @file match_collector.h
 * @brief Lock-free parallel match collection for joins.
 *
 * Collects (build_row_id, probe_row_id) pairs using chunk-chains for O(1)
 * merge. Thread-local buffers avoid contention; chains linked without copying.
 *
 * Template parameter Mode enables compile-time specialization for different
 * collection modes, eliminating runtime branching in hot loops.
 *
 * @see ThreadLocalMatchBuffer, materialize.h, construct_intermediate.h
 */
#pragma once

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

/**
 * @brief Per-thread buffer for collecting join matches without contention.
 *
 * Maintains separate left/right chunk chains based on Mode. Uses arena
 * allocation for chunks. After probe, buffers are iterated directly without
 * merging.
 *
 * @tparam Mode Collection mode (BOTH, LEFT_ONLY, RIGHT_ONLY). Determines which
 *              chains are allocated and which code path is used in add_match().
 *              Using if constexpr eliminates runtime branching.
 */
template <MatchCollectionMode Mode = MatchCollectionMode::BOTH>
class ThreadLocalMatchBuffer {

    IndexChunk *left_head = nullptr;
    IndexChunk *left_tail = nullptr;
    IndexChunk *right_head = nullptr;
    IndexChunk *right_tail = nullptr;
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

    ThreadLocalMatchBuffer() noexcept = default;

    explicit ThreadLocalMatchBuffer(Contest::platform::ThreadArena &arena)
        : arena_(&arena) {
        if constexpr (Mode != MatchCollectionMode::RIGHT_ONLY) {
            left_tail = left_head = alloc_chunk();
        }
        if constexpr (Mode != MatchCollectionMode::LEFT_ONLY) {
            right_tail = right_head = alloc_chunk();
        }
    }

    ThreadLocalMatchBuffer(ThreadLocalMatchBuffer &&other) noexcept
        : left_head(other.left_head), left_tail(other.left_tail),
          right_head(other.right_head), right_tail(other.right_tail),
          arena_(other.arena_), total_count(other.total_count) {
        other.left_head = other.left_tail = nullptr;
        other.right_head = other.right_tail = nullptr;
        other.total_count = 0;
    }

    ThreadLocalMatchBuffer &operator=(ThreadLocalMatchBuffer &&other) noexcept {
        if (this != &other) {
            left_head = other.left_head;
            left_tail = other.left_tail;
            right_head = other.right_head;
            right_tail = other.right_tail;
            arena_ = other.arena_;
            total_count = other.total_count;
            other.left_head = other.left_tail = nullptr;
            other.right_head = other.right_tail = nullptr;
            other.total_count = 0;
        }
        return *this;
    }

    // No destructor needed - arena manages memory

    /** @brief Forward iterator over row IDs in a thread-local chunk chain. */
    class ChainIterator {
        IndexChunk *current_chunk;
        uint32_t offset;
        size_t remaining;

      public:
        ChainIterator(IndexChunk *chunk, size_t count)
            : current_chunk(chunk), offset(0), remaining(count) {
            if (!current_chunk)
                remaining = 0;
        }

        inline uint32_t operator*() const { return current_chunk->ids[offset]; }

        inline ChainIterator &operator++() {
            if (remaining == 0 || !current_chunk)
                return *this;
            offset++;
            remaining--;
            if (offset >= current_chunk->count && current_chunk->next) {
                current_chunk = current_chunk->next;
                offset = 0;
            }
            return *this;
        }

        inline bool operator!=(const ChainIterator &other) const {
            return remaining != other.remaining;
        }
    };

    /** @brief Range adapter for iterating a thread-local chain. */
    class ChainRange {
        IndexChunk *head;
        size_t count;

      public:
        ChainRange(IndexChunk *h, size_t c) : head(h), count(c) {}
        ChainIterator begin() const { return ChainIterator(head, count); }
        ChainIterator end() const { return ChainIterator(nullptr, 0); }
    };

    /** @brief Returns range for iterating left (build) row IDs. */
    inline ChainRange left_range() const {
        return ChainRange(left_head, total_count);
    }

    /** @brief Returns range for iterating right (probe) row IDs. */
    inline ChainRange right_range() const {
        return ChainRange(right_head, total_count);
    }

    /** @brief Returns match count in this buffer. */
    size_t count() const { return total_count; }

    /**
     * @brief Records a match. Allocates new chunks as needed.
     *
     * Uses if constexpr for zero-overhead mode selection at compile time.
     * Each mode specialization only includes code for the chains it uses.
     */
    inline void add_match(uint32_t left, uint32_t right) {
        if constexpr (Mode == MatchCollectionMode::BOTH) {
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
        } else if constexpr (Mode == MatchCollectionMode::LEFT_ONLY) {
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

} // namespace Contest::join

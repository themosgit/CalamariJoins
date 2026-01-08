#pragma once

#include <vector>
#include <cstdint>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <worker_pool.h>
#include <sys/mman.h>

namespace Contest {

enum class MatchCollectionMode : uint8_t {
    BOTH = 0,
    LEFT_ONLY = 1,
    RIGHT_ONLY = 2
};

static constexpr size_t MATCH_CHUNK_CAP = 8192;

struct IndexChunk {
    uint32_t ids[MATCH_CHUNK_CAP];
    uint32_t count = 0;
    IndexChunk* next = nullptr;
};

class MatchCollector;

class ThreadLocalMatchBuffer {
    friend void merge_local_collectors(std::vector<ThreadLocalMatchBuffer>&, MatchCollector&);
    friend class MatchCollector;

    IndexChunk* left_head = nullptr;
    IndexChunk* left_tail = nullptr;
    IndexChunk* right_head = nullptr;
    IndexChunk* right_tail = nullptr;
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

    ThreadLocalMatchBuffer(ThreadLocalMatchBuffer&& other) noexcept
        : left_head(other.left_head), left_tail(other.left_tail),
          right_head(other.right_head), right_tail(other.right_tail),
          total_count(other.total_count) {
        other.left_head = other.left_tail = nullptr;
        other.right_head = other.right_tail = nullptr;
        other.total_count = 0;
    }

    ~ThreadLocalMatchBuffer() {
        while (left_head) {
            IndexChunk* temp = left_head;
            left_head = left_head->next;
            delete temp;
        }
        while (right_head) {
            IndexChunk* temp = right_head;
            right_head = right_head->next;
            delete temp;
        }
    }

    inline void add_match(uint32_t left, uint32_t right) {
        if (mode == MatchCollectionMode::BOTH) {
            if (left_tail->count == MATCH_CHUNK_CAP) [[unlikely]] {
                IndexChunk* new_left = new IndexChunk();
                IndexChunk* new_right = new IndexChunk();
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
                IndexChunk* new_left = new IndexChunk();
                left_tail->next = new_left;
                left_tail = new_left;
            }
            left_tail->ids[left_tail->count] = left;
            left_tail->count++;
        } else { // RIGHT_ONLY
            if (right_tail->count == MATCH_CHUNK_CAP) [[unlikely]] {
                IndexChunk* new_right = new IndexChunk();
                right_tail->next = new_right;
                right_tail = new_right;
            }
            right_tail->ids[right_tail->count] = right;
            right_tail->count++;
        }
        total_count++;
    }
};

class MatchCollector {
    IndexChunk* left_chain_head = nullptr;
    IndexChunk* left_chain_tail = nullptr;
    IndexChunk* right_chain_head = nullptr;
    IndexChunk* right_chain_tail = nullptr;

    std::atomic<size_t> total_matches_count{0};
    bool is_finalized = false;
    MatchCollectionMode collection_mode = MatchCollectionMode::BOTH;

    std::vector<IndexChunk*> all_chunks;

public:
    class ChunkIterator {
        IndexChunk* current_chunk;
        uint32_t offset;
        size_t remaining;

    public:
        ChunkIterator(IndexChunk* start, size_t count)
            : current_chunk(start), offset(0), remaining(count) {
            if (!current_chunk) {
                remaining = 0;
            }
        }

        inline uint32_t operator*() const {
            return current_chunk->ids[offset];
        }

        inline ChunkIterator& operator++() {
            if (remaining == 0 || !current_chunk) return *this;

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

        inline bool operator!=(const ChunkIterator& other) const {
            return remaining != other.remaining;
        }

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

    class ChunkRange {
        IndexChunk* head;
        size_t start_offset;
        size_t count;

    public:
        ChunkRange(IndexChunk* h, size_t start, size_t cnt)
            : head(h), start_offset(start), count(cnt) {}

        ChunkIterator begin() const {
            ChunkIterator it(head, count);
            it.advance(start_offset);
            return it;
        }

        ChunkIterator end() const {
            return ChunkIterator(nullptr, 0);
        }
    };

    explicit MatchCollector(MatchCollectionMode mode = MatchCollectionMode::BOTH)
        : collection_mode(mode) {}
    ~MatchCollector() {
        if (!is_finalized) {
            while (left_chain_head) {
                IndexChunk* temp = left_chain_head;
                left_chain_head = left_chain_head->next;
                delete temp;
            }
            while (right_chain_head) {
                IndexChunk* temp = right_chain_head;
                right_chain_head = right_chain_head->next;
                delete temp;
            }
        } else {
            for (auto* chunk : all_chunks) delete chunk;
        }
    }

    void reserve(size_t) {}

    /**
     *
     *  optimized merge takes linked lists of chunks
     *  and does simple pointer ops
     *
     **/
    void append_batch_chains(IndexChunk* left_head, IndexChunk* left_tail,
                             IndexChunk* right_head, IndexChunk* right_tail,
                             size_t batch_count) {

        if (collection_mode == MatchCollectionMode::LEFT_ONLY) {
            if (!left_head) return;
        } else if (collection_mode == MatchCollectionMode::RIGHT_ONLY) {
            if (!right_head) return;
        } else {
            if (!left_head || !right_head) return;
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

    void merge_thread_buffer(ThreadLocalMatchBuffer& tlb) {
        if (tlb.total_count == 0) return;
        append_batch_chains(tlb.left_head, tlb.left_tail,
                           tlb.right_head, tlb.right_tail,
                           tlb.total_count);
        tlb.left_head = tlb.left_tail = nullptr;
        tlb.right_head = tlb.right_tail = nullptr;
        tlb.total_count = 0;
    }

    void finalize_parallel() {
        if (is_finalized) return;

        size_t total = total_matches_count.load();
        if (total == 0) {
            is_finalized = true;
            return;
        }

        IndexChunk* curr = left_chain_head;
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

    void ensure_finalized() {
        if (!is_finalized) finalize_parallel();
    }

    inline size_t size() const { return total_matches_count; }

    ChunkRange get_left_range(size_t start, size_t count) const {
        return ChunkRange(left_chain_head, start, count);
    }

    ChunkRange get_right_range(size_t start, size_t count) const {
        return ChunkRange(right_chain_head, start, count);
    }
};

/**
 *
 *  merges all thread-local buffers into the global collector
 *  links all local buffers into separate left/right chains
 *  then pushes both chains with simple pointer ops
 *
 **/
inline void merge_local_collectors(
    std::vector<ThreadLocalMatchBuffer>& local_buffers,
    MatchCollector& global_collector) {

    IndexChunk* left_batch_head = nullptr;
    IndexChunk* left_batch_tail = nullptr;
    IndexChunk* right_batch_head = nullptr;
    IndexChunk* right_batch_tail = nullptr;
    size_t batch_total = 0;
    bool first_buffer = true;

    for (auto& buf : local_buffers) {
        if (buf.total_count == 0) continue;

        batch_total += buf.total_count;

        if (first_buffer) {
            first_buffer = false;
            if (buf.mode !=  MatchCollectionMode::RIGHT_ONLY) {
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

} // namespace Contest

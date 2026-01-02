#pragma once

#include <vector>
#include <cstdint>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <worker_pool.h>
#include <sys/mman.h>

namespace Contest {

static constexpr size_t MATCH_CHUNK_CAP = 8192; 

struct MatchChunk {
    uint64_t data[MATCH_CHUNK_CAP];
    uint32_t count = 0;
    MatchChunk* next = nullptr;
};

class MatchCollector;

class ThreadLocalMatchBuffer {
    friend void merge_local_collectors(std::vector<ThreadLocalMatchBuffer>&, MatchCollector&);
    friend class MatchCollector;

    MatchChunk* head = nullptr;
    MatchChunk* tail = nullptr;
    
public:
    size_t total_count = 0;

    ThreadLocalMatchBuffer() {
        allocate_new_chunk();
    }

    ThreadLocalMatchBuffer(ThreadLocalMatchBuffer&& other) noexcept
        : head(other.head), tail(other.tail), total_count(other.total_count) {
        other.head = nullptr;
        other.tail = nullptr;
        other.total_count = 0;
    }

    ~ThreadLocalMatchBuffer() {
        while (head) {
            MatchChunk* temp = head;
            head = head->next;
            delete temp;
        }
    }

    inline void add_match(uint32_t left, uint32_t right) {
        if (tail->count == MATCH_CHUNK_CAP) [[unlikely]] {
            allocate_new_chunk();
        }
        
        uint64_t packed = static_cast<uint64_t>(left) | (static_cast<uint64_t>(right) << 32);
        tail->data[tail->count++] = packed;
        total_count++;
    }

private:
    void allocate_new_chunk() {
        MatchChunk* new_chunk = new MatchChunk();
        if (tail) tail->next = new_chunk;
        else head = new_chunk;
        tail = new_chunk;
    }
};

class MatchCollector {
    std::vector<MatchChunk*> global_chunks;
    std::vector<size_t> chunk_start_offsets;
    
    MatchChunk* unified_list_head = nullptr;
    MatchChunk* unified_list_tail = nullptr;

    std::atomic<size_t> total_matches_count{0};
    bool is_finalized = false;

public:
    MatchCollector() = default;
    ~MatchCollector() {
        if (!is_finalized) {
            while(unified_list_head) {
                MatchChunk* temp = unified_list_head;
                unified_list_head = unified_list_head->next;
                delete temp;
            }
        } else {
            for (auto* chunk : global_chunks) delete chunk;
        }
    }

    void reserve(size_t) {}

    /**
     *
     *  optimized merge takes a linked list of chunks 
     *  and does a simple pointer op
     *
     **/
    void append_batch_chain(MatchChunk* chain_head, MatchChunk* chain_tail, size_t batch_count) {
        if (!chain_head) return;

        total_matches_count.fetch_add(batch_count, std::memory_order_relaxed);
        if (unified_list_tail) {
            unified_list_tail->next = chain_head;
        } else {
            unified_list_head = chain_head;
        }
        unified_list_tail = chain_tail;
    }

    // Deprecated single-merge (kept for safety, but optimized code uses append_batch_chain)
    void merge_thread_buffer(ThreadLocalMatchBuffer& tlb) {
        if (tlb.total_count == 0) return;
        append_batch_chain(tlb.head, tlb.tail, tlb.total_count);
        tlb.head = nullptr; tlb.tail = nullptr; tlb.total_count = 0;
    }

    void finalize_parallel() {
        if (is_finalized) return;

        size_t total = total_matches_count.load();
        if (total == 0) { is_finalized = true; return; }

        size_t estimated_chunks = total / MATCH_CHUNK_CAP + 16;
        global_chunks.reserve(estimated_chunks);
        chunk_start_offsets.reserve(estimated_chunks + 1);

        size_t current_offset = 0;
        MatchChunk* curr = unified_list_head;

        while (curr) {
            global_chunks.push_back(curr);
            chunk_start_offsets.push_back(current_offset);
            current_offset += curr->count;
            
            MatchChunk* next = curr->next;
            curr->next = nullptr; 
            curr = next;
        }
        chunk_start_offsets.push_back(current_offset);
        
        unified_list_head = nullptr;
        unified_list_tail = nullptr;
        is_finalized = true;
    }

    inline size_t size() const { return total_matches_count; }

    class Stream {
        const std::vector<MatchChunk*>& chunks;
        const std::vector<size_t>& starts;
        size_t chunk_idx;
        size_t idx_in_chunk;
        MatchChunk* current_chunk_ptr;

    public:
        Stream(const MatchCollector& mc, size_t global_start) 
            : chunks(mc.global_chunks), starts(mc.chunk_start_offsets) {
            if (global_start >= mc.size()) {
                chunk_idx = chunks.size();
                current_chunk_ptr = nullptr; 
                return;
            }
            auto it = std::upper_bound(starts.begin(), starts.end(), global_start);
            chunk_idx = std::distance(starts.begin(), it) - 1;
            idx_in_chunk = global_start - starts[chunk_idx];
            current_chunk_ptr = chunks[chunk_idx];
        }

        inline uint64_t next() {
            if (idx_in_chunk < current_chunk_ptr->count) [[likely]] {
                return current_chunk_ptr->data[idx_in_chunk++];
            }
            return next_chunk();
        }

    private:
        uint64_t next_chunk() {
            chunk_idx++;
            if (chunk_idx >= chunks.size()) return 0;
            current_chunk_ptr = chunks[chunk_idx];
            idx_in_chunk = 1;
            return current_chunk_ptr->data[0];
        }
    };

    Stream get_stream(size_t start_offset) const {
        return Stream(*this, start_offset);
    }
    
    void ensure_finalized() {
        if(!is_finalized) finalize_parallel();
    }
};

/**
 *
 *  links all local buffers into one 
 *  then pushes that single list to the collector
 *
 **/
inline void merge_local_collectors(
    std::vector<ThreadLocalMatchBuffer>& local_buffers,
    MatchCollector& global_collector) {
    
    MatchChunk* batch_head = nullptr;
    MatchChunk* batch_tail = nullptr;
    size_t batch_total = 0;

    for(auto& buf : local_buffers) {
        if (buf.total_count == 0) continue;

        batch_total += buf.total_count;

        if (!batch_head) {
            batch_head = buf.head;
            batch_tail = buf.tail;
        } else {
            batch_tail->next = buf.head;
            batch_tail = buf.tail;
        }

        buf.head = nullptr;
        buf.tail = nullptr;
        buf.total_count = 0;
    }

    global_collector.append_batch_chain(batch_head, batch_tail, batch_total);
}

} // namespace Contest

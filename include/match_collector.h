#pragma once

#include <vector>
#include <cstdint>
#include <atomic>
#include <cstring>
#include <worker_pool.h>
#include <sys/mman.h>


/**
 *
 * this is a work in progress match chunk size should align
 * with page size on order to partition work to worker threads
 * more efficiently also merge parallel still makes copies this will change
 *
 **/ 

namespace Contest {

static constexpr size_t MATCH_CHUNK_SIZE = 1024 * 4; // (32KB)

struct alignas(64) MatchBlock {
    uint64_t matches[MATCH_CHUNK_SIZE];
    uint32_t count = 0;
    MatchBlock* next = nullptr;

    inline bool is_full() const { return count == MATCH_CHUNK_SIZE; }
    
    inline void add(uint64_t val) {
        matches[count++] = val;
    }
};

class MatchCollector;

/**
 *
 *  Thread-local buffer that accumulates matches into blocks.
 *  Prevents false sharing and lock contention during the probe phase.
 *
 **/
class ThreadLocalMatchBuffer {
    friend class MatchCollector;
    MatchBlock* head = nullptr;
    MatchBlock* tail = nullptr;
    MatchBlock* current = nullptr;
    size_t total_count = 0;
    std::vector<MatchBlock*> free_blocks;

public:
    ThreadLocalMatchBuffer() = default;
    ThreadLocalMatchBuffer(ThreadLocalMatchBuffer&& other) noexcept 
        : head(other.head), tail(other.tail), current(other.current), 
          total_count(other.total_count) {
        other.head = other.tail = other.current = nullptr;
        other.total_count = 0;
    }

    ~ThreadLocalMatchBuffer() {
        while (head) {
            MatchBlock* next = head->next;
            delete head;
            head = next;
        }
    }

    inline void add_match(uint32_t left, uint32_t right) {
        if (!current || current->is_full()) {
            allocate_new_block();
        }
        uint64_t packed = static_cast<uint64_t>(left) | (static_cast<uint64_t>(right) << 32);
        current->add(packed);
        total_count++;
    }

private:
    void allocate_new_block() {
        MatchBlock* block = new MatchBlock();
        if (tail) {
            tail->next = block;
            tail = block;
        } else {
            head = tail = block;
        }
        current = block;
    }
};

class MatchCollector {
    struct ThreadChain {
        MatchBlock* head;
        size_t count;
    };
    std::vector<ThreadChain> chains;
    std::atomic<size_t> total_matches_count{0};
    std::vector<uint64_t> flat_results;
    bool is_flat = false;

public:
    MatchCollector() = default;
    MatchCollector(const MatchCollector&) = delete;
    MatchCollector& operator=(const MatchCollector&) = delete;
    
    ~MatchCollector() {
        for (auto& chain : chains) {
            MatchBlock* curr = chain.head;
            while (curr) {
                MatchBlock* next = curr->next;
                delete curr;
                curr = next;
            }
        }
    }
    /* keeps back combat */
    void reserve(size_t) {}
    
    /* no data copy just move head tail pointers */
    void merge_thread_buffer(ThreadLocalMatchBuffer& tlb) {
        if (tlb.total_count == 0) return;
        chains.push_back({tlb.head, tlb.total_count});
        total_matches_count += tlb.total_count;
        tlb.head = tlb.tail = tlb.current = nullptr;
        tlb.total_count = 0;
    }

    /* work in progress works for now */
    void finalize_parallel() {
        if (is_flat || total_matches_count == 0) return;
        flat_results.resize(total_matches_count);
        uint64_t* dest_base = flat_results.data();
        std::vector<size_t> chain_offsets;
        chain_offsets.reserve(chains.size());
        size_t running_offset = 0;
        struct CopyTask {
            const MatchBlock* block;
            size_t offset;
        };
        std::vector<CopyTask> tasks;
        tasks.reserve(total_matches_count / MATCH_CHUNK_SIZE + chains.size() * 2);

        for (const auto& chain : chains) {
            MatchBlock* curr = chain.head;
            size_t current_chain_offset = running_offset;
            
            while (curr) {
                tasks.push_back({curr, current_chain_offset});
                current_chain_offset += curr->count;
                curr = curr->next;
            }
            running_offset += chain.count;
        }

        size_t total_tasks = tasks.size();
        
        worker_pool.execute([&](size_t t, size_t num_threads) {
            size_t start = t * total_tasks / num_threads;
            size_t end = (t + 1) * total_tasks / num_threads;
            for (size_t i = start; i < end; ++i) {
                const auto& task = tasks[i];
                std::memcpy(dest_base + task.offset, 
                            task.block->matches, 
                            task.block->count * sizeof(uint64_t));
            }
        });
        is_flat = true;
        for (auto& chain : chains) {
            MatchBlock* curr = chain.head;
            while (curr) {
                MatchBlock* next = curr->next;
                delete curr;
                curr = next;
            }
        }
        chains.clear();
    }

    inline size_t size() const { return total_matches_count; }
    const std::vector<uint64_t>& get_flattened_matches() {
        if (!is_flat) finalize_parallel();
        return flat_results;
    }
};

} // namespace Contest

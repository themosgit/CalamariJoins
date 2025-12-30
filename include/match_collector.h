#pragma once


#include <vector>
#include <cstdint>
#include <memory>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <cstring>
#include <worker_pool.h>
#include <sys/mman.h>

namespace Contest {

// Align chunks to cache lines/pages for performance
static constexpr size_t MATCH_CHUNK_SIZE = 1024 * 4; // 4096 tuples per chunk (32KB)

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
 *  Thread-local buffer that accumulates matches into blocks.
 *  Prevents false sharing and lock contention during the probe phase.
 */
class ThreadLocalMatchBuffer {
    friend class MatchCollector;
    
    MatchBlock* head = nullptr;
    MatchBlock* tail = nullptr;
    MatchBlock* current = nullptr;
    size_t total_count = 0;
    
    // Simple pool for this thread to reduce allocation overhead during join
    std::vector<MatchBlock*> free_blocks;

public:
    ThreadLocalMatchBuffer() = default;
    
    // Move-only
    ThreadLocalMatchBuffer(ThreadLocalMatchBuffer&& other) noexcept 
        : head(other.head), tail(other.tail), current(other.current), 
          total_count(other.total_count) {
        other.head = other.tail = other.current = nullptr;
        other.total_count = 0;
    }

    ~ThreadLocalMatchBuffer() {
        // Cleanup happens when merged into global or explicitly destroyed
        // If not merged, we must delete blocks (safety check)
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
        // Pack: Left (Build) in lower 32, Right (Probe) in upper 32
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

/**
 *  Global Collector.
 *  1. Accepts lists of blocks from threads (Instant Merge).
 *  2. Flattens blocks to contiguous memory in parallel (Fast Read).
 */
class MatchCollector {
    // We store the chain of blocks from every thread
    struct ThreadChain {
        MatchBlock* head;
        size_t count;
    };
    
    std::vector<ThreadChain> chains;
    std::atomic<size_t> total_matches_count{0};

    // The contiguous result buffer (populated only on demand/finalize)
    std::vector<uint64_t> flat_results;
    bool is_flat = false;

public:
    MatchCollector() = default;

    // Delete copy, allow move
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

    /**
     *  Reserve is now mostly a hint or no-op since we use linked blocks,
     *  but we keep API compatibility.
     */
    void reserve(size_t) {}

    /**
     *  "Instant" merge: Takes ownership of the thread local buffer's linked list.
     *  No data copying happens here.
     */
    void merge_thread_buffer(ThreadLocalMatchBuffer& tlb) {
        if (tlb.total_count == 0) return;

        // Since this is called sequentially after parallel join, no lock needed.
        // If called concurrently, add a mutex here. 
        // Assuming usage in `hash_join.cpp` is sequential merge:
        chains.push_back({tlb.head, tlb.total_count});
        total_matches_count += tlb.total_count;

        // Detach ownership from TLB so it doesn't delete them
        tlb.head = tlb.tail = tlb.current = nullptr;
        tlb.total_count = 0;
    }

    /**
     *  Transforms the linked slabs into a single contiguous vector.
     *  Uses the WorkerPool to copy data in parallel.
     */
    void finalize_parallel() {
        if (is_flat || total_matches_count == 0) return;

        flat_results.resize(total_matches_count);
        uint64_t* dest_base = flat_results.data();

        // 1. Calculate write offsets for each chain (Prefix Sum)
        std::vector<size_t> chain_offsets;
        chain_offsets.reserve(chains.size());
        size_t running_offset = 0;
        
        // We also create a "Task List" of blocks to parallelize granularity better
        // A task is copying one specific block to a specific offset
        struct CopyTask {
            const MatchBlock* block;
            size_t offset;
        };
        std::vector<CopyTask> tasks;
        // Heuristic: Reserve enough for average blocks
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

        // 2. Parallel Copy
        // We use the worker pool to process the list of copy tasks
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
        
        // Optional: Free the linked lists now to save RAM, 
        // or keep them if you might need them (usually not).
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

    // Accessors for downstream compatibility
    inline size_t size() const { return total_matches_count; }
    
    // Note: Must call finalize_parallel() before calling matches.data()!
    // We modify the internal vector, so we return a reference to it.
    const std::vector<uint64_t>& get_flattened_matches() {
        if (!is_flat) finalize_parallel();
        return flat_results;
    }
    
    // Direct access for legacy code that expects a public vector
    // This allows `collector.matches` syntax to work if we rename flat_results or wrap it
    // But for cleaner code, we should update callsites to use `get_flattened_matches()`
};

} // namespace Contest

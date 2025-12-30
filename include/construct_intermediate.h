#pragma once


#include <columnar_reader.h>
#include <intermediate.h>
#include <join_setup.h>
#include <plan.h>
#include <vector>
#include <worker_pool.h>
#include <hardware_darwin.h>
#include <sys/mman.h>
#include <match_collector.h>
namespace Contest {

/**
 *
 *  batch-allocated memory using mmap
 *  manages lifecycle of large memory blocks for column pages
 *
 **/
class BatchAllocator {
private:
    void* memory_block = nullptr;
    size_t total_size = 0;

public:
    BatchAllocator() = default;

    void allocate(size_t total_pages) {
        total_size = total_pages * PAGE_SIZE;
        memory_block = mmap(nullptr, total_size,
                           PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory_block == MAP_FAILED) {
            memory_block = nullptr;
            throw std::bad_alloc();
        }
    }

    void* get_block() const { return memory_block; }

    ~BatchAllocator() {
        if (memory_block) {
            munmap(memory_block, total_size);
        }
    }

    BatchAllocator(const BatchAllocator&) = delete;
    BatchAllocator& operator=(const BatchAllocator&) = delete;
};

using ExecuteResult = std::vector<mema::column_t>;

/**
 *
 *  pre-resolved source information for a result column
 *  eliminates repetitive lookup logic inside parallel loops
 *
 **/
struct SourceInfo {
    const mema::column_t* intermediate_col = nullptr;
    const Column* columnar_col = nullptr;
    
    size_t remapped_col_idx = 0;
    uint32_t shift = 0; // 0 for Build, 32 for Probe
    
    bool is_columnar = false;
    bool from_build = false;
};

/**
 *
 *  calculates total pages needed and allocates single memory block
 *  distributes pre-allocated pages to result columns
 *
 **/
inline std::shared_ptr<BatchAllocator> batch_allocate_for_results(
    ExecuteResult& results, size_t total_matches) {

    size_t total_chunks = 0;
    for (auto& col : results) {
        total_chunks += (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;
    }
    
    size_t total_bytes = total_chunks * mema::CAP_PER_PAGE * sizeof(mema::value_t);
    size_t system_pages = (total_bytes + PAGE_SIZE - 1) / PAGE_SIZE;

    auto allocator = std::make_shared<BatchAllocator>();
    allocator->allocate(system_pages);
    
    void* block = allocator->get_block();
    size_t offset = 0;

    for (auto& col : results) {
        col.pre_allocate_from_block(block, offset, total_matches, allocator);
    }

    return allocator;
}

/**
 *
 *  resolves all data sources before execution
 *  handles logic for columnar/intermediate and build/probe mapping
 *
 **/
inline std::vector<SourceInfo> prepare_sources(
    const std::vector<std::tuple<size_t, DataType>>& remapped_attrs,
    const JoinInput& build_input,
    const JoinInput& probe_input,
    const PlanNode& build_node,
    const PlanNode& probe_node,
    size_t build_size)
{
    std::vector<SourceInfo> sources;
    sources.reserve(remapped_attrs.size());

    for (const auto& [col_idx, _] : remapped_attrs) {
        SourceInfo info;
        info.from_build = (col_idx < build_size);
        info.shift = info.from_build ? 0 : 32;
        
        size_t local_idx = info.from_build ? col_idx : col_idx - build_size;
        info.remapped_col_idx = local_idx;
        const JoinInput& input = info.from_build ? build_input : probe_input;
        const PlanNode& node = info.from_build ? build_node : probe_node;

        if (input.is_columnar()) {
            info.is_columnar = true;
            auto* table = std::get<const ColumnarTable*>(input.data);
            auto [actual_idx, _] = node.output_attrs[local_idx];
            info.columnar_col = &table->columns[actual_idx];
        } else {
            info.is_columnar = false;
            const auto& res = std::get<ExecuteResult>(input.data);
            info.intermediate_col = &res[local_idx];
        }

        sources.push_back(info);
    }
    return sources;
}

/**
 *
 *  constructs intermediate results from join matches
 *  parallelized with hoisted checks and batch memory allocation
 *
 **/
inline void construct_intermediate(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, ExecuteResult &results) {

    const size_t total_matches = collector.size();
    if (total_matches == 0) return;

    const auto& matches_vec = const_cast<MatchCollector&>(collector).get_flattened_matches();
    const uint64_t *matches_ptr = matches_vec.data();

    auto sources = prepare_sources(remapped_attrs, build_input, probe_input,
                                   build_node, probe_node, build_size);
    auto allocator = batch_allocate_for_results(results, total_matches);

    worker_pool.execute([&](size_t t, size_t num_threads) {
        size_t start = t * total_matches / num_threads;
        size_t end = (t + 1) * total_matches / num_threads;
        if (start >= end) return;

        for (size_t i = 0; i < sources.size(); ++i) {
            const auto& src = sources[i];
            auto& dest_col = results[i];

            if (src.is_columnar) {
                const auto& col = *src.columnar_col;
                if (src.from_build) {
                    for (size_t k = start; k < end; ++k) {
                        uint32_t rid = static_cast<uint32_t>(matches_ptr[k]);
                        dest_col.write_at(k, columnar_reader.read_value_build(
                            col, src.remapped_col_idx, rid, col.type));
                    }
                } else {
                    for (size_t k = start; k < end; ++k) {
                        uint32_t rid = static_cast<uint32_t>(matches_ptr[k] >> 32);
                        dest_col.write_at(k, columnar_reader.read_value_probe(
                            col, src.remapped_col_idx, rid, col.type));
                    }
                }
            } else {
                const auto& vec = *src.intermediate_col;
                uint32_t shift = src.shift;
                for (size_t k = start; k < end; ++k) {
                    uint32_t rid = static_cast<uint32_t>(matches_ptr[k] >> shift);
                    dest_col.write_at(k, vec[rid]);
                }
            }
        }
    });
}

} // namespace Contest

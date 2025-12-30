#pragma once

#include <columnar_reader.h>
#include <intermediate.h>
#include <join_setup.h>
#include <plan.h>
#include <vector>
#include <worker_pool.h>
#include <hardware_darwin.h>
#include <sys/mman.h>

namespace Contest {

/* batch-allocated memory using mmap */
class BatchAllocator {
private:
    void* memory_block = nullptr;
    size_t total_size = 0;

public:
    BatchAllocator() = default;
    /* allocate all pages */
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

    void* get_block() { return memory_block; }

    ~BatchAllocator() {
        if (memory_block) {
            munmap(memory_block, total_size);
        }
    }

    BatchAllocator(const BatchAllocator&) = delete;
    BatchAllocator& operator=(const BatchAllocator&) = delete;

    BatchAllocator(BatchAllocator&& other) noexcept
        : memory_block(other.memory_block), total_size(other.total_size) {
        other.memory_block = nullptr;
        other.total_size = 0;
    }
};

using ExecuteResult = std::vector<mema::column_t>;

struct ChunkTask {
    size_t col_idx;
    size_t chunk_start;
    size_t chunk_end;
};

struct ColumnSource {
    const mema::column_t* column;
    uint32_t shift;
};

inline size_t compute_chunk_size(size_t num_columns, size_t total_matches) {
    constexpr int NUM_CORES = SPC__CORE_COUNT;
    constexpr size_t L2_CACHE_SIZE = SPC__LEVEL2_CACHE_SIZE;
    size_t per_core_l2 = L2_CACHE_SIZE / NUM_CORES;
    size_t target_bytes_l2 = per_core_l2 / 2;
    size_t bytes_per_match = 8 + 4 + (num_columns * 4);
    size_t l2_optimal = target_bytes_l2 / bytes_per_match;
    #if defined(SPC__LEVEL3_CACHE_SIZE) && SPC__LEVEL3_CACHE_SIZE > 0
        constexpr size_t L3_CACHE_SIZE = SPC__LEVEL3_CACHE_SIZE;
        size_t per_core_l3 = L3_CACHE_SIZE / NUM_CORES;
        size_t target_bytes_l3 = per_core_l3 / 2;
        size_t l3_optimal = target_bytes_l3 / bytes_per_match;
        size_t cache_optimal = std::max(l2_optimal, l3_optimal);
    #else
        size_t cache_optimal = l2_optimal;
    #endif
    size_t min_total_tasks = NUM_CORES * 4;
    size_t total_work_items = total_matches * num_columns;
    size_t max_chunk_for_tasks = (total_work_items + min_total_tasks - 1) / min_total_tasks;
    size_t chunk_size = std::min(cache_optimal, max_chunk_for_tasks);
    return std::max(chunk_size, size_t(1024));
}

inline std::vector<ChunkTask> compute_tasks(
    size_t num_columns,
    size_t total_matches,
    size_t chunk_size)
{
    std::vector<ChunkTask> tasks;

    for (size_t col_idx = 0; col_idx < num_columns; ++col_idx) {
        for (size_t start = 0; start < total_matches; start += chunk_size) {
            size_t end = std::min(start + chunk_size, total_matches);
            tasks.push_back({col_idx, start, end});
        }
    }

    return tasks;
}

inline std::vector<ColumnSource> prepare_sources(
    const std::vector<std::tuple<size_t, DataType>>& output_attrs,
    const ExecuteResult& build,
    const ExecuteResult& probe,
    size_t build_size)
{
    std::vector<ColumnSource> sources;
    sources.reserve(output_attrs.size());

    for (const auto& [col_idx, _] : output_attrs) {
        bool from_build = (col_idx < build_size);
        const mema::column_t* src_col = from_build
            ? &build[col_idx]
            : &probe[col_idx - build_size];
        uint32_t shift = from_build ? 0 : 32;

        sources.push_back({src_col, shift});
    }

    return sources;
}

struct ColumnarSource {
    const Column* column;
    size_t remapped_col_idx;
    uint32_t shift;
    bool from_build;
};

inline std::vector<ColumnarSource> prepare_columnar_sources(
    const std::vector<std::tuple<size_t, DataType>>& remapped_attrs,
    const JoinInput& build_input,
    const JoinInput& probe_input,
    const PlanNode& build_node,
    const PlanNode& probe_node,
    size_t build_size)
{
    std::vector<ColumnarSource> sources;
    sources.reserve(remapped_attrs.size());

    auto* build_table = std::get<const ColumnarTable*>(build_input.data);
    auto* probe_table = std::get<const ColumnarTable*>(probe_input.data);

    for (const auto& [col_idx, data_type] : remapped_attrs) {
        bool from_build = col_idx < build_size;
        size_t remapped_col_idx = from_build ? col_idx : col_idx - build_size;

        const ColumnarTable* src_table = from_build ? build_table : probe_table;
        const PlanNode* src_node = from_build ? &build_node : &probe_node;
        auto [actual_col_idx, _] = src_node->output_attrs[remapped_col_idx];
        const Column& src_col = src_table->columns[actual_col_idx];

        uint32_t shift = from_build ? 0 : 32;

        sources.push_back({&src_col, remapped_col_idx, shift, from_build});
    }

    return sources;
}

struct MixedSource {
    const mema::column_t* intermediate_col;
    const Column* columnar_col;
    size_t remapped_col_idx;

    bool is_columnar;
    bool from_build;
    uint32_t shift;
};

/**
 *
 *  batch allocates memory for all result columns
 *  uses single mmap call and distributes pages across columns
 *  returns allocator that will be owned by columns for cleanup
 *
 **/
inline std::shared_ptr<BatchAllocator> batch_allocate_for_results(
    ExecuteResult& results, size_t total_matches) {

    size_t total_chunks = 0;
    for (auto& col : results) {
        /* Calculate how many logical chunks (mema::column_t::Page) are needed */
        total_chunks += (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;
    }

    /* 
     * FIX: Calculate total system pages required.
     * Each mema::column_t::Page holds CAP_PER_PAGE * sizeof(value_t) bytes.
     * CAP_PER_PAGE = 2048, sizeof(value_t) = 4, so Chunk Size = 8192 bytes.
     * System PAGE_SIZE is typically 4096 bytes.
     * We must allocate enough system pages to cover the chunk size.
     */
    size_t bytes_per_chunk = mema::CAP_PER_PAGE * sizeof(mema::value_t);
    size_t total_bytes = total_chunks * bytes_per_chunk;
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

inline std::vector<MixedSource> prepare_mixed_sources(
    const std::vector<std::tuple<size_t, DataType>>& remapped_attrs,
    const JoinInput& build_input,
    const JoinInput& probe_input,
    const PlanNode& build_node,
    const PlanNode& probe_node,
    size_t build_size)
{
    std::vector<MixedSource> sources;
    sources.reserve(remapped_attrs.size());

    for (const auto& [col_idx, data_type] : remapped_attrs) {
        bool from_build = col_idx < build_size;
        size_t remapped_col_idx = from_build ? col_idx : col_idx - build_size;
        uint32_t shift = from_build ? 0 : 32;

        bool source_is_columnar = false;
        const Column* columnar_src_col = nullptr;
        const mema::column_t* intermediate_src_col = nullptr;

        if (from_build) {
            if (build_input.is_columnar()) {
                source_is_columnar = true;
                auto* build_table = std::get<const ColumnarTable*>(build_input.data);
                auto [actual_col_idx, _] = build_node.output_attrs[remapped_col_idx];
                columnar_src_col = &build_table->columns[actual_col_idx];
            } else {
                const auto& build_result = std::get<ExecuteResult>(build_input.data);
                intermediate_src_col = &build_result[remapped_col_idx];
            }
        } else {
            if (probe_input.is_columnar()) {
                source_is_columnar = true;
                auto* probe_table = std::get<const ColumnarTable*>(probe_input.data);
                auto [actual_col_idx, _] = probe_node.output_attrs[remapped_col_idx];
                columnar_src_col = &probe_table->columns[actual_col_idx];
            } else {
                const auto& probe_result = std::get<ExecuteResult>(probe_input.data);
                intermediate_src_col = &probe_result[remapped_col_idx];
            }
        }

        sources.push_back({intermediate_src_col, columnar_src_col, remapped_col_idx,
                          source_is_columnar, from_build, shift});
    }

    return sources;
}

/**
 *
 *  stores join matches as packed 64-bit values
 *  lower 32 bits hold left row id
 *  upper 32 bits hold right row id
 *  enables efficient storage and cache locality
 *
 **/
struct MatchCollector {
    std::vector<uint64_t> matches;

    inline void reserve(size_t estimated_matches) {
        matches.reserve(estimated_matches);
    }

    /* packs left and right row ids into single 64-bit value */
    inline void add_match(uint32_t left, uint32_t right) {
        matches.push_back(static_cast<uint64_t>(left) |
                          (static_cast<uint64_t>(right) << 32));
    }

    inline size_t size() const { return matches.size(); }
};

struct JoinInput;
class ColumnarReader;

/**
 *
 * constructs intermediate results (vectors of values) from join matches
 * parallelized construction of non-columnar intermediate results
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

    /* 
     * Pre-allocate memory using the batch allocator.
     * This replaces the individual resize() calls which would do individual allocs.
     * Note: This function now correctly handles 8KB chunks vs 4KB system pages.
     */
    auto allocator = batch_allocate_for_results(results, total_matches);

    const uint64_t *matches_ptr = collector.matches.data();
    constexpr int num_threads = SPC__CORE_COUNT;

    worker_pool.execute([&](size_t t, size_t num_threads) {
        size_t start = t * total_matches / num_threads;
        size_t end = (t + 1) * total_matches / num_threads;
        if (start >= end) return;

        /* iterate over each requested output column */
        for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
            auto [col_idx, data_type] = remapped_attrs[out_idx];
            bool from_build = col_idx < build_size;
            size_t remapped_col_idx = from_build ? col_idx : col_idx - build_size;

            /* reference to destination column (paged vector) */
            auto& dest_col = results[out_idx];

            /* resolve data source */
            const Column *columnar_src_col = nullptr;
            const mema::column_t *intermediate_src_col = nullptr;

            if (from_build) {
                if (build_input.is_columnar()) {
                    auto *table = std::get<const ColumnarTable *>(build_input.data);
                    auto [actual_idx, _] = build_node.output_attrs[remapped_col_idx];
                    columnar_src_col = &table->columns[actual_idx];
                } else {
                    const auto &res = std::get<ExecuteResult>(build_input.data);
                    intermediate_src_col = &res[remapped_col_idx];
                }
            } else {
                if (probe_input.is_columnar()) {
                    auto *table = std::get<const ColumnarTable *>(probe_input.data);
                    auto [actual_idx, _] = probe_node.output_attrs[remapped_col_idx];
                    columnar_src_col = &table->columns[actual_idx];
                } else {
                    const auto &res = std::get<ExecuteResult>(probe_input.data);
                    intermediate_src_col = &res[remapped_col_idx];
                }
            }

            /* inner tight loop for filling this column chunk */
            if (columnar_src_col) {
                const auto& col = *columnar_src_col;
                /* 
                 * FIX: Use col.type (source physical type) instead of data_type (destination logical type).
                 * ColumnarReader needs the physical type to interpret the page bits correctly (e.g. VARCHAR vs INT32).
                 * If they mismatch, the Reader produces garbage which is then written to the result.
                 */
                if (from_build) {
                    for (size_t i = start; i < end; ++i) {
                        uint32_t rid = static_cast<uint32_t>(matches_ptr[i]);
                        dest_col.write_at(i, columnar_reader.read_value_build(
                            col, remapped_col_idx, rid, col.type));
                    }
                } else {
                    for (size_t i = start; i < end; ++i) {
                        uint32_t rid = static_cast<uint32_t>(matches_ptr[i] >> 32);
                        dest_col.write_at(i, columnar_reader.read_value_probe(
                            col, remapped_col_idx, rid, col.type));
                    }
                }
            } else {
                /* fast copy from intermediate vector */
                const auto& src_vec = *intermediate_src_col;
                if (from_build) {
                    for (size_t i = start; i < end; ++i) {
                        uint32_t rid = static_cast<uint32_t>(matches_ptr[i]);
                        dest_col.write_at(i, src_vec[rid]);
                    }
                } else {
                    for (size_t i = start; i < end; ++i) {
                        uint32_t rid = static_cast<uint32_t>(matches_ptr[i] >> 32);
                        dest_col.write_at(i, src_vec[rid]);
                    }
                }
            }
        }
    });
}
} // namespace Contest

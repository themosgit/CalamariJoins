#pragma once

#include <columnar_reader.h>
#include <intermediate.h>
#include <join_setup.h>
#include <plan.h>
#include <vector>
#include <thread>
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
 *  constructs intermediate results from column_t intermediate format
 *  both build and probe are intermediate results
 *  parallel implementation using multiple threads
 *
 **/
inline void construct_intermediate_from_intermediate(
    const MatchCollector &collector, const ExecuteResult &build,
    const ExecuteResult &probe,
    const std::vector<std::tuple<size_t, DataType>> &output_attrs,
    ExecuteResult &results) {
    const size_t total_matches = collector.size();
    if (total_matches == 0)
        return;

    constexpr size_t PARALLEL_THRESHOLD = 5000;
    if (total_matches < PARALLEL_THRESHOLD) {
        const uint64_t *matches_ptr = collector.matches.data();
        const size_t build_size = build.size();
        for (size_t out_idx = 0; out_idx < output_attrs.size(); ++out_idx) {
            auto [col_idx, _] = output_attrs[out_idx];
            const bool from_build = col_idx < build_size;
            const mema::column_t *column = from_build ? &build[col_idx] : &probe[col_idx - build_size];
            const uint32_t shift = from_build ? 0 : 32;
            results[out_idx].reserve(total_matches);
            for (size_t match_idx = 0; match_idx < total_matches; ++match_idx) {
                uint32_t row_id = (matches_ptr[match_idx] >> shift);
                const mema::value_t &val = (*column)[row_id];
                results[out_idx].append(val);
            }
        }
        return;
    }

    const uint64_t *matches_ptr = collector.matches.data();
    const size_t build_size = build.size();
    const size_t chunk_size = compute_chunk_size(output_attrs.size(), total_matches);
    const auto tasks = compute_tasks(output_attrs.size(), total_matches, chunk_size);
    const auto sources = prepare_sources(output_attrs, build, probe, build_size);

    size_t total_pages = 0;
    for (auto& col : results) {
        total_pages += (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;
    }

    auto allocator = std::make_shared<BatchAllocator>();
    allocator->allocate(total_pages);
    void* block = allocator->get_block();
    size_t offset = 0;

    for (auto& col : results) {
        col.pre_allocate_from_block(block, offset, total_matches, allocator);
    }

    constexpr int num_threads = SPC__CORE_COUNT;

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t] {
            for (size_t task_idx = t; task_idx < tasks.size(); task_idx += num_threads) {
                const auto& task = tasks[task_idx];
                const auto& src = sources[task.col_idx];
                auto& dest = results[task.col_idx];
                for (size_t i = task.chunk_start; i < task.chunk_end; ++i) {
                    const uint32_t row_id = static_cast<uint32_t>(matches_ptr[i] >> src.shift);
                    dest.write_at(i, (*src.column)[row_id]);
                }
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
}

/**
 *
 *  constructs intermediate results directly from ColumnarTable columnar inputs
 *  both build and probe are reading from original page format
 *  parallel implementation using multiple threads
 *
 **/
inline void construct_intermediate_from_columnar(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, ExecuteResult &results) {

    const size_t total_matches = collector.size();
    if (total_matches == 0)
        return;

    const uint64_t *matches_ptr = collector.matches.data();
    const size_t chunk_size = compute_chunk_size(remapped_attrs.size(), total_matches);
    const auto tasks = compute_tasks(remapped_attrs.size(), total_matches, chunk_size);
    const auto sources = prepare_columnar_sources(remapped_attrs, build_input, probe_input,
                                                   build_node, probe_node, build_size);

    size_t total_pages = 0;
    for (auto& col : results) {
        total_pages += (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;
    }

    auto allocator = std::make_shared<BatchAllocator>();
    allocator->allocate(total_pages);
    void* block = allocator->get_block();
    size_t offset = 0;

    for (auto& col : results) {
        col.pre_allocate_from_block(block, offset, total_matches, allocator);
    }

    constexpr int num_threads = SPC__CORE_COUNT;
    std::vector<ColumnarReader> thread_readers(num_threads);

    // prepare column lists for readers
    std::vector<const Column*> build_columns;
    auto* build_table = std::get<const ColumnarTable*>(build_input.data);
    for (size_t i = 0; i < build_node.output_attrs.size(); ++i) {
        auto [actual_col_idx, _] = build_node.output_attrs[i];
        build_columns.push_back(&build_table->columns[actual_col_idx]);
    }
    std::vector<const Column*> probe_columns;
    auto* probe_table = std::get<const ColumnarTable*>(probe_input.data);
    for (size_t i = 0; i < probe_node.output_attrs.size(); ++i) {
        auto [actual_col_idx, _] = probe_node.output_attrs[i];
        probe_columns.push_back(&probe_table->columns[actual_col_idx]);
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t] {
            thread_readers[t].prepare_build(build_columns);
            thread_readers[t].prepare_probe(probe_columns);

            auto& reader = thread_readers[t];
            for (size_t task_idx = t; task_idx < tasks.size(); task_idx += num_threads) {
                const auto& task = tasks[task_idx];
                const auto& src = sources[task.col_idx];
                auto& dest = results[task.col_idx];
                for (size_t i = task.chunk_start; i < task.chunk_end; ++i) {
                    const uint32_t row_id = static_cast<uint32_t>(matches_ptr[i] >> src.shift);
                    mema::value_t value = src.from_build
                                              ? reader.read_value_build(
                                                    *src.column, src.remapped_col_idx, row_id,
                                                    src.column->type)
                                              : reader.read_value_probe(
                                                    *src.column, src.remapped_col_idx, row_id,
                                                    src.column->type);
                    dest.write_at(i, value);
                }
            }
        });
    }

    for (auto& w : workers) {
        w.join();
    }
}

/**
 *
 *  constructs intermediate results from mixed input types
 *  one side is ColumnarTable columnar other is column_t intermediate
 *  parallel implementation using multiple threads
 *
 **/
inline void construct_intermediate_mixed(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, ExecuteResult &results) {

    const size_t total_matches = collector.size();
    if (total_matches == 0)
        return;

    const uint64_t *matches_ptr = collector.matches.data();
    const size_t chunk_size = compute_chunk_size(remapped_attrs.size(), total_matches);
    const auto tasks = compute_tasks(remapped_attrs.size(), total_matches, chunk_size);
    const auto sources = prepare_mixed_sources(remapped_attrs, build_input, probe_input,
                                                build_node, probe_node, build_size);

    size_t total_pages = 0;
    for (auto& col : results) {
        total_pages += (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;
    }

    auto allocator = std::make_shared<BatchAllocator>();
    allocator->allocate(total_pages);
    void* block = allocator->get_block();
    size_t offset = 0;

    for (auto& col : results) {
        col.pre_allocate_from_block(block, offset, total_matches, allocator);
    }

    constexpr int num_threads = SPC__CORE_COUNT;
    std::vector<ColumnarReader> thread_readers(num_threads);
    std::vector<const Column*> build_columns;
    std::vector<const Column*> probe_columns;
    bool has_build_columnar = build_input.is_columnar();
    bool has_probe_columnar = probe_input.is_columnar();

    if (has_build_columnar) {
        auto* build_table = std::get<const ColumnarTable*>(build_input.data);
        for (size_t i = 0; i < build_node.output_attrs.size(); ++i) {
            auto [actual_col_idx, _] = build_node.output_attrs[i];
            build_columns.push_back(&build_table->columns[actual_col_idx]);
        }
    }
    if (has_probe_columnar) {
        auto* probe_table = std::get<const ColumnarTable*>(probe_input.data);
        for (size_t i = 0; i < probe_node.output_attrs.size(); ++i) {
            auto [actual_col_idx, _] = probe_node.output_attrs[i];
            probe_columns.push_back(&probe_table->columns[actual_col_idx]);
        }
    }

    std::vector<std::thread> workers;
    workers.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t, has_build_columnar, has_probe_columnar] {
            if (has_build_columnar) {
                thread_readers[t].prepare_build(build_columns);
            }
            if (has_probe_columnar) {
                thread_readers[t].prepare_probe(probe_columns);
            }

            auto& reader = thread_readers[t];
            for (size_t task_idx = t; task_idx < tasks.size(); task_idx += num_threads) {
                const auto& task = tasks[task_idx];
                const auto& src = sources[task.col_idx];
                auto& dest = results[task.col_idx];
                for (size_t i = task.chunk_start; i < task.chunk_end; ++i) {
                    const uint32_t row_id = static_cast<uint32_t>(matches_ptr[i] >> src.shift);

                    mema::value_t value;
                    if (src.is_columnar) {
                        value = src.from_build
                                    ? reader.read_value_build(
                                          *src.columnar_col, src.remapped_col_idx,
                                          row_id, src.columnar_col->type)
                                    : reader.read_value_probe(
                                          *src.columnar_col, src.remapped_col_idx,
                                          row_id, src.columnar_col->type);
                    } else {
                        value = (*src.intermediate_col)[row_id];
                    }
                    dest.write_at(i, value);
                }
            }
        });
    }
    for (auto& w : workers) {
        w.join();
    }
}

} // namespace Contest

/**
 * @file construct_intermediate.h
 * @brief Constructs intermediate results for multi-way joins.
 *
 * Allocates and populates ExecuteResult (column_t) from match collectors.
 */
#pragma once

#include <data_access/columnar_reader.h>
#include <data_model/intermediate.h>
#include <data_model/plan.h>
#include <join_execution/join_setup.h>
#include <join_execution/match_collector.h>
#include <platform/worker_pool.h>
#include <sys/mman.h>
#include <vector>
namespace Contest {

/**
 * @brief Batch memory allocator for intermediate result pages.
 *
 * Allocates a contiguous mmap'd block for all column_t pages, avoiding
 * per-page allocation overhead. Shared ownership via shared_ptr ensures
 * memory outlives columns referencing it.
 */
class BatchAllocator {
  private:
    void *memory_block = nullptr;
    size_t total_size = 0;

  public:
    BatchAllocator() = default;
    void allocate(size_t total_pages) {
        total_size = total_pages * mema::IR_PAGE_SIZE;
        memory_block = mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory_block == MAP_FAILED) {
            memory_block = nullptr;
            throw std::bad_alloc();
        }
    }
    void *get_block() const { return memory_block; }
    ~BatchAllocator() {
        if (memory_block) {
            munmap(memory_block, total_size);
        }
    }
    BatchAllocator(const BatchAllocator &) = delete;
    BatchAllocator &operator=(const BatchAllocator &) = delete;
};

/**
 * @brief Metadata for resolving an output column's source.
 *
 * Precomputed during prepare_sources() to avoid repeated lookups
 * during the hot loop of construct_intermediate().
 */
struct alignas(8) SourceInfo {
    const mema::column_t *intermediate_col =
        nullptr;                          /**< Source if intermediate. */
    const Column *columnar_col = nullptr; /**< Source if columnar. */
    size_t remapped_col_idx = 0; /**< Local index within source side. */
    bool is_columnar = false;    /**< True if source is columnar table. */
    bool from_build = false; /**< True if from build side, false if probe. */
};

/**
 * @brief Preallocates mmap'd memory for all output columns.
 *
 * Computes total pages needed across all columns, allocates single block,
 * then partitions to each column via pre_allocate_from_block().
 * Returns allocator holding shared ownership of the memory.
 */
inline std::shared_ptr<BatchAllocator>
batch_allocate_for_results(ExecuteResult &results, size_t total_matches) {

    size_t total_chunks = 0;
    for (auto &col : results) {
        total_chunks +=
            (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;
    }

    size_t total_bytes =
        total_chunks * mema::CAP_PER_PAGE * sizeof(mema::value_t);
    size_t system_pages =
        (total_bytes + mema::IR_PAGE_SIZE - 1) / mema::IR_PAGE_SIZE;
    auto allocator = std::make_shared<BatchAllocator>();
    allocator->allocate(system_pages);
    void *block = allocator->get_block();
    size_t offset = 0;
    for (auto &col : results) {
        col.pre_allocate_from_block(block, offset, total_matches, allocator);
    }
    return allocator;
}

/** @brief Builds SourceInfo for each output column for fast lookup during copy.
 */
inline std::vector<SourceInfo>
prepare_sources(const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
                const JoinInput &build_input, const JoinInput &probe_input,
                const PlanNode &build_node, const PlanNode &probe_node,
                size_t build_size) {
    std::vector<SourceInfo> sources;
    sources.reserve(remapped_attrs.size());
    for (const auto &[col_idx, _] : remapped_attrs) {
        SourceInfo info;
        info.from_build = (col_idx < build_size);
        size_t local_idx = info.from_build ? col_idx : col_idx - build_size;
        info.remapped_col_idx = local_idx;
        const JoinInput &input = info.from_build ? build_input : probe_input;
        const PlanNode &node = info.from_build ? build_node : probe_node;
        if (input.is_columnar()) {
            info.is_columnar = true;
            auto *table = std::get<const ColumnarTable *>(input.data);
            auto [actual_idx, _] = node.output_attrs[local_idx];
            info.columnar_col = &table->columns[actual_idx];
        } else {
            info.is_columnar = false;
            const auto &res = std::get<ExecuteResult>(input.data);
            info.intermediate_col = &res[local_idx];
        }
        sources.push_back(info);
    }
    return sources;
}

/**
 * @brief Populates ExecuteResult columns from join matches in parallel.
 *
 * Each worker thread processes a slice of matches, copying values from
 * source columns (columnar or intermediate) to pre-allocated output columns.
 * Uses ColumnarReader cursor caching for efficient page access.
 *
 * @param collector        Source of match row IDs.
 * @param build_input      Build side data.
 * @param probe_input      Probe side data.
 * @param remapped_attrs   Output column specs (remapped to build/probe order).
 * @param build_node       PlanNode for build side column mapping.
 * @param probe_node       PlanNode for probe side column mapping.
 * @param build_size       Number of columns from build side.
 * @param columnar_reader  Reader with prepared columns for page access.
 * @param results          Pre-initialized output columns to populate.
 */
inline void construct_intermediate(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, ExecuteResult &results) {
    const_cast<MatchCollector &>(collector).ensure_finalized();
    const size_t total_matches = collector.size();
    if (total_matches == 0)
        return;

    auto sources = prepare_sources(remapped_attrs, build_input, probe_input,
                                   build_node, probe_node, build_size);
    auto allocator = batch_allocate_for_results(results, total_matches);

    worker_pool.execute([&](size_t t) {
        Contest::ColumnarReader::Cursor cursor;
        size_t num_threads = worker_pool.thread_count();
        size_t start = t * total_matches / num_threads;
        size_t end = (t + 1) * total_matches / num_threads;
        if (start >= end)
            return;

        for (size_t i = 0; i < sources.size(); ++i) {
            const auto &src = sources[i];
            auto &dest_col = results[i];

            auto range = src.from_build
                             ? collector.get_left_range(start, end - start)
                             : collector.get_right_range(start, end - start);

            if (src.is_columnar) {
                const auto &col = *src.columnar_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    dest_col.write_at(k++,
                                      columnar_reader.read_value(
                                          col, src.remapped_col_idx, rid,
                                          col.type, cursor, src.from_build));
                }
            } else {
                const auto &vec = *src.intermediate_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    dest_col.write_at(k++, vec[rid]);
                }
            }
        }
    });
}
} // namespace Contest

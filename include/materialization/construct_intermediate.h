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
#include <platform/arena.h>
#include <platform/worker_pool.h>
#include <vector>
/**
 * @namespace Contest::materialize
 * @brief Materialization of join results into columnar format.
 *
 * @see intermediate.h for column_t/value_t format details.
 */
namespace Contest::materialize {

using Contest::ExecuteResult;
using Contest::io::ColumnarReader;
using Contest::join::JoinInput;
using Contest::join::MatchCollector;
using Contest::platform::worker_pool;

/**
 * @brief Preallocates arena memory for all output columns.
 *
 * Distributes page allocations across all thread arenas to balance memory
 * usage.
 *
 * @param results       Output columns to allocate (modified in-place).
 * @param total_matches Number of rows (match count from MatchCollector).
 *
 * @see column_t::pre_allocate_from_arena() for arena partitioning.
 */
inline void batch_allocate_for_results(ExecuteResult &results,
                                       size_t total_matches) {
    size_t num_threads = worker_pool.thread_count();
    size_t col_idx = 0;
    for (auto &col : results) {
        // Round-robin allocation across threads to distribute memory pressure
        size_t thread_id = col_idx % num_threads;
        col.pre_allocate_from_arena(Contest::platform::get_arena(thread_id),
                                    total_matches);
        col_idx++;
    }
}

/**
 * @brief Precomputed metadata for resolving an output column's source.
 *
 * Avoids per-value std::variant accesses and tuple lookups in hot loop.
 * 8-byte alignment optimizes struct packing for vector iteration.
 *
 * @see prepare_sources() for precomputation logic.
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
 * @brief Builds SourceInfo for each output column for fast hot-loop lookup.
 *
 * @param remapped_attrs Output column specifications (global indexing).
 * @param build_input    Build side data (ColumnarTable* or ExecuteResult).
 * @param probe_input    Probe side data (ColumnarTable* or ExecuteResult).
 * @param build_node     PlanNode for build side (contains output_attrs).
 * @param probe_node     PlanNode for probe side (contains output_attrs).
 * @param build_size     Number of columns from build side.
 * @return Vector of SourceInfo, one per output column.
 *
 * @see SourceInfo for field documentation.
 * @see construct_intermediate() for consumption in hot loop.
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
 * Core materialization for non-root joins. Transforms match row ID pairs into
 * column_t with value_t encoding. Uses intermediate format (4-byte references)
 * for cache efficiency; only root node converts to ColumnarTable.
 *
 * @param collector        MatchCollector with finalized match row IDs.
 * @param build_input      Build side data (ColumnarTable* or ExecuteResult).
 * @param probe_input      Probe side data (ColumnarTable* or ExecuteResult).
 * @param remapped_attrs   Output column specifications (global indexing).
 * @param build_node       PlanNode for build side output_attrs mapping.
 * @param probe_node       PlanNode for probe side output_attrs mapping.
 * @param build_size       Number of output columns from build side.
 * @param columnar_reader  ColumnarReader with Cursor caching for page access.
 * @param results          Pre-initialized ExecuteResult, populated in-place.
 *
 * @note For root joins, use materialize() instead.
 * @see intermediate.h for column_t format and value_t encoding.
 * @see prepare_sources() for source metadata precomputation.
 * @see batch_allocate_for_results() for single mmap allocation.
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
    batch_allocate_for_results(results, total_matches);

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
} // namespace Contest::materialize

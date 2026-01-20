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
using Contest::platform::THREAD_COUNT;
using Contest::platform::worker_pool;

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
 * @brief Constructs intermediate results directly from thread-local buffers.
 *
 * Each thread iterates its own buffer, avoiding the merge step. Total matches
 * computed by summing buffer counts. Each thread writes its contiguous portion
 * of output pages.
 *
 * @param buffers          Vector of ThreadLocalMatchBuffer from probe.
 * @param build_input      Build side data (ColumnarTable* or ExecuteResult).
 * @param probe_input      Probe side data (ColumnarTable* or ExecuteResult).
 * @param remapped_attrs   Output column specifications (global indexing).
 * @param build_node       PlanNode for build side output_attrs mapping.
 * @param probe_node       PlanNode for probe side output_attrs mapping.
 * @param build_size       Number of output columns from build side.
 * @param columnar_reader  ColumnarReader with Cursor caching for page access.
 * @param results          Pre-initialized ExecuteResult, populated in-place.
 */
inline void construct_intermediate_from_buffers(
    std::vector<Contest::join::ThreadLocalMatchBuffer> &buffers,
    const JoinInput &build_input, const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, ExecuteResult &results) {

    // Compute total matches and per-buffer start offsets
    size_t total_matches = 0;
    std::vector<size_t> buffer_starts(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        buffer_starts[i] = total_matches;
        total_matches += buffers[i].count();
    }

    if (total_matches == 0)
        return;

    auto sources = prepare_sources(remapped_attrs, build_input, probe_input,
                                   build_node, probe_node, build_size);

    const size_t num_threads = THREAD_COUNT;
    const size_t num_cols = sources.size();

    // Pre-size page vectors for each column
    using Page = mema::column_t::Page;
    size_t total_pages_needed =
        (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;

    for (size_t c = 0; c < num_cols; ++c) {
        auto &col = results[c];
        col.pages.resize(total_pages_needed);
        col.set_row_count(total_matches);
    }

    // Parallel page allocation - each thread allocates its own pages
    worker_pool().execute([&](size_t t) {
        for (size_t c = 0; c < num_cols; ++c) {
            auto &col = results[c];
            for (size_t p = t; p < total_pages_needed; p += num_threads) {
                void *ptr =
                    Contest::platform::get_arena(t)
                        .alloc_chunk<Contest::platform::ChunkType::IR_PAGE>();
                col.pages[p] = reinterpret_cast<Page *>(ptr);
            }
        }
    });

    // Parallel: each thread processes its own buffer
    worker_pool().execute([&](size_t t) {
        if (t >= buffers.size())
            return;
        auto &buf = buffers[t];
        size_t my_count = buf.count();
        if (my_count == 0)
            return;

        size_t start = buffer_starts[t];
        Contest::ColumnarReader::Cursor cursor;

        for (size_t c = 0; c < num_cols; ++c) {
            const auto &src = sources[c];
            auto &dest_col = results[c];

            auto left_range = buf.left_range();
            auto right_range = buf.right_range();

            if (src.is_columnar) {
                const auto &col = *src.columnar_col;
                if (src.from_build) {
                    size_t k = start;
                    for (uint32_t rid : left_range) {
                        dest_col.write_at(k++,
                                          columnar_reader.read_value(
                                              col, src.remapped_col_idx, rid,
                                              col.type, cursor, true));
                    }
                } else {
                    size_t k = start;
                    for (uint32_t rid : right_range) {
                        dest_col.write_at(k++,
                                          columnar_reader.read_value(
                                              col, src.remapped_col_idx, rid,
                                              col.type, cursor, false));
                    }
                }
            } else {
                const auto &vec = *src.intermediate_col;
                if (src.from_build) {
                    size_t k = start;
                    for (uint32_t rid : left_range) {
                        dest_col.write_at(k++, vec[rid]);
                    }
                } else {
                    size_t k = start;
                    for (uint32_t rid : right_range) {
                        dest_col.write_at(k++, vec[rid]);
                    }
                }
            }
        }
    });
}

} // namespace Contest::materialize

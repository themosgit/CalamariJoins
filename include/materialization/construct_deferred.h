/**
 * @file construct_deferred.h
 * @brief Constructs deferred intermediate results for multi-way joins.
 *
 * Allocates and populates DeferredResult with only MATERIALIZE columns
 * (typically just the parent's join key). Row ID columns are always
 * populated for provenance tracking.
 *
 * @see construct_intermediate.h for the eager materialization equivalent.
 * @see materialize_deferred.h for final resolution of deferred columns.
 */
#pragma once

#include <algorithm>
#include <cstring>
#include <vector>

#include <data_access/columnar_reader.h>
#include <data_model/deferred_intermediate.h>
#include <data_model/deferred_plan.h>
#include <foundation/common.h>
#include <join_execution/match_collector.h>
#include <platform/arena.h>
#include <platform/worker_pool.h>

namespace Contest {
namespace materialize {

using Contest::io::ColumnarReader;
using Contest::join::MatchCollectionMode;
using Contest::join::ThreadLocalMatchBuffer;
using Contest::platform::THREAD_COUNT;
using Contest::platform::worker_pool;

/**
 * @brief Collect columns needed from a DeferredInput for page index building.
 */
inline platform::ArenaVector<const Column *>
collect_deferred_columns(const DeferredInput &input,
                         const platform::ArenaVector<uint8_t> &needed,
                         platform::ThreadArena &arena) {
    platform::ArenaVector<const Column *> columns(arena);
    if (!input.node)
        return columns;

    columns.resize(input.node->output_attrs.size());
    std::memset(columns.data(), 0, columns.size() * sizeof(const Column *));

    if (!input.is_columnar())
        return columns;

    auto *table = std::get<const ColumnarTable *>(input.data);
    for (size_t i = 0; i < input.node->output_attrs.size(); ++i) {
        if (i < needed.size() && needed[i]) {
            auto [actual_col_idx, _] = input.node->output_attrs[i];
            columns[i] = &table->columns[actual_col_idx];
        }
    }
    return columns;
}

/**
 * @brief Prepare ColumnarReader for deferred materialization path.
 *
 * Sets up page indices for columns that need to be read from columnar inputs.
 */
inline void prepare_deferred_columns(
    ColumnarReader &reader, const DeferredInput &build_input,
    const DeferredInput &probe_input, const DeferredJoinNode &join_node,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    size_t build_size, bool build_is_left) {

    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();

    if (!build_is_columnar && !probe_is_columnar)
        return;

    auto &arena = Contest::platform::get_arena(0);

    // Determine which columns from each side are needed
    platform::ArenaVector<uint8_t> build_needed(arena);
    if (build_input.node) {
        build_needed.resize(build_input.node->output_attrs.size());
        std::memset(build_needed.data(), 0, build_needed.size());
    }

    platform::ArenaVector<uint8_t> probe_needed(arena);
    if (probe_input.node) {
        probe_needed.resize(probe_input.node->output_attrs.size());
        std::memset(probe_needed.data(), 0, probe_needed.size());
    }

    // Mark columns needed based on materialization decisions
    // from_left refers to original left child
    // build_is_left tells us if build side is the left child
    for (const auto &col : join_node.columns) {
        if (col.resolution == ColumnResolution::MATERIALIZE) {
            bool from_build = (col.from_left == build_is_left);
            if (from_build && col.child_output_idx < build_needed.size()) {
                build_needed[col.child_output_idx] = 1;
            } else if (!from_build &&
                       col.child_output_idx < probe_needed.size()) {
                probe_needed[col.child_output_idx] = 1;
            }
        }
    }

    if (build_is_columnar) {
        reader.prepare_build(
            collect_deferred_columns(build_input, build_needed, arena));
    }

    if (probe_is_columnar) {
        reader.prepare_probe(
            collect_deferred_columns(probe_input, probe_needed, arena));
    }
}

/**
 * @brief Create empty deferred result with proper schema.
 *
 * Used when total_matches == 0. Creates empty materialized columns
 * for columns marked MATERIALIZE so they can be used in subsequent joins.
 */
inline DeferredResult
create_empty_deferred_result(const DeferredJoinNode &node) {
    DeferredResult result;
    result.node_info = &node;
    result.num_rows = 0;
    result.materialized_map.resize(node.columns.size(), std::nullopt);
    result.table_ids = node.tracked_table_ids;

    // Count and allocate empty materialized columns
    size_t mat_count = 0;
    for (const auto &col : node.columns) {
        if (col.resolution == ColumnResolution::MATERIALIZE) {
            result.materialized_map[col.original_idx] = mat_count++;
        }
    }
    result.materialized.resize(mat_count);
    // Each column has 0 rows, which is valid for empty result

    // Also create empty row ID columns
    result.row_ids.resize(node.tracked_table_ids.size());
    for (size_t i = 0; i < node.tracked_table_ids.size(); ++i) {
        result.row_ids[i].table_id = node.tracked_table_ids[i];
    }

    return result;
}

/**
 * @brief Precomputed metadata for row ID column sources.
 *
 * Mirrors RowIdSource from construct_intermediate.h but adapted for
 * DeferredInput.
 */
struct DeferredRowIdSource {
    const mema::rowid_column_t *source_col =
        nullptr;               ///< Source if from intermediate.
    uint8_t table_id = 0;      ///< Table ID for encoding.
    bool from_build = false;   ///< True if from build side.
    bool needs_encode = false; ///< True if columnar (needs GlobalRowId encode).
};

/**
 * @brief Prepare row ID sources for deferred intermediate construction.
 */
inline std::vector<DeferredRowIdSource>
prepare_deferred_rowid_sources(const std::vector<uint8_t> &merged_table_ids,
                               const DeferredInput &build_input,
                               const DeferredInput &probe_input) {
    std::vector<DeferredRowIdSource> sources;
    sources.reserve(merged_table_ids.size());

    for (uint8_t tid : merged_table_ids) {
        DeferredRowIdSource src;
        src.table_id = tid;

        // Check build side first
        auto build_tables = build_input.tracked_tables();
        bool in_build = std::find(build_tables.begin(), build_tables.end(),
                                  tid) != build_tables.end();
        if (in_build) {
            src.from_build = true;
            if (build_input.is_columnar()) {
                src.needs_encode = true;
                src.source_col = nullptr;
            } else {
                src.needs_encode = false;
                src.source_col = build_input.get_rowid_column(tid);
            }
        } else {
            // Must be from probe side
            src.from_build = false;
            if (probe_input.is_columnar()) {
                src.needs_encode = true;
                src.source_col = nullptr;
            } else {
                src.needs_encode = false;
                src.source_col = probe_input.get_rowid_column(tid);
            }
        }
        sources.push_back(src);
    }
    return sources;
}

/**
 * @brief Constructs deferred intermediate result from thread-local buffers.
 *
 * Only materializes columns marked MATERIALIZE in the DeferredJoinNode.
 * All row ID columns are populated for provenance tracking.
 *
 * @tparam Mode            Collection mode for compile-time specialization.
 * @param buffers          Thread-local match buffers from probe.
 * @param build_input      Build side data source.
 * @param probe_input      Probe side data source.
 * @param join_node        Deferred join node with materialization decisions.
 * @param remapped_attrs   Output attributes (after build/probe remapping).
 * @param build_output_size Number of columns from build side.
 * @param columnar_reader  Reader for columnar data access.
 * @param out_result       Output DeferredResult (populated in-place).
 * @param merged_table_ids Sorted table IDs to track.
 * @param deferred_plan    Full deferred plan for base table access (deferred
 * resolution).
 */
template <MatchCollectionMode Mode>
void construct_deferred_from_buffers(
    std::vector<ThreadLocalMatchBuffer<Mode>> &buffers,
    const DeferredInput &build_input, const DeferredInput &probe_input,
    const DeferredJoinNode &join_node,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    size_t build_output_size, bool build_is_left,
    ColumnarReader &columnar_reader, DeferredResult &out_result,
    const std::vector<uint8_t> &merged_table_ids,
    const DeferredPlan &deferred_plan) {

    // Count total matches
    size_t total_matches = 0;
    std::vector<size_t> buffer_starts(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        buffer_starts[i] = total_matches;
        total_matches += buffers[i].count();
    }

    if (total_matches == 0) {
        out_result = create_empty_deferred_result(join_node);
        return;
    }

    out_result.node_info = &join_node;
    out_result.num_rows = total_matches;
    out_result.table_ids = merged_table_ids;

    // Build materialized_map: count MATERIALIZE columns and create mapping
    // materialized_map[original_idx] -> index into out_result.materialized
    out_result.materialized_map.resize(join_node.columns.size(), std::nullopt);
    size_t mat_count = 0;

    // Iterate over join_node.columns (which uses original output order)
    // and assign materialized indices to MATERIALIZE columns
    for (const auto &col : join_node.columns) {
        if (col.resolution == ColumnResolution::MATERIALIZE) {
            out_result.materialized_map[col.original_idx] = mat_count++;
        }
    }

    // Prepare row ID sources
    auto rowid_sources = prepare_deferred_rowid_sources(
        merged_table_ids, build_input, probe_input);

    const size_t num_rowid_cols = rowid_sources.size();

    // Pre-allocate pages
    using Page = mema::column_t::Page;
    using RowIdPage = mema::rowid_column_t::Page;
    size_t total_pages_needed =
        (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;

    // Allocate materialized columns
    out_result.materialized.resize(mat_count);
    for (size_t c = 0; c < mat_count; ++c) {
        out_result.materialized[c].pages.resize(total_pages_needed);
        out_result.materialized[c].set_row_count(total_matches);
    }

    // Allocate row ID columns
    out_result.row_ids.resize(num_rowid_cols);
    for (size_t r = 0; r < num_rowid_cols; ++r) {
        out_result.row_ids[r].table_id = merged_table_ids[r];
        out_result.row_ids[r].pages.resize(total_pages_needed);
        out_result.row_ids[r].set_row_count(total_matches);
    }

    // Parallel page allocation
    const size_t num_threads = THREAD_COUNT;
    worker_pool().execute([&](size_t t) {
        for (size_t c = 0; c < mat_count; ++c) {
            auto &col = out_result.materialized[c];
            for (size_t p = t; p < total_pages_needed; p += num_threads) {
                void *ptr =
                    Contest::platform::get_arena(t)
                        .alloc_chunk<Contest::platform::ChunkType::IR_PAGE>();
                col.pages[p] = reinterpret_cast<Page *>(ptr);
            }
        }
        for (size_t r = 0; r < num_rowid_cols; ++r) {
            auto &rid_col = out_result.row_ids[r];
            for (size_t p = t; p < total_pages_needed; p += num_threads) {
                void *ptr =
                    Contest::platform::get_arena(t)
                        .alloc_chunk<Contest::platform::ChunkType::IR_PAGE>();
                rid_col.pages[p] = reinterpret_cast<RowIdPage *>(ptr);
            }
        }
    });

    // Set source metadata for materialized columns
    for (const auto &col : join_node.columns) {
        if (col.resolution == ColumnResolution::MATERIALIZE) {
            size_t mat_idx = *out_result.materialized_map[col.original_idx];
            out_result.materialized[mat_idx].source_table =
                col.provenance.base_table_id;
            out_result.materialized[mat_idx].source_column =
                col.provenance.base_column_idx;
        }
    }

    // Parallel population: each thread processes its own buffer
    worker_pool().execute([&](size_t t) {
        if (t >= buffers.size())
            return;
        auto &buf = buffers[t];
        size_t my_count = buf.count();
        if (my_count == 0)
            return;

        size_t start = buffer_starts[t];
        ColumnarReader::Cursor cursor;

        // Iterate through matches
        auto left_it = buf.left_range().begin();
        auto right_it = buf.right_range().begin();

        for (size_t m = 0; m < my_count; ++m) {
            uint32_t build_row = 0, probe_row = 0;

            if constexpr (Mode == MatchCollectionMode::BOTH) {
                build_row = *left_it;
                probe_row = *right_it;
                ++left_it;
                ++right_it;
            } else if constexpr (Mode == MatchCollectionMode::LEFT_ONLY) {
                build_row = *left_it;
                ++left_it;
            } else {
                probe_row = *right_it;
                ++right_it;
            }

            size_t out_row = start + m;

            // Write materialized columns
            for (const auto &col : join_node.columns) {
                if (col.resolution != ColumnResolution::MATERIALIZE)
                    continue;

                size_t mat_col_idx =
                    *out_result.materialized_map[col.original_idx];
                auto &out_col = out_result.materialized[mat_col_idx];

                // Determine source based on from_left and build/probe mapping
                // col.from_left refers to original left child
                // build_is_left tells us if build side is the left child
                // If from_left && build_is_left => from build
                // If from_left && !build_is_left => from probe (left became
                // probe)
                bool from_build = (col.from_left == build_is_left);
                uint32_t src_row = from_build ? build_row : probe_row;
                const auto &src_input = from_build ? build_input : probe_input;

                mema::value_t val;
                if (src_input.is_columnar()) {
                    const auto *table =
                        std::get<const ColumnarTable *>(src_input.data);
                    auto [actual_idx, _] =
                        src_input.node->output_attrs[col.child_output_idx];
                    val = columnar_reader.read_value(
                        table->columns[actual_idx], col.child_output_idx,
                        src_row, col.type, cursor, from_build);
                } else {
                    const auto &ir = std::get<DeferredResult>(src_input.data);
                    // Check if materialized in child
                    const auto *src_col =
                        ir.get_materialized(col.child_output_idx);
                    if (src_col) {
                        val = (*src_col)[src_row];
                    } else {
                        // Deferred - resolve via row ID to base table
                        // This should only happen if materialization wasn't
                        // propagated properly. Use direct read as fallback.
                        const auto *rowid_col =
                            ir.get_rowid_column(col.provenance.base_table_id);
                        if (rowid_col && deferred_plan.original_plan) {
                            uint32_t encoded = (*rowid_col)[src_row];
                            uint32_t base_row = GlobalRowId::row(encoded);
                            const auto &base_table =
                                deferred_plan.original_plan
                                    ->inputs[col.provenance.base_table_id];
                            val = columnar_reader.read_value_direct_public(
                                base_table
                                    .columns[col.provenance.base_column_idx],
                                base_row, col.type);
                        } else {
                            val = mema::value_t{mema::value_t::NULL_VALUE};
                        }
                    }
                }

                out_col.write_at(out_row, val);
            }

            // Write row ID columns
            for (size_t r = 0; r < num_rowid_cols; ++r) {
                const auto &rid_src = rowid_sources[r];
                auto &dest_rid_col = out_result.row_ids[r];

                uint32_t local_idx = rid_src.from_build ? build_row : probe_row;

                if (rid_src.needs_encode) {
                    dest_rid_col.write_at(
                        out_row,
                        GlobalRowId::encode(rid_src.table_id, local_idx));
                } else if (rid_src.source_col) {
                    dest_rid_col.write_at(out_row,
                                          (*rid_src.source_col)[local_idx]);
                }
            }
        }
    });
}

} // namespace materialize
} // namespace Contest

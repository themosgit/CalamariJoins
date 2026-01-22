/**
 * @file construct_intermediate.h
 * @brief Constructs intermediate results for multi-way joins.
 *
 * Allocates and populates IntermediateResult with only MATERIALIZE columns
 * (typically just the parent's join key). Deferred columns use per-table
 * 32-bit row ID storage for memory efficiency.
 *
 * Optimized with:
 * - Column-major iteration for cache locality
 * - Precomputed source metadata to avoid per-row variant access
 * - Per-table 32-bit row ID storage (vs per-column 64-bit provenance)
 * - Batch access to match collector chunks
 *
 * @see materialize.h for final resolution of deferred columns.
 */
#pragma once

#include <cstring>
#include <unordered_map>
#include <vector>

#include <data_access/columnar_reader.h>
#include <data_model/deferred_plan.h>
#include <data_model/intermediate.h>
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

// ============================================================================
// Row ID Batch Operations (for 32-bit per-table deferred)
// ============================================================================

namespace row_id_ops {

/**
 * @brief Write row IDs directly from columnar input.
 *
 * For columnar inputs, we just write the row_id directly (it's already
 * the base table row ID).
 */
inline size_t write_row_ids_direct(mema::DeferredTable &dest, size_t start_idx,
                                   const uint32_t *row_ids, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dest.write_at(start_idx + i, row_ids[i]);
    }
    return count;
}

/**
 * @brief Copy row IDs from child deferred table.
 *
 * For intermediate inputs, we look up the base table row ID from the
 * child's deferred table and copy it to the parent's deferred table.
 */
inline size_t copy_row_ids_from_child(mema::DeferredTable &dest,
                                      size_t start_idx,
                                      const mema::DeferredTable &src,
                                      const uint32_t *row_ids, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dest.write_at(start_idx + i, src[row_ids[i]]);
    }
    return count;
}

} // namespace row_id_ops

// ============================================================================
// Source Precomputation Structures
// ============================================================================

/**
 * @brief Precomputed metadata for a deferred table source.
 *
 * Groups columns by (from_build, base_table_id) so we only store 32-bit
 * row IDs once per unique base table instead of 64-bit provenance per column.
 */
struct DeferredTableSource {
    const mema::DeferredTable *child_table =
        nullptr;                ///< Source deferred table from child (if any).
    uint8_t base_table_id = 0;  ///< Base table ID.
    uint8_t dest_table_idx = 0; ///< Index in result.deferred_tables[].
    bool from_build = false;    ///< True if from build side.
    bool needs_direct = false;  ///< True if columnar (write row IDs directly).
};

/**
 * @brief Precomputed metadata for materialized column sources.
 *
 * Eliminates per-row std::variant access and conditional checks in hot loop.
 */
struct alignas(8) MaterializedColumnSource {
    const mema::column_t *intermediate_col =
        nullptr; ///< Source if from IntermediateResult materialized
    const Column *columnar_col = nullptr; ///< Source if from ColumnarTable
    const mema::DeferredTable *deferred_table =
        nullptr;                 ///< Source deferred table if needs resolution
    size_t child_output_idx = 0; ///< Index in child's output
    size_t mat_col_idx = 0;      ///< Index in result.materialized[]
    DataType type = DataType::INT32;
    uint8_t base_table_id = 0;           ///< For VARCHAR source tracking
    uint8_t base_column_idx = 0;         ///< For VARCHAR source tracking
    bool is_columnar = false;            ///< True if source is ColumnarTable
    bool from_build = false;             ///< True if from build side
    bool needs_deferred_resolve = false; ///< True if child deferred this column
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Collect columns needed from a JoinInput for page index building.
 */
inline platform::ArenaVector<const Column *>
collect_input_columns(const JoinInput &input,
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
 * @brief Prepare ColumnarReader for intermediate construction.
 *
 * Sets up page indices for columns that need to be read from columnar inputs.
 */
inline void prepare_intermediate_columns(
    ColumnarReader &reader, const JoinInput &build_input,
    const JoinInput &probe_input, const AnalyzedJoinNode &join_node,
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
            collect_input_columns(build_input, build_needed, arena));
    }

    if (probe_is_columnar) {
        reader.prepare_probe(
            collect_input_columns(probe_input, probe_needed, arena));
    }
}

/**
 * @brief Create empty intermediate result with proper schema.
 */
inline IntermediateResult
create_empty_intermediate_result(const AnalyzedJoinNode &node) {
    IntermediateResult result;
    result.node_info = &node;
    result.num_rows = 0;
    result.materialized_map.resize(node.columns.size(), std::nullopt);
    result.deferred_map.resize(node.columns.size(), std::nullopt);

    size_t mat_count = 0;
    for (const auto &col : node.columns) {
        if (col.resolution == ColumnResolution::MATERIALIZE) {
            result.materialized_map[col.original_idx] = mat_count++;
        }
        // For empty result, we don't need to set up deferred tables
    }
    result.materialized.resize(mat_count);

    return result;
}

/**
 * @brief Prepare deferred table sources for intermediate construction.
 *
 * Groups deferred columns by (from_build, base_table_id) to create
 * DeferredTable entries. Returns list of sources for populating the tables.
 */
inline std::vector<DeferredTableSource>
prepare_deferred_table_sources(const AnalyzedJoinNode &join_node,
                               const JoinInput &build_input,
                               const JoinInput &probe_input, bool build_is_left,
                               IntermediateResult &out_result) {
    // Map from (from_build << 8 | base_table_id) -> dest_table_idx
    std::unordered_map<uint16_t, uint8_t> table_key_to_idx;
    std::vector<DeferredTableSource> sources;

    for (const auto &col : join_node.columns) {
        if (col.resolution != ColumnResolution::DEFER)
            continue;

        bool from_build = (col.from_left == build_is_left);
        uint16_t key = (static_cast<uint16_t>(from_build) << 8) |
                       col.provenance.base_table_id;

        auto it = table_key_to_idx.find(key);
        uint8_t dest_idx;

        if (it == table_key_to_idx.end()) {
            // New deferred table needed
            dest_idx = static_cast<uint8_t>(out_result.deferred_tables.size());
            table_key_to_idx[key] = dest_idx;

            mema::DeferredTable dt;
            dt.base_table_id = col.provenance.base_table_id;
            dt.from_build = from_build;
            out_result.deferred_tables.push_back(std::move(dt));

            // Create source entry
            DeferredTableSource src;
            src.base_table_id = col.provenance.base_table_id;
            src.dest_table_idx = dest_idx;
            src.from_build = from_build;

            const auto &src_input = from_build ? build_input : probe_input;
            if (src_input.is_columnar()) {
                src.needs_direct = true;
                src.child_table = nullptr;
            } else {
                // Find child's deferred table for this base table
                const auto *child_ref =
                    src_input.get_deferred_ref(col.child_output_idx);
                if (child_ref) {
                    src.needs_direct = false;
                    src.child_table =
                        src_input.get_deferred_table(col.child_output_idx);
                } else {
                    // Child materialized this, shouldn't happen for DEFER cols
                    src.needs_direct = true;
                    src.child_table = nullptr;
                }
            }
            sources.push_back(src);
        } else {
            dest_idx = it->second;
        }

        // Add column to deferred table's column list
        out_result.deferred_tables[dest_idx].column_indices.push_back(
            col.provenance.base_column_idx);

        // Set up deferred_map entry
        DeferredColumnRef ref;
        ref.table_idx = dest_idx;
        ref.base_col = col.provenance.base_column_idx;
        out_result.deferred_map[col.original_idx] = ref;
    }

    return sources;
}

/**
 * @brief Precompute materialized column sources for column-major iteration.
 *
 * For each MATERIALIZE column, determines source type and caches pointers
 * to avoid per-row std::variant access in the hot loop.
 */
inline std::vector<MaterializedColumnSource>
prepare_materialized_sources(const AnalyzedJoinNode &join_node,
                             const JoinInput &build_input,
                             const JoinInput &probe_input, bool build_is_left) {
    std::vector<MaterializedColumnSource> sources;
    sources.reserve(join_node.columns.size());

    size_t mat_idx = 0;
    for (const auto &col : join_node.columns) {
        if (col.resolution != ColumnResolution::MATERIALIZE)
            continue;

        MaterializedColumnSource src;
        src.mat_col_idx = mat_idx++;
        src.child_output_idx = col.child_output_idx;
        src.type = col.type;
        src.base_table_id = col.provenance.base_table_id;
        src.base_column_idx = col.provenance.base_column_idx;
        src.from_build = (col.from_left == build_is_left);

        const auto &src_input = src.from_build ? build_input : probe_input;

        if (src_input.is_columnar()) {
            src.is_columnar = true;
            const auto *table = std::get<const ColumnarTable *>(src_input.data);
            auto [actual_idx, _] =
                src_input.node->output_attrs[col.child_output_idx];
            src.columnar_col = &table->columns[actual_idx];
        } else {
            src.is_columnar = false;
            const auto &ir = std::get<IntermediateResult>(src_input.data);

            if (ir.is_materialized(col.child_output_idx)) {
                src.intermediate_col =
                    ir.get_materialized(col.child_output_idx);
            } else if (ir.is_deferred(col.child_output_idx)) {
                src.needs_deferred_resolve = true;
                src.deferred_table =
                    ir.get_deferred_table(col.child_output_idx);
                // base_column_idx is already set from col.provenance
            }
        }
        sources.push_back(src);
    }

    return sources;
}

// ============================================================================
// Main Construction Function
// ============================================================================

/**
 * @brief Constructs intermediate result from thread-local buffers.
 *
 * Optimized with column-major iteration and per-table 32-bit row ID storage.
 * Only materializes columns marked MATERIALIZE in the AnalyzedJoinNode.
 * Deferred columns share row ID storage per unique base table.
 *
 * @tparam Mode            Collection mode for compile-time specialization.
 * @param buffers          Thread-local match buffers from probe.
 * @param build_input      Build side data source.
 * @param probe_input      Probe side data source.
 * @param join_node        Analyzed join node with materialization decisions.
 * @param remapped_attrs   Output attributes (after build/probe remapping).
 * @param build_output_size Number of columns from build side.
 * @param build_is_left    True if build side is the original left child.
 * @param columnar_reader  Reader for columnar data access.
 * @param out_result       Output IntermediateResult (populated in-place).
 * @param analyzed_plan    Full analyzed plan for base table access.
 */
template <MatchCollectionMode Mode>
void construct_intermediate_from_buffers(
    std::vector<ThreadLocalMatchBuffer<Mode>> &buffers,
    const JoinInput &build_input, const JoinInput &probe_input,
    const AnalyzedJoinNode &join_node,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    size_t build_output_size, bool build_is_left,
    ColumnarReader &columnar_reader, IntermediateResult &out_result,
    const AnalyzedPlan &analyzed_plan) {

    // Count total matches and compute buffer start offsets
    size_t total_matches = 0;
    std::vector<size_t> buffer_starts(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i) {
        buffer_starts[i] = total_matches;
        total_matches += buffers[i].count();
    }

    if (total_matches == 0) {
        out_result = create_empty_intermediate_result(join_node);
        return;
    }

    // Initialize result metadata
    out_result.node_info = &join_node;
    out_result.num_rows = total_matches;
    out_result.materialized_map.resize(join_node.columns.size(), std::nullopt);
    out_result.deferred_map.resize(join_node.columns.size(), std::nullopt);

    // Count materialized columns and set up maps
    size_t mat_count = 0;
    for (const auto &col : join_node.columns) {
        if (col.resolution == ColumnResolution::MATERIALIZE) {
            out_result.materialized_map[col.original_idx] = mat_count++;
        }
    }

    // Prepare deferred table sources (this populates deferred_tables and
    // deferred_map)
    auto deferred_sources = prepare_deferred_table_sources(
        join_node, build_input, probe_input, build_is_left, out_result);

    // Precompute materialized sources
    auto mat_sources = prepare_materialized_sources(join_node, build_input,
                                                    probe_input, build_is_left);

    // Pre-allocate pages
    using Page = mema::column_t::Page;
    using DeferredPage = mema::DeferredTable::Page;
    size_t mat_pages_needed =
        (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;
    size_t def_pages_needed =
        (total_matches + mema::DeferredTable::ENTRIES_PER_PAGE - 1) /
        mema::DeferredTable::ENTRIES_PER_PAGE;

    out_result.materialized.resize(mat_count);
    for (size_t c = 0; c < mat_count; ++c) {
        out_result.materialized[c].pages.resize(mat_pages_needed);
        out_result.materialized[c].set_row_count(total_matches);
    }

    for (auto &dt : out_result.deferred_tables) {
        dt.pages.resize(def_pages_needed);
        dt.set_row_count(total_matches);
    }

    // Set source metadata for materialized columns
    for (const auto &src : mat_sources) {
        out_result.materialized[src.mat_col_idx].source_table =
            src.base_table_id;
        out_result.materialized[src.mat_col_idx].source_column =
            src.base_column_idx;
    }

    const size_t num_threads = THREAD_COUNT;
    const size_t num_deferred_tables = out_result.deferred_tables.size();

    // Parallel page allocation
    worker_pool().execute([&](size_t t) {
        for (size_t c = 0; c < mat_count; ++c) {
            auto &col = out_result.materialized[c];
            for (size_t p = t; p < mat_pages_needed; p += num_threads) {
                void *ptr =
                    Contest::platform::get_arena(t)
                        .alloc_chunk<Contest::platform::ChunkType::IR_PAGE>();
                col.pages[p] = reinterpret_cast<Page *>(ptr);
            }
        }
        for (size_t d = 0; d < num_deferred_tables; ++d) {
            auto &dt = out_result.deferred_tables[d];
            for (size_t p = t; p < def_pages_needed; p += num_threads) {
                // Use IR_PAGE (16KB) for DeferredTable pages
                void *ptr =
                    Contest::platform::get_arena(t)
                        .alloc_chunk<Contest::platform::ChunkType::IR_PAGE>();
                dt.pages[p] = reinterpret_cast<DeferredPage *>(ptr);
            }
        }
    });

    // ========================================================================
    // COLUMN-MAJOR PARALLEL POPULATION
    // ========================================================================
    worker_pool().execute([&](size_t t) {
        if (t >= buffers.size())
            return;
        auto &buf = buffers[t];
        size_t my_count = buf.count();
        if (my_count == 0)
            return;

        size_t start = buffer_starts[t];
        ColumnarReader::Cursor cursor;

        // ====================================================================
        // Process MATERIALIZED columns (column-major for cache locality)
        // ====================================================================
        for (const auto &src : mat_sources) {
            auto &dest_col = out_result.materialized[src.mat_col_idx];

            // Get appropriate range based on which side this column comes from
            auto range = src.from_build ? buf.left_range() : buf.right_range();

            if (src.is_columnar) {
                // Columnar source - use ColumnarReader with cursor caching
                const auto &col = *src.columnar_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    dest_col.write_at(k++,
                                      columnar_reader.read_value(
                                          col, src.child_output_idx, rid,
                                          src.type, cursor, src.from_build));
                }
            } else if (src.intermediate_col) {
                // Intermediate materialized source - direct copy
                const auto &vec = *src.intermediate_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    dest_col.write_at(k++, vec[rid]);
                }
            } else if (src.needs_deferred_resolve && src.deferred_table) {
                // Deferred in child - resolve via deferred table + base table
                const auto &def_table = *src.deferred_table;
                size_t k = start;
                for (uint32_t rid : range) {
                    uint32_t base_row = def_table[rid];

                    if (analyzed_plan.original_plan) [[likely]] {
                        const auto &base_table =
                            analyzed_plan.original_plan
                                ->inputs[src.base_table_id];
                        mema::value_t val =
                            columnar_reader.read_value_direct_public(
                                base_table.columns[src.base_column_idx],
                                base_row, src.type);
                        dest_col.write_at(k++, val);
                    } else {
                        dest_col.write_at(
                            k++, mema::value_t{mema::value_t::NULL_VALUE});
                    }
                }
            }
        }

        // ====================================================================
        // Process DEFERRED tables (one pass per unique base table)
        // ====================================================================
        for (const auto &def_src : deferred_sources) {
            auto &dest_table =
                out_result.deferred_tables[def_src.dest_table_idx];

            auto batch_reader = def_src.from_build ? buf.left_batch_reader()
                                                   : buf.right_batch_reader();

            size_t k = start;
            while (batch_reader.has_more()) {
                size_t batch_count;
                const uint32_t *row_ids =
                    batch_reader.get_batch(256, batch_count);

                if (batch_count > 0) {
                    if (def_src.needs_direct) {
                        // Columnar input: write row IDs directly
                        row_id_ops::write_row_ids_direct(dest_table, k, row_ids,
                                                         batch_count);
                    } else if (def_src.child_table) {
                        // Intermediate input: copy from child's deferred table
                        row_id_ops::copy_row_ids_from_child(
                            dest_table, k, *def_src.child_table, row_ids,
                            batch_count);
                    }
                    k += batch_count;
                }
            }
        }
    });
}

} // namespace materialize
} // namespace Contest

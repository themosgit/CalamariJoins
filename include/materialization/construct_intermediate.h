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
#include <join_execution/join_setup.h>
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
        nullptr; ///< Source deferred table if needs resolution
    const mema::key_row_column_t *tuple_col =
        nullptr;                 ///< Source if from child's join_key_tuples
    size_t child_output_idx = 0; ///< Index in child's output
    size_t mat_col_idx = 0;      ///< Index in result.materialized[]
    DataType type = DataType::INT32;
    uint8_t base_table_id = 0;           ///< For VARCHAR source tracking
    uint8_t base_column_idx = 0;         ///< For VARCHAR source tracking
    bool is_columnar = false;            ///< True if source is ColumnarTable
    bool from_build = false;             ///< True if from build side
    bool needs_deferred_resolve = false; ///< True if child deferred this column
    bool needs_tuple_key_read = false;   ///< True if reading key from tuples
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
 * If parent_key_idx is provided, also prepares the join key column for tuple
 * population.
 */
inline void prepare_intermediate_columns(
    ColumnarReader &reader, const JoinInput &build_input,
    const JoinInput &probe_input, const AnalyzedJoinNode &join_node,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    size_t build_size, bool build_is_left,
    std::optional<size_t> parent_key_idx = std::nullopt) {

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

    // If parent needs a join key via tuples, mark that column as needed too
    // This ensures page indices are prepared for efficient tuple population
    if (parent_key_idx.has_value()) {
        for (const auto &col : join_node.columns) {
            if (col.original_idx == *parent_key_idx) {
                bool from_build = (col.from_left == build_is_left);
                if (from_build && col.child_output_idx < build_needed.size()) {
                    build_needed[col.child_output_idx] = 1;
                } else if (!from_build &&
                           col.child_output_idx < probe_needed.size()) {
                    probe_needed[col.child_output_idx] = 1;
                }
                break;
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
                const auto &child_ir =
                    std::get<IntermediateResult>(src_input.data);
                // Find child's deferred table for this base table
                const auto *child_ref =
                    src_input.get_deferred_ref(col.child_output_idx);
                if (child_ref) {
                    src.needs_direct = false;
                    src.child_table =
                        src_input.get_deferred_table(col.child_output_idx);
                } else if (child_ir.is_join_key(col.child_output_idx)) {
                    // Child stored this as tuples - the row_id in tuples
                    // is an IR index, but we need base table row IDs for
                    // deferred resolution. This shouldn't happen if the
                    // join key column is properly excluded from DEFER.
                    std::fprintf(stderr,
                                 "[BUG] DEFER column %zu is child's "
                                 "join key - this is unexpected!\n",
                                 col.child_output_idx);
                    src.needs_direct = true;
                    src.child_table = nullptr;
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

            // Check source type in priority order:
            // 1. Tuples (join key stored as key-row pairs)
            // 2. Materialized column
            // 3. Deferred table
            if (ir.is_join_key(col.child_output_idx)) {
                // Child stored this column as tuples - read key from there
                src.needs_tuple_key_read = true;
                src.tuple_col = &(*ir.join_key_tuples);
            } else if (ir.is_materialized(col.child_output_idx)) {
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
            } else if (src.needs_tuple_key_read && src.tuple_col) {
                // Child stored this column as tuples - read key from there
                const auto &tuples = *src.tuple_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    int32_t key = tuples.key_at(rid);
                    dest_col.write_at(k++, mema::value_t{key});
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

// ============================================================================
// Tuple-Based Intermediate Construction
// ============================================================================

/**
 * @brief Resolves a row ID to base table row ID if possible.
 *
 * For columnar inputs: row ID is already base row ID (direct).
 * For IR with tuples storing base rows: lookup via key_row_column_t.
 * For IR with tuples storing IR indices: lookup via deferred table.
 * For IR without tuples: lookup via deferred table.
 *
 * @param input The JoinInput to resolve from.
 * @param row_id The row ID from match buffer.
 * @param key_col_idx The join key column index in input's output.
 * @return Resolved base table row ID.
 */
inline uint32_t resolve_to_base_row(const JoinInput &input, uint32_t row_id,
                                    size_t key_col_idx) {
    if (input.is_columnar()) {
        // Columnar input: row ID is already base table row
        return row_id;
    }

    const auto &ir = std::get<IntermediateResult>(input.data);

    if (ir.has_join_key_tuples() && ir.join_key_has_base_rows()) {
        // IR stores base row IDs in tuples - one lookup
        return ir.join_key_tuples->row_id_at(row_id);
    }

    // IR stores IR indices - need deferred table lookup
    const auto *def_table = ir.get_deferred_table(key_col_idx);
    if (def_table) {
        return (*def_table)[row_id];
    }

    // Fallback: return as-is (shouldn't happen for correct plans)
    return row_id;
}

/**
 * @brief Populates join key tuples column from match buffers.
 *
 * Extracts join keys and resolves row IDs based on tracking configuration.
 * For tracked side with base rows, embeds base table row IDs directly.
 * For non-tracked side, embeds IR indices for later DeferredTable lookup.
 *
 * @tparam Mode Match collection mode.
 * @param buffers Thread-local match buffers.
 * @param buffer_starts Per-buffer write offsets.
 * @param build_input Build side input.
 * @param probe_input Probe side input.
 * @param key_from_build True if parent's join key comes from build side.
 * @param key_child_output_idx Column index in the key input's output.
 * @param out_tuples Output tuple column (pre-allocated).
 * @param columnar_reader Reader for columnar access.
 */
template <MatchCollectionMode Mode>
void populate_join_key_tuples(
    std::vector<ThreadLocalMatchBuffer<Mode>> &buffers,
    const std::vector<size_t> &buffer_starts, const JoinInput &build_input,
    const JoinInput &probe_input, bool key_from_build,
    size_t key_child_output_idx, mema::key_row_column_t &out_tuples,
    ColumnarReader &columnar_reader) {

    const JoinInput &key_input = key_from_build ? build_input : probe_input;
    size_t key_attr = key_child_output_idx;

    worker_pool().execute([&](size_t t) {
        if (t >= buffers.size())
            return;
        auto &buf = buffers[t];
        size_t my_count = buf.count();
        if (my_count == 0)
            return;

        size_t write_pos = buffer_starts[t];

        // Get the appropriate range based on which side provides the key
        auto range = key_from_build ? buf.left_range() : buf.right_range();

        if (key_input.is_columnar()) {
            // Columnar source - read key from base table using prepared page
            // index Store OUTPUT IR index (write_pos) so parent can use it to
            // index into this IR
            auto *table = std::get<const ColumnarTable *>(key_input.data);
            auto [actual_col_idx, _] = key_input.node->output_attrs[key_attr];
            const Column &col = table->columns[actual_col_idx];

            // Use cursor for efficient sequential/near-sequential access
            ColumnarReader::Cursor cursor;
            for (uint32_t row_id : range) {
                // Use read_value with prepared page index (O(1) amortized)
                // instead of read_value_direct_public (O(n) per read)
                int32_t key =
                    columnar_reader
                        .read_value(col, key_attr, row_id, DataType::INT32,
                                    cursor, key_from_build)
                        .value;
                // Store OUTPUT IR index (write_pos), not base table row_id
                // Parent needs IR index to access other columns in this IR
                uint32_t output_ir_idx = static_cast<uint32_t>(write_pos);
                out_tuples.write_at(write_pos++, {key, output_ir_idx});
            }
        } else {
            // Intermediate source - store OUTPUT IR index
            const auto &ir = std::get<IntermediateResult>(key_input.data);

            // Only propagate existing tuples if they contain the column we need
            // Otherwise, read from materialized column
            if (ir.has_join_key_tuples() && ir.join_key_idx.has_value() &&
                *ir.join_key_idx == key_attr) {
                // IR's tuples contain the column we need - propagate directly
                const auto &src_tuples = *ir.join_key_tuples;

                for (uint32_t ir_idx : range) {
                    mema::KeyRowPair src = src_tuples[ir_idx];
                    // Store OUTPUT IR index for parent to index into this IR
                    uint32_t output_ir_idx = static_cast<uint32_t>(write_pos);
                    out_tuples.write_at(write_pos++, {src.key, output_ir_idx});
                }
            } else {
                // IR's tuples contain a different column, or no tuples exist
                // Read from materialized column instead
                const auto *mat_col = ir.get_materialized(key_attr);
                if (mat_col) {
                    for (uint32_t ir_idx : range) {
                        int32_t key = (*mat_col)[ir_idx].value;
                        // Store OUTPUT IR index for parent to index into this
                        // IR
                        uint32_t output_ir_idx =
                            static_cast<uint32_t>(write_pos);
                        out_tuples.write_at(write_pos++, {key, output_ir_idx});
                    }
                }
            }
        }
    });
}

/**
 * @brief Constructs intermediate result with tuple-based join key storage.
 *
 * Stores join key as (value, row_id) tuples for accelerated hashtable build
 * and zero-indirection row ID propagation. Other columns handled normally
 * via deferred tables or materialization.
 *
 * @tparam Mode Collection mode for compile-time specialization.
 * @param buffers Thread-local match buffers from probe.
 * @param build_input Build side data source.
 * @param probe_input Probe side data source.
 * @param join_node Analyzed join node with materialization decisions.
 * @param config Build/probe configuration.
 * @param build_is_left True if build side is the original left child.
 * @param parent_key_idx Index of column that will be parent's join key.
 * @param columnar_reader Reader for columnar data access.
 * @param out_result Output IntermediateResult (populated in-place).
 * @param analyzed_plan Full analyzed plan for base table access.
 */
template <MatchCollectionMode Mode>
void construct_intermediate_with_tuples(
    std::vector<ThreadLocalMatchBuffer<Mode>> &buffers,
    const JoinInput &build_input, const JoinInput &probe_input,
    const AnalyzedJoinNode &join_node, const join::BuildProbeConfig &config,
    bool build_is_left, size_t parent_key_idx, ColumnarReader &columnar_reader,
    IntermediateResult &out_result, const AnalyzedPlan &analyzed_plan) {

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

    // Determine if parent's join key comes from build or probe side
    // and which base table it traces back to
    bool key_from_build = true;
    size_t key_child_output_idx = 0; // Column index in child's output
    uint8_t key_base_table_id = 0;
    uint8_t key_base_column = 0;

    for (const auto &col : join_node.columns) {
        if (col.original_idx == parent_key_idx) {
            key_from_build = (col.from_left == build_is_left);
            key_child_output_idx = col.child_output_idx;
            key_base_table_id = col.provenance.base_table_id;
            key_base_column = col.provenance.base_column_idx;
            break;
        }
    }

    // Allocate join key tuples column
    out_result.join_key_tuples.emplace();
    out_result.join_key_tuples->pre_allocate_from_arena(
        Contest::platform::get_arena(0), total_matches);
    out_result.join_key_tuples->base_table_id = key_base_table_id;
    out_result.join_key_tuples->source_column = key_base_column;
    // Always store OUTPUT IR indices (not base row IDs) so parent can
    // index into this IR to access deferred columns
    out_result.join_key_tuples->stores_base_row_ids = false;
    out_result.join_key_idx = parent_key_idx;
    const JoinInput &key_input = key_from_build ? build_input : probe_input;
    (void)key_input; // Used in populate_join_key_tuples

    // Count non-join-key materialized columns and set up maps
    size_t mat_count = 0;
    for (const auto &col : join_node.columns) {
        if (col.resolution == ColumnResolution::MATERIALIZE &&
            col.original_idx != parent_key_idx) {
            out_result.materialized_map[col.original_idx] = mat_count++;
        }
    }

    // Prepare deferred table sources (unchanged from non-tuple version)
    auto deferred_sources = prepare_deferred_table_sources(
        join_node, build_input, probe_input, build_is_left, out_result);

    // Precompute materialized sources (excluding join key)
    std::vector<MaterializedColumnSource> mat_sources;
    mat_sources.reserve(join_node.columns.size());
    size_t mat_idx = 0;
    for (const auto &col : join_node.columns) {
        if (col.resolution != ColumnResolution::MATERIALIZE)
            continue;
        if (col.original_idx == parent_key_idx)
            continue; // Skip join key - handled via tuples

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

            // Check source type in priority order:
            // 1. Tuples (join key stored as key-row pairs)
            // 2. Materialized column
            // 3. Deferred table
            if (ir.is_join_key(col.child_output_idx)) {
                // Child stored this column as tuples - read key from there
                src.needs_tuple_key_read = true;
                src.tuple_col = &(*ir.join_key_tuples);
            } else if (ir.is_materialized(col.child_output_idx)) {
                src.intermediate_col =
                    ir.get_materialized(col.child_output_idx);
            } else if (ir.is_deferred(col.child_output_idx)) {
                src.needs_deferred_resolve = true;
                src.deferred_table =
                    ir.get_deferred_table(col.child_output_idx);
            }
        }
        mat_sources.push_back(src);
    }

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
                void *ptr =
                    Contest::platform::get_arena(t)
                        .alloc_chunk<Contest::platform::ChunkType::IR_PAGE>();
                dt.pages[p] = reinterpret_cast<DeferredPage *>(ptr);
            }
        }
    });

    // Populate join key tuples
    populate_join_key_tuples<Mode>(
        buffers, buffer_starts, build_input, probe_input, key_from_build,
        key_child_output_idx, *out_result.join_key_tuples, columnar_reader);

    // Populate other materialized columns and deferred tables
    // (same logic as construct_intermediate_from_buffers)
    worker_pool().execute([&](size_t t) {
        if (t >= buffers.size())
            return;
        auto &buf = buffers[t];
        size_t my_count = buf.count();
        if (my_count == 0)
            return;

        size_t start = buffer_starts[t];
        ColumnarReader::Cursor cursor;

        // Process MATERIALIZED columns (excluding join key)
        for (const auto &src : mat_sources) {
            auto &dest_col = out_result.materialized[src.mat_col_idx];

            auto range = src.from_build ? buf.left_range() : buf.right_range();

            if (src.is_columnar) {
                const auto &col = *src.columnar_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    mema::value_t val = columnar_reader.read_value(
                        col, src.child_output_idx, rid, src.type, cursor,
                        src.from_build);
                    dest_col.write_at(k++, val);
                }
            } else if (src.needs_tuple_key_read && src.tuple_col) {
                // Child stored this column as tuples - read key from there
                const auto &tuples = *src.tuple_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    int32_t key = tuples.key_at(rid);
                    dest_col.write_at(k++, mema::value_t{key});
                }
            } else if (src.intermediate_col) {
                const auto &vec = *src.intermediate_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    mema::value_t val = vec[rid];
                    dest_col.write_at(k++, val);
                }
            } else if (src.needs_deferred_resolve && src.deferred_table) {
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

        // Process DEFERRED tables
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
                        row_id_ops::write_row_ids_direct(dest_table, k, row_ids,
                                                         batch_count);
                    } else if (def_src.child_table) {
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

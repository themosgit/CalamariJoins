/**
 * @file construct_intermediate.h
 * @brief Constructs intermediate results for multi-way joins.
 *
 * Allocates and populates IntermediateResult with only MATERIALIZE columns
 * (typically just the parent's join key). Deferred columns store 64-bit
 * provenance (table_id, column_idx, row_id) for resolution at final output.
 *
 * Optimized with:
 * - Column-major iteration for cache locality
 * - Precomputed source metadata to avoid per-row variant access
 * - SIMD provenance encoding (AVX2/NEON) for deferred columns
 * - Batch access to match collector chunks
 *
 * @see materialize.h for final resolution of deferred columns.
 */
#pragma once

#include <cstring>
#include <vector>

#include <data_access/columnar_reader.h>
#include <data_model/deferred_plan.h>
#include <data_model/intermediate.h>
#include <foundation/common.h>
#include <join_execution/match_collector.h>
#include <platform/arena.h>
#include <platform/worker_pool.h>

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace Contest {
namespace materialize {

using Contest::io::ColumnarReader;
using Contest::join::MatchCollectionMode;
using Contest::join::ThreadLocalMatchBuffer;
using Contest::platform::THREAD_COUNT;
using Contest::platform::worker_pool;

// ============================================================================
// SIMD Provenance Encoding
// ============================================================================

namespace simd_provenance {

#if defined(__x86_64__) && defined(__AVX2__)
inline constexpr size_t BATCH_SIZE = 4; ///< 4 x uint64_t in AVX2 (256-bit)
#elif defined(__aarch64__)
inline constexpr size_t BATCH_SIZE = 2; ///< 2 x uint64_t in NEON (128-bit)
#else
inline constexpr size_t BATCH_SIZE = 0; ///< No SIMD available
#endif

/**
 * @brief Encode provenance for batch of row IDs using SIMD.
 *
 * Encodes (table_id << 56) | (column_idx << 48) | row_id for each row.
 * Uses AVX2 on x86_64 or NEON on aarch64, with scalar fallback.
 *
 * @param dest       Destination deferred column
 * @param start_idx  Starting output index
 * @param row_ids    Pointer to row IDs (from IndexChunk, contiguous)
 * @param count      Number of row IDs to process
 * @param table_id   Base table ID (constant for all rows)
 * @param column_idx Base column index (constant for all rows)
 * @return Number of rows processed (always == count)
 */
inline size_t encode_provenance_batch(mema::deferred_column_t &dest,
                                      size_t start_idx, const uint32_t *row_ids,
                                      size_t count, uint8_t table_id,
                                      uint8_t column_idx) {
    // Precompute constant prefix: (table_id << 56) | (column_idx << 48)
    const uint64_t prefix = DeferredProvenance::encode(table_id, column_idx, 0);

    size_t i = 0;

#if defined(__x86_64__) && defined(__AVX2__)
    // AVX2: Process 4 x uint64_t at a time
    // Load 4 x uint32_t, zero-extend to 4 x uint64_t, OR with prefix
    const __m256i prefix_vec = _mm256_set1_epi64x(static_cast<int64_t>(prefix));

    for (; i + 4 <= count; i += 4) {
        // Load 4 x uint32_t and zero-extend to 4 x uint64_t
        __m128i rows_32 =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(row_ids + i));
        __m256i rows_64 = _mm256_cvtepu32_epi64(rows_32);

        // OR with prefix to create provenance values
        __m256i result = _mm256_or_si256(rows_64, prefix_vec);

        // Store to aligned buffer, then write individually (page-safe)
        alignas(32) uint64_t out[4];
        _mm256_store_si256(reinterpret_cast<__m256i *>(out), result);

        dest.write_at(start_idx + i, out[0]);
        dest.write_at(start_idx + i + 1, out[1]);
        dest.write_at(start_idx + i + 2, out[2]);
        dest.write_at(start_idx + i + 3, out[3]);
    }
#elif defined(__aarch64__)
    // NEON: Process 2 x uint64_t at a time
    const uint64x2_t prefix_vec = vdupq_n_u64(prefix);

    for (; i + 2 <= count; i += 2) {
        // Load 2 x uint32_t and zero-extend to 2 x uint64_t
        uint32x2_t rows_32 = vld1_u32(row_ids + i);
        uint64x2_t rows_64 = vmovl_u32(rows_32);

        // OR with prefix
        uint64x2_t result = vorrq_u64(rows_64, prefix_vec);

        // Store individually (page boundary safe)
        dest.write_at(start_idx + i, vgetq_lane_u64(result, 0));
        dest.write_at(start_idx + i + 1, vgetq_lane_u64(result, 1));
    }
#endif

    // Scalar remainder
    for (; i < count; ++i) {
        dest.write_at(start_idx + i,
                      prefix | static_cast<uint64_t>(row_ids[i]));
    }

    return count;
}

/**
 * @brief Copy provenance from source column using batch reads.
 *
 * Copies existing 64-bit provenance values from child intermediate.
 * Uses contiguous batch access for better cache behavior.
 *
 * @param dest       Destination deferred column
 * @param start_idx  Starting output index
 * @param src        Source deferred column (from child)
 * @param row_ids    Row indices into source column
 * @param count      Number of rows to copy
 * @return Number of rows processed (always == count)
 */
inline size_t copy_provenance_batch(mema::deferred_column_t &dest,
                                    size_t start_idx,
                                    const mema::deferred_column_t &src,
                                    const uint32_t *row_ids, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dest.write_at(start_idx + i, src[row_ids[i]]);
    }
    return count;
}

} // namespace simd_provenance

// ============================================================================
// Source Precomputation Structures
// ============================================================================

/**
 * @brief Precomputed metadata for deferred column sources.
 *
 * Tracks where each deferred column's provenance comes from:
 * - For columnar inputs: encode fresh (table_id, column_idx, row_id)
 * - For IntermediateResult inputs: copy existing provenance from child
 */
struct DeferredColumnSource {
    const mema::deferred_column_t *source_col =
        nullptr;                 ///< Source if from intermediate.
    uint8_t base_table_id = 0;   ///< Base table ID for encoding.
    uint8_t base_column_idx = 0; ///< Base column index for encoding.
    bool from_build = false;     ///< True if from build side.
    bool needs_encode = false;   ///< True if columnar (needs fresh encode).
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
    const mema::deferred_column_t *deferred_resolve_col =
        nullptr;                 ///< Source if needs deferred resolution
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
    size_t def_count = 0;
    for (const auto &col : node.columns) {
        if (col.resolution == ColumnResolution::MATERIALIZE) {
            result.materialized_map[col.original_idx] = mat_count++;
        } else {
            result.deferred_map[col.original_idx] = def_count++;
        }
    }
    result.materialized.resize(mat_count);
    result.deferred_columns.resize(def_count);

    return result;
}

/**
 * @brief Prepare deferred column sources for intermediate construction.
 */
inline std::vector<DeferredColumnSource>
prepare_deferred_sources(const AnalyzedJoinNode &join_node,
                         const JoinInput &build_input,
                         const JoinInput &probe_input, bool build_is_left) {
    std::vector<DeferredColumnSource> sources;
    sources.reserve(join_node.num_deferred_columns);

    for (const auto &col : join_node.columns) {
        if (col.resolution != ColumnResolution::DEFER)
            continue;

        DeferredColumnSource src;
        src.base_table_id = col.provenance.base_table_id;
        src.base_column_idx = col.provenance.base_column_idx;
        src.from_build = (col.from_left == build_is_left);

        const auto &src_input = src.from_build ? build_input : probe_input;

        if (src_input.is_columnar()) {
            src.needs_encode = true;
            src.source_col = nullptr;
        } else {
            const auto *child_def =
                src_input.get_deferred_column(col.child_output_idx);
            if (child_def) {
                src.needs_encode = false;
                src.source_col = child_def;
            } else {
                src.needs_encode = true;
                src.source_col = nullptr;
            }
        }
        sources.push_back(src);
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
                src.deferred_resolve_col =
                    ir.get_deferred(col.child_output_idx);
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
 * Optimized with column-major iteration and SIMD provenance encoding.
 * Only materializes columns marked MATERIALIZE in the AnalyzedJoinNode.
 * Deferred columns store 64-bit provenance encoding for resolution at final
 * output.
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

    size_t mat_count = 0;
    size_t def_count = 0;
    for (const auto &col : join_node.columns) {
        if (col.resolution == ColumnResolution::MATERIALIZE) {
            out_result.materialized_map[col.original_idx] = mat_count++;
        } else {
            out_result.deferred_map[col.original_idx] = def_count++;
        }
    }

    // Precompute sources for column-major iteration
    auto mat_sources = prepare_materialized_sources(join_node, build_input,
                                                    probe_input, build_is_left);
    auto deferred_sources = prepare_deferred_sources(
        join_node, build_input, probe_input, build_is_left);

    // Pre-allocate pages
    using Page = mema::column_t::Page;
    using DeferredPage = mema::deferred_column_t::Page;
    size_t mat_pages_needed =
        (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;
    size_t def_pages_needed =
        (total_matches + mema::deferred_column_t::ENTRIES_PER_PAGE - 1) /
        mema::deferred_column_t::ENTRIES_PER_PAGE;

    out_result.materialized.resize(mat_count);
    for (size_t c = 0; c < mat_count; ++c) {
        out_result.materialized[c].pages.resize(mat_pages_needed);
        out_result.materialized[c].set_row_count(total_matches);
    }

    out_result.deferred_columns.resize(def_count);
    for (size_t d = 0; d < def_count; ++d) {
        out_result.deferred_columns[d].pages.resize(def_pages_needed);
        out_result.deferred_columns[d].set_row_count(total_matches);
    }

    // Set source metadata for materialized columns
    for (const auto &src : mat_sources) {
        out_result.materialized[src.mat_col_idx].source_table =
            src.base_table_id;
        out_result.materialized[src.mat_col_idx].source_column =
            src.base_column_idx;
    }

    const size_t num_threads = THREAD_COUNT;

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
        for (size_t d = 0; d < def_count; ++d) {
            auto &def_col = out_result.deferred_columns[d];
            for (size_t p = t; p < def_pages_needed; p += num_threads) {
                void *ptr =
                    Contest::platform::get_arena(t)
                        .alloc_chunk<
                            Contest::platform::ChunkType::DEFERRED_PAGE>();
                def_col.pages[p] = reinterpret_cast<DeferredPage *>(ptr);
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
            } else if (src.needs_deferred_resolve && src.deferred_resolve_col) {
                // Deferred in child - resolve via provenance
                const auto &def_col = *src.deferred_resolve_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    uint64_t prov = def_col[rid];
                    uint8_t base_tid = DeferredProvenance::table(prov);
                    uint8_t base_col = DeferredProvenance::column(prov);
                    uint64_t base_row = DeferredProvenance::row(prov);

                    if (analyzed_plan.original_plan) [[likely]] {
                        const auto &base_table =
                            analyzed_plan.original_plan->inputs[base_tid];
                        mema::value_t val =
                            columnar_reader.read_value_direct_public(
                                base_table.columns[base_col],
                                static_cast<uint32_t>(base_row), src.type);
                        dest_col.write_at(k++, val);
                    } else {
                        dest_col.write_at(
                            k++, mema::value_t{mema::value_t::NULL_VALUE});
                    }
                }
            }
        }

        // ====================================================================
        // Process DEFERRED columns (column-major with SIMD batch encoding)
        // ====================================================================
        for (size_t d = 0; d < deferred_sources.size(); ++d) {
            const auto &def_src = deferred_sources[d];
            auto &dest_def_col = out_result.deferred_columns[d];

            if (def_src.needs_encode) {
                // Fresh encoding from columnar input - use SIMD batch
                auto batch_reader = def_src.from_build
                                        ? buf.left_batch_reader()
                                        : buf.right_batch_reader();

                size_t k = start;
                while (batch_reader.has_more()) {
                    size_t batch_count;
                    // Request larger batches for SIMD efficiency
                    constexpr size_t MAX_BATCH =
                        simd_provenance::BATCH_SIZE > 0 ? 64 : 256;
                    const uint32_t *row_ids =
                        batch_reader.get_batch(MAX_BATCH, batch_count);

                    if (batch_count > 0) {
                        simd_provenance::encode_provenance_batch(
                            dest_def_col, k, row_ids, batch_count,
                            def_src.base_table_id, def_src.base_column_idx);
                        k += batch_count;
                    }
                }
            } else if (def_src.source_col) {
                // Copy existing provenance from child intermediate
                auto batch_reader = def_src.from_build
                                        ? buf.left_batch_reader()
                                        : buf.right_batch_reader();

                size_t k = start;
                while (batch_reader.has_more()) {
                    size_t batch_count;
                    const uint32_t *row_ids =
                        batch_reader.get_batch(256, batch_count);

                    if (batch_count > 0) {
                        simd_provenance::copy_provenance_batch(
                            dest_def_col, k, *def_src.source_col, row_ids,
                            batch_count);
                        k += batch_count;
                    }
                }
            }
        }
    });
}

} // namespace materialize
} // namespace Contest

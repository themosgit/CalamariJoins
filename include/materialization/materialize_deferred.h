/**
 * @file materialize_deferred.h
 * @brief Final materialization for deferred execution path.
 *
 * Materializes all output columns at the root join, resolving deferred
 * columns by decoding 64-bit provenance (table_id, column_idx, row_id) back
 * to base tables.
 *
 * @see construct_deferred.h for building DeferredResult intermediates.
 * @see materialize.h for the eager materialization equivalent.
 */
#pragma once

#include <algorithm>
#include <cstring>
#include <functional>
#include <sys/mman.h>
#include <vector>

#include <data_access/columnar_reader.h>
#include <data_model/deferred_intermediate.h>
#include <data_model/deferred_plan.h>
#include <foundation/common.h>
#include <join_execution/match_collector.h>
#include <materialization/page_builders.h>
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
 * @brief Collect columns needed from a DeferredInput for final materialization.
 */
inline platform::ArenaVector<const Column *>
collect_final_columns(const DeferredInput &input,
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
 * @brief Prepare ColumnarReader for final deferred materialization at root.
 *
 * Sets up page indices for ALL output columns (since all need materialization
 * at root).
 */
inline void prepare_final_deferred_columns(
    ColumnarReader &reader, const DeferredInput &build_input,
    const DeferredInput &probe_input, const DeferredJoinNode &join_node,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    size_t build_size, bool build_is_left) {

    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();

    if (!build_is_columnar && !probe_is_columnar)
        return;

    auto &arena = Contest::platform::get_arena(0);

    // All output columns needed at root
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

    // Mark ALL columns needed for final materialization
    // from_left refers to original left child
    // build_is_left tells us if build side is the left child
    for (const auto &col : join_node.columns) {
        bool from_build = (col.from_left == build_is_left);
        if (from_build && col.child_output_idx < build_needed.size()) {
            build_needed[col.child_output_idx] = 1;
        } else if (!from_build && col.child_output_idx < probe_needed.size()) {
            probe_needed[col.child_output_idx] = 1;
        }
    }

    if (build_is_columnar) {
        reader.prepare_build(
            collect_final_columns(build_input, build_needed, arena));
    }

    if (probe_is_columnar) {
        reader.prepare_probe(
            collect_final_columns(probe_input, probe_needed, arena));
    }
}

/**
 * @brief Create empty result for zero-match case in deferred path.
 */
inline ColumnarTable create_empty_deferred_final(
    const std::vector<std::tuple<size_t, DataType>> &output_attrs) {
    ColumnarTable empty_result;
    empty_result.num_rows = 0;
    for (auto [_, data_type] : output_attrs) {
        empty_result.columns.emplace_back(data_type);
    }
    return empty_result;
}

/**
 * @brief Materialize a single column from deferred sources.
 *
 * Handles three cases:
 * 1. COLUMNAR_DIRECT: Input is columnar, read directly via row index
 * 2. MATERIALIZED: Column was materialized in DeferredResult
 * 3. DEFERRED: Resolve via 64-bit provenance to base table
 *
 * @tparam Mode Collection mode for compile-time specialization.
 * @tparam BuilderType Int32PageBuilder or VarcharPageBuilder.
 * @tparam ReaderFunc Callable: (row_idx, cursor) -> value_t.
 * @tparam InitBuilderFunc Callable: (page_allocator) -> BuilderType.
 */
template <MatchCollectionMode Mode, typename BuilderType, typename ReaderFunc,
          typename InitBuilderFunc>
inline void materialize_deferred_column(
    Column &dest_col, std::vector<ThreadLocalMatchBuffer<Mode>> &buffers,
    size_t total_matches, ReaderFunc &&read_value,
    InitBuilderFunc &&init_builder, bool from_build, size_t est_bytes_per_row) {

    if (total_matches == 0)
        return;

    const int num_threads = THREAD_COUNT;

    size_t matches_per_thread = (total_matches + num_threads - 1) / num_threads;
    size_t usable_per_page = PAGE_SIZE - 256;
    size_t rows_per_page = std::max(1ul, usable_per_page / est_bytes_per_row);
    size_t pages_per_thread =
        (matches_per_thread + rows_per_page - 1) / rows_per_page + 10;
    size_t total_pages = pages_per_thread * num_threads;

    void *page_memory =
        mmap(nullptr, total_pages * PAGE_SIZE, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page_memory == MAP_FAILED)
        throw std::bad_alloc();

    std::vector<Column> thread_columns;
    thread_columns.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        thread_columns.emplace_back(dest_col.type);
    }

    worker_pool().execute([&](size_t t) {
        if (t >= buffers.size())
            return;
        auto &buf = buffers[t];
        size_t my_count = buf.count();
        if (my_count == 0)
            return;

        Column &local_col = thread_columns[t];

        size_t thread_page_start = t * pages_per_thread;
        size_t thread_page_limit = pages_per_thread;
        size_t used_pages = 0;

        ColumnarReader::Cursor cursor;

        auto page_allocator = [&]() -> Page * {
            Page *p;
            if (used_pages < thread_page_limit) {
                p = reinterpret_cast<Page *>(static_cast<char *>(page_memory) +
                                             (thread_page_start + used_pages) *
                                                 PAGE_SIZE);
                used_pages++;
            } else {
                p = new Page();
            }
            local_col.pages.push_back(p);
            return p;
        };

        BuilderType builder = init_builder(page_allocator);
        builder.prepare(my_count);

        const size_t check_interval = BuilderType::MIN_ROWS_PER_PAGE_CHECK;
        size_t rows_since_check = 0;

        auto range = from_build ? buf.left_range() : buf.right_range();

        for (uint32_t row_id : range) {
            bool flushed = builder.add(read_value(row_id, cursor));

            if (flushed) {
                rows_since_check = 0;
            } else {
                rows_since_check++;
                if (rows_since_check >= check_interval) {
                    if (builder.should_check_overflow()) {
                        builder.save_to_page(builder.current_page);
                        rows_since_check = 0;
                    }
                    if (rows_since_check > check_interval * 2)
                        rows_since_check = 0;
                }
            }
        }

        if (builder.num_rows != 0) {
            builder.save_to_page(builder.current_page);
        }
    });

    for (auto &thread_col : thread_columns) {
        for (auto *page : thread_col.pages) {
            dest_col.pages.push_back(page);
        }
        thread_col.pages.clear();
    }

    auto *mapped_mem = new MappedMemory(page_memory, total_pages * PAGE_SIZE);
    dest_col.assign_mapped_memory(mapped_mem);
}

/**
 * @brief Materialize single output column handling deferred resolution.
 *
 * For deferred columns, resolves via 64-bit provenance encoding back to
 * base table.
 *
 * @tparam Mode Collection mode for compile-time specialization.
 */
template <MatchCollectionMode Mode>
inline void materialize_single_deferred_column(
    Column &dest_col, size_t col_idx, size_t build_size, bool build_is_left,
    std::vector<ThreadLocalMatchBuffer<Mode>> &buffers, size_t total_matches,
    const DeferredInput &build_input, const DeferredInput &probe_input,
    const DeferredJoinNode &join_node, ColumnarReader &columnar_reader,
    const DeferredPlan &deferred_plan) {

    // Find column info
    const DeferredColumnInfo *col_info = nullptr;
    for (const auto &col : join_node.columns) {
        if (col.original_idx == col_idx) {
            col_info = &col;
            break;
        }
    }

    if (!col_info) {
        // Fallback - shouldn't happen
        return;
    }

    // Determine if this column comes from build or probe side at runtime
    bool from_build = (col_info->from_left == build_is_left);
    const DeferredInput &src_input = from_build ? build_input : probe_input;

    // Determine how to read the value
    const Column *columnar_source = nullptr;
    const mema::column_t *materialized_source = nullptr;
    const mema::deferred_column_t *deferred_source = nullptr;

    if (src_input.is_columnar()) {
        // Direct columnar read
        const auto *table = std::get<const ColumnarTable *>(src_input.data);
        auto [actual_idx, _] =
            src_input.node->output_attrs[col_info->child_output_idx];
        columnar_source = &table->columns[actual_idx];
    } else {
        const auto &ir = std::get<DeferredResult>(src_input.data);
        if (ir.is_materialized(col_info->child_output_idx)) {
            // Read from materialized column
            materialized_source =
                ir.get_materialized(col_info->child_output_idx);
        } else if (ir.is_deferred(col_info->child_output_idx)) {
            // Deferred - need to resolve via 64-bit provenance
            deferred_source = ir.get_deferred(col_info->child_output_idx);
        }
    }

    // Create reader lambda
    auto reader = [&](uint32_t local_row_id,
                      ColumnarReader::Cursor &cursor) -> mema::value_t {
        if (columnar_source) {
            return columnar_reader.read_value(
                *columnar_source, col_info->child_output_idx, local_row_id,
                col_info->type, cursor, from_build);
        } else if (materialized_source) {
            return (*materialized_source)[local_row_id];
        } else if (deferred_source && deferred_plan.original_plan) {
            // Deferred resolution: decode 64-bit provenance
            uint64_t prov = (*deferred_source)[local_row_id];
            uint8_t base_tid = DeferredProvenance::table(prov);
            uint8_t base_col = DeferredProvenance::column(prov);
            uint64_t base_row = DeferredProvenance::row(prov);
            const auto &base_table =
                deferred_plan.original_plan->inputs[base_tid];
            return columnar_reader.read_value(
                base_table.columns[base_col], base_col,
                static_cast<uint32_t>(base_row), col_info->type, cursor, true);
        }
        return mema::value_t{mema::value_t::NULL_VALUE};
    };

    // Materialize based on type
    if (dest_col.type == DataType::INT32) {
        auto init = [](std::function<Page *()> alloc) {
            return Int32PageBuilder(std::move(alloc));
        };
        materialize_deferred_column<Mode, Int32PageBuilder>(
            dest_col, buffers, total_matches,
            [&](uint32_t rid, ColumnarReader::Cursor &cursor) {
                return reader(rid, cursor);
            },
            init, from_build, 4);
        return;
    }

    // VARCHAR
    const Column *str_src_ptr = columnar_source;
    if (!str_src_ptr) {
        if (materialized_source) {
            str_src_ptr = &deferred_plan.original_plan
                               ->inputs[materialized_source->source_table]
                               .columns[materialized_source->source_column];
        } else if (deferred_source && deferred_plan.original_plan) {
            // For deferred VARCHAR, get source from provenance of first row
            // All rows in a deferred column share the same base table/column
            str_src_ptr = &deferred_plan.original_plan
                               ->inputs[col_info->provenance.base_table_id]
                               .columns[col_info->provenance.base_column_idx];
        }
    }

    if (!str_src_ptr) {
        // Shouldn't happen, but handle gracefully
        return;
    }

    auto init = [str_src_ptr](std::function<Page *()> alloc) {
        return VarcharPageBuilder(*str_src_ptr, std::move(alloc));
    };

    materialize_deferred_column<Mode, VarcharPageBuilder>(
        dest_col, buffers, total_matches,
        [&](uint32_t rid, ColumnarReader::Cursor &cursor) {
            return reader(rid, cursor);
        },
        init, from_build, 35);
}

/**
 * @brief Materialize all output columns from deferred intermediate.
 *
 * For root join in deferred execution path. Resolves all deferred columns
 * by decoding 64-bit provenance to base tables.
 *
 * @tparam Mode Collection mode for compile-time specialization.
 * @param buffers Thread-local match buffers from probe.
 * @param build_input Build side deferred input.
 * @param probe_input Probe side deferred input.
 * @param join_node Deferred join node with column info.
 * @param remapped_attrs Output projection after build/probe remapping.
 * @param build_size Number of columns from build side.
 * @param columnar_reader Reader for columnar data.
 * @param deferred_plan Full deferred plan for base table access.
 * @return ColumnarTable with final output.
 */
template <MatchCollectionMode Mode>
inline ColumnarTable materialize_deferred_from_buffers(
    std::vector<ThreadLocalMatchBuffer<Mode>> &buffers,
    const DeferredInput &build_input, const DeferredInput &probe_input,
    const DeferredJoinNode &join_node,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    size_t build_size, bool build_is_left, ColumnarReader &columnar_reader,
    const DeferredPlan &deferred_plan) {

    // Compute total matches
    size_t total_matches = 0;
    for (const auto &buf : buffers) {
        total_matches += buf.count();
    }

    if (total_matches == 0) {
        return create_empty_deferred_final(remapped_attrs);
    }

    ColumnarTable result;
    result.num_rows = total_matches;

    for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
        auto [col_idx, data_type] = remapped_attrs[out_idx];
        result.columns.emplace_back(data_type);
        Column &dest_col = result.columns.back();

        // Pass out_idx (output position) not col_idx (global column index)
        // because materialize_single_deferred_column searches by original_idx
        // which is the output position in join_node.columns
        materialize_single_deferred_column<Mode>(
            dest_col, out_idx, build_size, build_is_left, buffers,
            total_matches, build_input, probe_input, join_node, columnar_reader,
            deferred_plan);
    }

    return result;
}

} // namespace materialize
} // namespace Contest

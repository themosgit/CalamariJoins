/**
 * @file materialize.h
 * @brief Materialization of join results into ColumnarTable format.
 *
 * Parallel materialization using per-thread page builders and mmap allocation.
 */
#pragma once

#include <algorithm>
#include <cstring>
#include <data_access/columnar_reader.h>
#include <data_model/intermediate.h>
#include <data_model/plan.h>
#include <functional>
#include <join_execution/join_setup.h>
#include <join_execution/match_collector.h>
#include <materialization/construct_intermediate.h>
#include <materialization/page_builders.h>
#include <platform/worker_pool.h>
#include <sys/mman.h>
#include <vector>

/**
 * @namespace Contest::materialize
 * @brief Materialization of join results into columnar format.
 *
 * Key components in this file:
 * - materialize_column<>(): Parallel page construction with mmap'd memory pool
 * - materialize_single_column(): Dispatcher selecting builder type and source
 * - materialize(): Full ColumnarTable construction from join matches
 * - create_empty_result(): Zero-match fast path
 *
 * @see page_builders.h for Int32PageBuilder/VarcharPageBuilder
 * @see construct_intermediate.h for non-root join materialization
 */
namespace Contest::materialize {

// Types from Contest:: namespace
using Contest::ExecuteResult;

// Types from global scope (data_model/plan.h, foundation/attribute.h)
// Column, ColumnarTable, DataType, MappedMemory, Page, PAGE_SIZE, Plan,
// PlanNode are accessible without qualification

// Types from Contest::platform::
using Contest::platform::worker_pool;

// Types from Contest::io::
using Contest::io::ColumnarReader;

// Types from Contest::join::
using Contest::join::JoinInput;
using Contest::join::MatchCollector;
using Contest::join::resolve_input_source;

/**
 * @brief Parallel materialization of a single output column from match results.
 *
 * Divides matches across worker threads, each building pages independently
 * using preallocated mmap'd memory. Thread-local Column objects collect pages,
 * then merge into dest_col. Uses BuilderType (Int32PageBuilder or
 * VarcharPageBuilder) for type-specific page construction.
 *
 * **Why parallel construction:** Large join results (millions of rows) benefit
 * from dividing work across cores. Each thread processes a contiguous slice of
 * matches using range-based assignment (start = t*N/T, end = (t+1)*N/T) to
 * minimize cache conflicts.
 *
 * **Memory management:** Preallocates mmap'd memory pool for all threads to
 * avoid allocation contention. Each thread consumes pages from its private
 * slice; if exhausted, falls back to heap allocation (rare for properly sized
 * est_bytes_per_row).
 *
 * @tparam BuilderType     Page builder type
 * (Int32PageBuilder/VarcharPageBuilder).
 * @tparam ReaderFunc      Callable: (row_id, cursor) -> value_t.
 * @tparam InitBuilderFunc Callable: (page_allocator) -> BuilderType.
 * @param dest_col         Output column receiving materialized pages.
 * @param collector        Source of (build_id, probe_id) match pairs.
 * @param read_value       Function to read source value by row ID.
 * @param init_builder     Factory creating builder with page allocator.
 * @param from_build       True if reading from build side, false for probe.
 * @param est_bytes_per_row Estimated average bytes per row including overhead
 * (4 for INT32, ~35 for VARCHAR). Used to calculate mmap pool size:
 * rows_per_page = PAGE_SIZE / est_bytes_per_row determines pages_per_thread.
 * **Too small** triggers fallback heap allocations (performance penalty);
 * **too large** wastes mmap'd memory. VARCHAR estimate should account for
 * average string length + null bitmap + offset array overhead.
 */
template <typename BuilderType, typename ReaderFunc, typename InitBuilderFunc>
inline void
materialize_column(Column &dest_col, const MatchCollector &collector,
                   ReaderFunc &&read_value, InitBuilderFunc &&init_builder,
                   bool from_build, size_t est_bytes_per_row) {
    const size_t total_matches = collector.size();
    if (total_matches == 0)
        return;

    const_cast<MatchCollector &>(collector).ensure_finalized();
    constexpr int num_threads = worker_pool.thread_count();

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

    worker_pool.execute([&](size_t t) {
        size_t num_threads = worker_pool.thread_count();
        size_t start = t * total_matches / num_threads;
        size_t end = (t + 1) * total_matches / num_threads;
        if (start >= end)
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
        builder.prepare(end - start);

        const size_t check_interval = BuilderType::MIN_ROWS_PER_PAGE_CHECK;
        size_t rows_since_check = 0;

        auto range = from_build ? collector.get_left_range(start, end - start)
                                : collector.get_right_range(start, end - start);

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
 * @brief Materializes a single output column from join matches.
 *
 * **Purpose:** Dispatcher that determines source location, selects the correct
 * page builder type, and invokes materialize_column<> with appropriate readers.
 *
 * **Source resolution:** Column may originate from:
 * - Build side columnar table (base table leaf)
 * - Probe side columnar table (base table leaf)
 * - Build side intermediate result (ExecuteResult from previous join)
 * - Probe side intermediate result (ExecuteResult from previous join)
 *
 * **Page builder integration:** Creates builder factory (init_builder lambda)
 * that initializes Int32PageBuilder or VarcharPageBuilder with page allocator.
 * The builder accumulates values via add() calls, automatically flushing full
 * pages. See page_builders.h for builder implementations.
 *
 * **VARCHAR handling:** VarcharPageBuilder requires source Column pointer to
 * dereference value_t references (page_idx/offset_idx pairs) into actual string
 * bytes. For intermediate sources, resolves via Plan::inputs using
 * source_table/source_column metadata.
 *
 * @param dest_col         Output Column to populate with materialized pages.
 * @param col_idx          Global output column index (build columns first, then
 * probe).
 * @param build_size       Number of columns from build side (partition point).
 * @param collector        Match pairs from join execution.
 * @param build_input      Build side data (ColumnarTable or ExecuteResult).
 * @param probe_input      Probe side data (ColumnarTable or ExecuteResult).
 * @param build_node       PlanNode for build side output_attrs mapping.
 * @param probe_node       PlanNode for probe side output_attrs mapping.
 * @param columnar_reader  PageIndex-based reader for efficient page access.
 * @param plan             Full query plan, used to resolve base table metadata
 * for VARCHAR dereferencing when source is intermediate.
 */
inline void materialize_single_column(
    Column &dest_col, size_t col_idx, size_t build_size,
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input, const PlanNode &build_node,
    const PlanNode &probe_node, ColumnarReader &columnar_reader,
    const Plan &plan) {

    auto [input, node, local_idx] = resolve_input_source(
        col_idx, build_size, build_input, build_node, probe_input, probe_node);
    bool from_build = col_idx < build_size;

    const Column *col_source = nullptr;
    const mema::column_t *inter_source = nullptr;

    if (input.is_columnar()) {
        auto *table = std::get<const ColumnarTable *>(input.data);
        auto [actual_idx, _] = node.output_attrs[local_idx];
        col_source = &table->columns[actual_idx];
    } else {
        const auto &res = std::get<ExecuteResult>(input.data);
        inter_source = &res[local_idx];
    }

    auto reader = [&](uint32_t rid, ColumnarReader::Cursor &cursor,
                      DataType type) {
        if (col_source) {
            return columnar_reader.read_value(*col_source, local_idx, rid, type,
                                              cursor, from_build);
        }
        return (*inter_source)[rid];
    };

    if (dest_col.type == DataType::INT32) {
        auto init = [](std::function<Page *()> alloc) {
            return Int32PageBuilder(std::move(alloc));
        };
        materialize_column<Int32PageBuilder>(
            dest_col, collector,
            [&](uint32_t rid, ColumnarReader::Cursor &cursor) {
                return reader(rid, cursor, DataType::INT32);
            },
            init, from_build, 4);
        return;
    }

    const Column *str_src_ptr = col_source;
    if (!str_src_ptr && inter_source) {
        str_src_ptr = &plan.inputs[inter_source->source_table]
                           .columns[inter_source->source_column];
    }

    auto init = [str_src_ptr](std::function<Page *()> alloc) {
        return VarcharPageBuilder(*str_src_ptr, std::move(alloc));
    };

    materialize_column<VarcharPageBuilder>(
        dest_col, collector,
        [&](uint32_t rid, ColumnarReader::Cursor &cursor) {
            return reader(rid, cursor, DataType::VARCHAR);
        },
        init, from_build, 35);
}

/**
 * @brief Materializes all output columns into a new ColumnarTable.
 *
 * **Why ColumnarTable:** The contest API requires final query results in
 * ColumnarTable format - a page-based columnar layout with actual string data
 * copied into output pages. This differs from intermediate results
 * (ExecuteResult/column_t) which use compact value_t references.
 *
 * **Key operation:** Dereferences VARCHAR value_t references (page_idx,
 * offset_idx pairs) into actual string bytes. Intermediate results from
 * multi-way joins store pointers back to original base table pages; this
 * function copies the referenced strings into self-contained output pages.
 *
 * **Difference from construct_intermediate:**
 * - **Output format:** ColumnarTable (8KB pages) vs ExecuteResult (16KB pages,
 * value_t)
 * - **VARCHAR handling:** Copies string bytes vs preserves references
 * - **Use case:** Final query result vs intermediate join stage
 * - **Builder type:** Int32PageBuilder/VarcharPageBuilder (materialization) vs
 * direct column_t writes (intermediate)
 *
 * **Parallelization:** Each column is materialized sequentially, but within
 * each column, rows are divided across threads via materialize_column<>.
 *
 * @param collector        Match collection from join execution, provides
 * (build_id, probe_id) pairs.
 * @param build_input      Build side data source - either ColumnarTable (base
 * table) or ExecuteResult (previous join output).
 * @param probe_input      Probe side data source - either ColumnarTable or
 * ExecuteResult.
 * @param remapped_attrs   Output projection specification: list of (col_idx,
 * DataType) pairs. col_idx < build_size indicates build column, otherwise probe
 * column. Defines output schema and column ordering.
 * @param build_node       Metadata for build side: output_attrs maps local
 * column indices to source table columns and types.
 * @param probe_node       Metadata for probe side: output_attrs for probe
 * columns.
 * @param build_size       Number of columns from build side (split point in
 * remapped_attrs).
 * @param columnar_reader  PageIndex-accelerated reader for Column page access,
 * avoids linear page scans.
 * @param plan             Full query plan, needed to resolve base table Column
 * pointers for VARCHAR dereferencing from intermediate results
 * (inter_source->source_table/source_column).
 *
 * @return ColumnarTable with self-contained page data. Empty table (num_rows=0)
 * with correct column types if no matches.
 *
 * @see construct_intermediate.h for creating intermediate ExecuteResult format.
 * @see page_builders.h for Int32PageBuilder and VarcharPageBuilder
 * implementations.
 */
inline ColumnarTable
materialize(const MatchCollector &collector, const JoinInput &build_input,
            const JoinInput &probe_input,
            const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
            const PlanNode &build_node, const PlanNode &probe_node,
            size_t build_size, ColumnarReader &columnar_reader,
            const Plan &plan) {

    ColumnarTable result;
    result.num_rows = collector.size();

    if (collector.size() == 0) {
        for (auto [_, dtype] : remapped_attrs) {
            result.columns.emplace_back(dtype);
        }
        return result;
    }

    for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
        auto [col_idx, data_type] = remapped_attrs[out_idx];
        result.columns.emplace_back(data_type);
        Column &dest_col = result.columns.back();
        materialize_single_column(dest_col, col_idx, build_size, collector,
                                  build_input, probe_input, build_node,
                                  probe_node, columnar_reader, plan);
    }
    return result;
}

/** @brief Creates empty ColumnarTable with correct column types for zero-match
 * case. */
inline ColumnarTable create_empty_result(
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs) {
    ColumnarTable empty_result;
    empty_result.num_rows = 0;
    for (auto [_, data_type] : remapped_attrs) {
        empty_result.columns.emplace_back(data_type);
    }
    return empty_result;
}

} // namespace Contest::materialize

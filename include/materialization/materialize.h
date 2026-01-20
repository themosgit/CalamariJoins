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

/** @namespace Contest::materialize @brief Join result materialization. */
namespace Contest::materialize {

using Contest::ExecuteResult;
using Contest::io::ColumnarReader;
using Contest::join::JoinInput;
using Contest::join::resolve_input_source;
using Contest::join::ThreadLocalMatchBuffer;
using Contest::platform::THREAD_COUNT;
using Contest::platform::worker_pool;

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

/**
 * @brief Parallel materialization of a single output column from thread-local
 * buffers.
 *
 * Each thread processes its own buffer directly without merge overhead.
 *
 * @tparam BuilderType     Int32PageBuilder or VarcharPageBuilder.
 * @tparam ReaderFunc      Callable: (row_id, cursor) -> value_t.
 * @tparam InitBuilderFunc Callable: (page_allocator) -> BuilderType.
 * @param est_bytes_per_row Average bytes per row (4 for INT32, ~35 for
 * VARCHAR).
 */
template <typename BuilderType, typename ReaderFunc, typename InitBuilderFunc>
inline void materialize_column_from_buffers(
    Column &dest_col, std::vector<ThreadLocalMatchBuffer> &buffers,
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
 * @brief Materializes a single output column from thread-local buffers.
 *
 * Dispatcher that determines source location (columnar/intermediate,
 * build/probe), selects page builder type, and invokes
 * materialize_column_from_buffers<>. VARCHAR handling requires source Column
 * pointer for string dereferencing.
 */
inline void materialize_single_column_from_buffers(
    Column &dest_col, size_t col_idx, size_t build_size,
    std::vector<ThreadLocalMatchBuffer> &buffers, size_t total_matches,
    const JoinInput &build_input, const JoinInput &probe_input,
    const PlanNode &build_node, const PlanNode &probe_node,
    ColumnarReader &columnar_reader, const Plan &plan) {

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
        materialize_column_from_buffers<Int32PageBuilder>(
            dest_col, buffers, total_matches,
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

    materialize_column_from_buffers<VarcharPageBuilder>(
        dest_col, buffers, total_matches,
        [&](uint32_t rid, ColumnarReader::Cursor &cursor) {
            return reader(rid, cursor, DataType::VARCHAR);
        },
        init, from_build, 35);
}

/**
 * @brief Materializes all output columns from thread-local buffers into
 * ColumnarTable.
 *
 * Dereferences VARCHAR value_t references into actual string bytes.
 *
 * @param buffers          Thread-local match buffers from probe.
 * @param build_input      Build side data source.
 * @param probe_input      Probe side data source.
 * @param remapped_attrs   Output projection: (col_idx, DataType) pairs.
 * @param build_node       Metadata for build side output_attrs mapping.
 * @param probe_node       Metadata for probe side output_attrs mapping.
 * @param build_size       Number of columns from build side.
 * @param columnar_reader  PageIndex-accelerated reader for Column page access.
 * @param plan             Full query plan for VARCHAR dereferencing.
 * @return ColumnarTable with self-contained page data.
 *
 * @see construct_intermediate.h for creating intermediate ExecuteResult.
 * @see page_builders.h for Int32PageBuilder and VarcharPageBuilder.
 */
inline ColumnarTable materialize_from_buffers(
    std::vector<ThreadLocalMatchBuffer> &buffers, const JoinInput &build_input,
    const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, const Plan &plan) {

    // Compute total_matches
    size_t total_matches = 0;
    for (const auto &buf : buffers) {
        total_matches += buf.count();
    }

    ColumnarTable result;
    result.num_rows = total_matches;

    if (total_matches == 0) {
        for (auto [_, dtype] : remapped_attrs) {
            result.columns.emplace_back(dtype);
        }
        return result;
    }

    for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
        auto [col_idx, data_type] = remapped_attrs[out_idx];
        result.columns.emplace_back(data_type);
        Column &dest_col = result.columns.back();
        materialize_single_column_from_buffers(
            dest_col, col_idx, build_size, buffers, total_matches, build_input,
            probe_input, build_node, probe_node, columnar_reader, plan);
    }
    return result;
}

} // namespace Contest::materialize

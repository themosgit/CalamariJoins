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

namespace Contest {

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
inline ColumnarTable create_empty_result(
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs) {
    ColumnarTable empty_result;
    empty_result.num_rows = 0;
    for (auto [_, data_type] : remapped_attrs) {
        empty_result.columns.emplace_back(data_type);
    }
    return empty_result;
}

} // namespace Contest

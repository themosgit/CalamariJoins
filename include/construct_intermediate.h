#pragma once

#include <columnar_reader.h>
#include <intermediate.h>
#include <join_setup.h>
#include <plan.h>
#include <vector>

namespace Contest {

using ExecuteResult = std::vector<mema::column_t>;

/**
 *
 *  stores join matches as packed 64-bit values
 *  lower 32 bits hold left row id
 *  upper 32 bits hold right row id
 *  enables efficient storage and cache locality
 *
 **/
struct MatchCollector {
    std::vector<uint64_t> matches;

    inline void reserve(size_t estimated_matches) {
        matches.reserve(estimated_matches);
    }

    /* packs left and right row ids into single 64-bit value */
    inline void add_match(uint32_t left, uint32_t right) {
        matches.push_back(static_cast<uint64_t>(left) |
                          (static_cast<uint64_t>(right) << 32));
    }

    inline size_t size() const { return matches.size(); }
};

struct JoinInput;
class ColumnarReader;

/**
 *
 *  constructs intermediate results from column_t intermediate format
 *  both build and probe are intermediate results
 *  detects sequential runs to enable bulk copy
 *  shift extracts correct row id from packed match
 *
 **/
inline void construct_intermediate_from_intermediate(
    const MatchCollector &collector, const ExecuteResult &build,
    const ExecuteResult &probe,
    const std::vector<std::tuple<size_t, DataType>> &output_attrs,
    ExecuteResult &results) {
    const size_t total_matches = collector.size();
    if (total_matches == 0)
        return;

    const uint64_t *matches_ptr = collector.matches.data();
    const size_t build_size = build.size();

    for (size_t out_idx = 0; out_idx < output_attrs.size(); ++out_idx) {
        auto [col_idx, _] = output_attrs[out_idx];

        const bool from_build = col_idx < build_size;
        const mema::column_t *column =
            from_build ? &build[col_idx] : &probe[col_idx - build_size];
        const uint32_t shift = from_build ? 0 : 32;

        results[out_idx].reserve(total_matches);

        if (column->has_direct_access()) {
            size_t match_idx = 0;

            while (match_idx < total_matches) {
                uint32_t row_id = (matches_ptr[match_idx] >> shift);

                size_t run_end = match_idx + 1;
                uint32_t expected_row = row_id + 1;
                while (run_end < total_matches &&
                       (matches_ptr[run_end] >> shift) == expected_row) {
                    ++expected_row;
                    ++run_end;
                }
                size_t total_run_length = run_end - match_idx;

                size_t processed = 0;
                while (processed < total_run_length) {
                    uint32_t current_row = row_id + processed;
                    size_t page_offset = current_row % mema::CAP_PER_PAGE;
                    size_t rows_in_page =
                        std::min(mema::CAP_PER_PAGE - page_offset,
                                 total_run_length - processed);

                    if (page_offset == 0 &&
                        rows_in_page == mema::CAP_PER_PAGE) {
                        results[out_idx].append_bulk(
                            &(*column)[current_row].value, rows_in_page);
                    } else if (rows_in_page >= 4) {
                        results[out_idx].append_consecutive(
                            &(*column)[current_row], rows_in_page);
                    } else {
                        for (size_t i = 0; i < rows_in_page; ++i) {
                            results[out_idx].append((*column)[current_row + i]);
                        }
                    }
                    processed += rows_in_page;
                }
                match_idx = run_end;
            }
        } else {
            size_t match_idx = 0;
            while (match_idx < total_matches) {
                uint32_t row_id = (matches_ptr[match_idx] >> shift);
                const mema::value_t *first_value = column->get_by_row(row_id);

                if (first_value == nullptr) {
                    results[out_idx].append_null();
                    ++match_idx;
                    continue;
                }

                size_t run_end = match_idx + 1;
                uint32_t expected_row = row_id + 1;
                while (run_end < total_matches) {
                    uint32_t next_row = (matches_ptr[run_end] >> shift);
                    if (next_row != expected_row)
                        break;
                    if (column->get_by_row(next_row) == nullptr)
                        break;
                    ++expected_row;
                    ++run_end;
                }

                size_t run_length = run_end - match_idx;

                if (run_length >= 4) {
                    results[out_idx].append_consecutive(first_value,
                                                        run_length);
                } else {
                    results[out_idx].append(*first_value);
                    for (size_t i = 1; i < run_length; ++i) {
                        uint32_t r = row_id + i;
                        const mema::value_t *val = column->get_by_row(r);
                        results[out_idx].append(*val);
                    }
                }
                match_idx = run_end;
            }
        }
    }
}

/**
 *
 *  constructs intermediate results directly from ColumnarTable columnar inputs
 *  both build and probe are reading from original page format
 *  uses columnar_reader to efficiently read from pages
 *  avoids intermediate column_t conversion
 *
 **/
inline void construct_intermediate_from_columnar(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, ExecuteResult &results) {

    const size_t total_matches = collector.size();
    if (total_matches == 0)
        return;

    const uint64_t *matches_ptr = collector.matches.data();
    auto *build_table = std::get<const ColumnarTable *>(build_input.data);
    auto *probe_table = std::get<const ColumnarTable *>(probe_input.data);

    for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
        auto [col_idx, data_type] = remapped_attrs[out_idx];
        bool from_build = col_idx < build_size;
        size_t remapped_col_idx = from_build ? col_idx : col_idx - build_size;

        results[out_idx].reserve(total_matches);

        const ColumnarTable *src_table = from_build ? build_table : probe_table;
        const PlanNode *src_node = from_build ? &build_node : &probe_node;
        auto [actual_col_idx, _] = src_node->output_attrs[remapped_col_idx];
        const Column &src_col = src_table->columns[actual_col_idx];

        if (from_build) {
            for (size_t i = 0; i < total_matches; ++i) {
                uint32_t row_id =
                    static_cast<uint32_t>(matches_ptr[i] & 0xFFFFFFFF);
                mema::value_t value = columnar_reader.read_value_build(
                    src_col, remapped_col_idx, row_id, src_col.type);
                results[out_idx].append(value);
            }
        } else {
            for (size_t i = 0; i < total_matches; ++i) {
                uint32_t row_id = static_cast<uint32_t>(matches_ptr[i] >> 32);
                mema::value_t value = columnar_reader.read_value_probe(
                    src_col, remapped_col_idx, row_id, src_col.type);
                results[out_idx].append(value);
            }
        }
    }
}

/**
 *
 *  constructs intermediate results from mixed input types
 *  one side is ColumnarTable columnar other is column_t intermediate
 *  dispatches to appropriate reader based on source type
 *  handles all combinations of intermediate and columnar inputs
 *
 **/
inline void construct_intermediate_mixed(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, ExecuteResult &results) {

    const size_t total_matches = collector.size();
    if (total_matches == 0)
        return;

    const uint64_t *matches_ptr = collector.matches.data();

    for (size_t out_idx = 0; out_idx < remapped_attrs.size(); ++out_idx) {
        auto [col_idx, data_type] = remapped_attrs[out_idx];
        bool from_build = col_idx < build_size;
        size_t remapped_col_idx = from_build ? col_idx : col_idx - build_size;

        results[out_idx].reserve(total_matches);

        bool source_is_columnar = false;
        const Column *columnar_src_col = nullptr;
        const mema::column_t *intermediate_src_col = nullptr;

        if (from_build) {
            if (build_input.is_columnar()) {
                source_is_columnar = true;
                auto *build_table =
                    std::get<const ColumnarTable *>(build_input.data);
                auto [actual_col_idx, _] =
                    build_node.output_attrs[remapped_col_idx];
                columnar_src_col = &build_table->columns[actual_col_idx];
            } else {
                const auto &build_result =
                    std::get<ExecuteResult>(build_input.data);
                intermediate_src_col = &build_result[remapped_col_idx];
            }
        } else {
            if (probe_input.is_columnar()) {
                source_is_columnar = true;
                auto *probe_table =
                    std::get<const ColumnarTable *>(probe_input.data);
                auto [actual_col_idx, _] =
                    probe_node.output_attrs[remapped_col_idx];
                columnar_src_col = &probe_table->columns[actual_col_idx];
            } else {
                const auto &probe_result =
                    std::get<ExecuteResult>(probe_input.data);
                intermediate_src_col = &probe_result[remapped_col_idx];
            }
        }

        if (source_is_columnar) {
            if (from_build) {
                for (size_t i = 0; i < total_matches; ++i) {
                    uint32_t row_id =
                        static_cast<uint32_t>(matches_ptr[i] & 0xFFFFFFFF);
                    mema::value_t value = columnar_reader.read_value_build(
                        *columnar_src_col, remapped_col_idx, row_id,
                        columnar_src_col->type);
                    results[out_idx].append(value);
                }
            } else {
                for (size_t i = 0; i < total_matches; ++i) {
                    uint32_t row_id =
                        static_cast<uint32_t>(matches_ptr[i] >> 32);
                    mema::value_t value = columnar_reader.read_value_probe(
                        *columnar_src_col, remapped_col_idx, row_id,
                        columnar_src_col->type);
                    results[out_idx].append(value);
                }
            }
        } else {
            bool direct_access = intermediate_src_col->has_direct_access();
            const mema::column_t &col = *intermediate_src_col;

            if (from_build) {
                if (direct_access) {
                    for (size_t i = 0; i < total_matches; ++i) {
                        uint32_t row_id =
                            static_cast<uint32_t>(matches_ptr[i] & 0xFFFFFFFF);
                        results[out_idx].append(col[row_id]);
                    }
                } else {
                    for (size_t i = 0; i < total_matches; ++i) {
                        uint32_t row_id =
                            static_cast<uint32_t>(matches_ptr[i] & 0xFFFFFFFF);
                        const mema::value_t *val = col.get_by_row(row_id);
                        results[out_idx].append(val ? *val : mema::value_t{});
                    }
                }
            } else {
                if (direct_access) {
                    for (size_t i = 0; i < total_matches; ++i) {
                        uint32_t row_id =
                            static_cast<uint32_t>(matches_ptr[i] >> 32);
                        results[out_idx].append(col[row_id]);
                    }
                } else {
                    for (size_t i = 0; i < total_matches; ++i) {
                        uint32_t row_id =
                            static_cast<uint32_t>(matches_ptr[i] >> 32);
                        const mema::value_t *val = col.get_by_row(row_id);
                        results[out_idx].append(val ? *val : mema::value_t{});
                    }
                }
            }
        }
    }
}

} // namespace Contest

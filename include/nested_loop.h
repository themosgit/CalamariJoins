#pragma once

#include <columnar_reader.h>
#include <construct_intermediate.h>
#include <cstdint>
#include <intermediate.h>
#include <join_setup.h>
#include <plan.h>

namespace Contest {

/**
 *
 *  performs nested loop join between two ColumnarTable columnar inputs
 *  collects matches into MatchCollector for later materialization
 *  uses prefix sums for efficient sparse page bitmap navigation
 *  iterates all build rows against all probe rows
 *
 **/
inline void nested_loop_from_columnar(const JoinInput &build_input,
                                      const JoinInput &probe_input,
                                      size_t build_attr, size_t probe_attr,
                                      ColumnarReader &columnar_reader,
                                      MatchCollector &collector) {

    auto *build_table = std::get<const ColumnarTable *>(build_input.data);
    auto *probe_table = std::get<const ColumnarTable *>(probe_input.data);

    auto [build_col_idx, _] = build_input.node->output_attrs[build_attr];
    auto [probe_col_idx, __] = probe_input.node->output_attrs[probe_attr];

    const Column &build_col = build_table->columns[build_col_idx];
    const Column &probe_col = probe_table->columns[probe_col_idx];

    const auto &build_prefix = columnar_reader.get_build_page_index(build_attr);
    const auto &probe_prefix = columnar_reader.get_probe_page_index(probe_attr);

    size_t build_rows = build_input.row_count(build_attr);
    size_t probe_rows = probe_input.row_count(probe_attr);
    collector.reserve(build_rows * probe_rows / 2);

    uint32_t build_row_id = 0;
    size_t build_page_idx = 0;
    for (auto *build_page_obj : build_col.pages) {
        auto *build_page = build_page_obj->data;
        auto build_num_rows = *reinterpret_cast<uint16_t *>(build_page);
        auto build_num_values = *reinterpret_cast<uint16_t *>(build_page + 2);
        auto *build_data = reinterpret_cast<const int32_t *>(build_page + 4);
        const auto &build_page_prefix =
            build_prefix.page_prefix_sums[build_page_idx];

        for (uint16_t i = 0; i < build_num_rows; i++) {
            int32_t build_value;
            bool build_valid = true;

            if (build_num_rows == build_num_values) {
                build_value = build_data[i];
            } else {
                auto *build_bitmap = reinterpret_cast<const uint8_t *>(
                    build_page + PAGE_SIZE - (build_num_rows + 7) / 8);
                build_valid = build_bitmap[i / 8] & (1u << (i % 8));
                if (build_valid) {
                    size_t chunk_idx = i >> 6;
                    size_t bit_offset = i & 0x3F;
                    uint16_t data_idx = build_page_prefix[chunk_idx];
                    auto *bitmap64 =
                        reinterpret_cast<const uint64_t *>(build_bitmap);
                    uint64_t mask = (1ULL << bit_offset) - 1;
                    data_idx +=
                        __builtin_popcountll(bitmap64[chunk_idx] & mask);
                    build_value = build_data[data_idx];
                }
            }
            if (!build_valid) {
                build_row_id++;
                continue;
            }

            uint32_t probe_row_id = 0;
            for (auto *probe_page_obj : probe_col.pages) {
                auto *probe_page = probe_page_obj->data;
                auto probe_num_rows =
                    *reinterpret_cast<const uint16_t *>(probe_page);
                auto probe_num_values =
                    *reinterpret_cast<const uint16_t *>(probe_page + 2);
                auto *probe_data =
                    reinterpret_cast<const int32_t *>(probe_page + 4);

                if (probe_num_rows == probe_num_values) {
                    for (uint16_t j = 0; j < probe_num_rows; j++) {
                        if (probe_data[j] == build_value) {
                            collector.add_match(build_row_id, probe_row_id);
                        }
                        probe_row_id++;
                    }
                } else {
                    auto *probe_bitmap = reinterpret_cast<const uint8_t *>(
                        probe_page + PAGE_SIZE - (probe_num_rows + 7) / 8);
                    uint16_t data_idx = 0;
                    for (uint16_t j = 0; j < probe_num_rows; j++) {
                        bool probe_valid =
                            probe_bitmap[j / 8] & (1u << (j % 8));
                        if (probe_valid) {
                            if (probe_data[data_idx] == build_value) {
                                collector.add_match(build_row_id, probe_row_id);
                            }
                            data_idx++;
                        }
                        probe_row_id++;
                    }
                }
            }
            build_row_id++;
        }
        build_page_idx++;
    }
}

/**
 *
 *  performs nested loop join between two column_t intermediate inputs
 *  collects matches into MatchCollector for later materialization
 *  handles both direct access and sparse column_t formats
 *
 **/
inline void nested_loop_from_intermediate(const ExecuteResult &build,
                                          const ExecuteResult &probe,
                                          size_t build_col, size_t probe_col,
                                          MatchCollector &collector) {
    if (build.size() == 0 || probe.size() == 0 || build_col >= build.size() ||
        probe_col >= probe.size()) {
        return;
    }

    const mema::column_t &build_column = build[build_col];
    const mema::column_t &probe_column = probe[probe_col];
    const size_t build_count = build_column.row_count();
    const size_t probe_count = probe_column.row_count();

    const bool build_direct = build_column.has_direct_access();
    const bool probe_direct = probe_column.has_direct_access();

    collector.reserve(build_count * probe_count / 2);

    if (build_direct && probe_direct) {
        for (size_t build_idx = 0; build_idx < build_count; build_idx++) {
            int build_value = build_column[build_idx].value;
            for (size_t probe_idx = 0; probe_idx < probe_count; probe_idx++) {
                if (probe_column[probe_idx].value == build_value) {
                    collector.add_match(build_idx, probe_idx);
                }
            }
        }
    } else {
        for (size_t build_idx = 0; build_idx < build_count; build_idx++) {
            const mema::value_t *build_key = build_column.get_by_row(build_idx);
            if (!build_key)
                continue;
            for (size_t probe_idx = 0; probe_idx < probe_count; probe_idx++) {
                const mema::value_t *probe_key =
                    probe_column.get_by_row(probe_idx);
                if (probe_key && probe_key->value == build_key->value) {
                    collector.add_match(build_idx, probe_idx);
                }
            }
        }
    }
}

/**
 *
 *  performs nested loop join between mixed input types
 *  one side is ColumnarTable columnar other is column_t intermediate
 *  collects matches into MatchCollector for later materialization
 *
 **/
inline void nested_loop_mixed(const JoinInput &build_input,
                              const JoinInput &probe_input, size_t build_attr,
                              size_t probe_attr,
                              ColumnarReader &columnar_reader,
                              MatchCollector &collector) {
    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();
    size_t build_rows = build_input.row_count(build_attr);
    size_t probe_rows = probe_input.row_count(probe_attr);
    collector.reserve(build_rows * probe_rows / 2);

    if (build_is_columnar && !probe_is_columnar) {
        auto *build_table = std::get<const ColumnarTable *>(build_input.data);
        auto [build_col_idx, _] = build_input.node->output_attrs[build_attr];
        const Column &build_col = build_table->columns[build_col_idx];
        const auto &build_prefix =
            columnar_reader.get_build_page_index(build_attr);

        const auto &probe_result = std::get<ExecuteResult>(probe_input.data);
        const auto &probe_column = probe_result[probe_attr];

        uint32_t build_row_id = 0;
        size_t build_page_idx = 0;
        for (auto *build_page_obj : build_col.pages) {
            auto *build_page = build_page_obj->data;
            auto build_num_rows =
                *reinterpret_cast<const uint16_t *>(build_page);
            auto build_num_values =
                *reinterpret_cast<const uint16_t *>(build_page + 2);
            auto *build_data =
                reinterpret_cast<const int32_t *>(build_page + 4);
            const auto &build_page_prefix =
                build_prefix.page_prefix_sums[build_page_idx];

            for (uint16_t i = 0; i < build_num_rows; ++i) {
                int32_t build_value;
                bool build_valid = true;

                if (build_num_rows == build_num_values) {
                    build_value = build_data[i];
                } else {
                    auto *build_bitmap = reinterpret_cast<const uint8_t *>(
                        build_page + PAGE_SIZE - (build_num_rows + 7) / 8);
                    build_valid = build_bitmap[i / 8] & (1u << (i % 8));
                    if (build_valid) {
                        size_t chunk_idx = i >> 6;
                        size_t bit_offset = i & 0x3F;
                        uint16_t data_idx = build_page_prefix[chunk_idx];
                        auto *bitmap64 =
                            reinterpret_cast<const uint64_t *>(build_bitmap);
                        uint64_t mask = (1ULL << bit_offset) - 1;
                        data_idx +=
                            __builtin_popcountll(bitmap64[chunk_idx] & mask);
                        build_value = build_data[data_idx];
                    }
                }

                if (build_valid) {
                    if (probe_column.has_direct_access()) {
                        for (size_t probe_idx = 0; probe_idx < probe_rows;
                             ++probe_idx) {
                            if (probe_column[probe_idx].value == build_value) {
                                collector.add_match(build_row_id, probe_idx);
                            }
                        }
                    } else {
                        for (size_t probe_idx = 0; probe_idx < probe_rows;
                             ++probe_idx) {
                            const mema::value_t *probe_key =
                                probe_column.get_by_row(probe_idx);
                            if (probe_key && probe_key->value == build_value) {
                                collector.add_match(build_row_id, probe_idx);
                            }
                        }
                    }
                }
                build_row_id++;
            }
            build_page_idx++;
        }
    } else {
        const auto &build_result = std::get<ExecuteResult>(build_input.data);
        const auto &build_column = build_result[build_attr];

        auto *probe_table = std::get<const ColumnarTable *>(probe_input.data);
        auto [probe_col_idx, _] = probe_input.node->output_attrs[probe_attr];
        const Column &probe_col = probe_table->columns[probe_col_idx];

        if (build_column.has_direct_access()) {
            for (size_t build_idx = 0; build_idx < build_rows; ++build_idx) {
                int build_value = build_column[build_idx].value;

                uint32_t probe_row_id = 0;
                for (auto *probe_page_obj : probe_col.pages) {
                    auto *probe_page = probe_page_obj->data;
                    auto probe_num_rows =
                        *reinterpret_cast<const uint16_t *>(probe_page);
                    auto probe_num_values =
                        *reinterpret_cast<const uint16_t *>(probe_page + 2);
                    auto *probe_data =
                        reinterpret_cast<const int32_t *>(probe_page + 4);

                    if (probe_num_rows == probe_num_values) {
                        for (uint16_t j = 0; j < probe_num_rows; ++j) {
                            if (probe_data[j] == build_value) {
                                collector.add_match(build_idx, probe_row_id);
                            }
                            probe_row_id++;
                        }
                    } else {
                        auto *probe_bitmap = reinterpret_cast<const uint8_t *>(
                            probe_page + PAGE_SIZE - (probe_num_rows + 7) / 8);
                        uint16_t data_idx = 0;
                        for (uint16_t j = 0; j < probe_num_rows; ++j) {
                            bool probe_valid =
                                probe_bitmap[j / 8] & (1u << (j % 8));
                            if (probe_valid) {
                                if (probe_data[data_idx] == build_value) {
                                    collector.add_match(build_idx,
                                                        probe_row_id);
                                }
                                data_idx++;
                            }
                            probe_row_id++;
                        }
                    }
                }
            }
        } else {
            for (size_t build_idx = 0; build_idx < build_rows; ++build_idx) {
                const mema::value_t *build_key =
                    build_column.get_by_row(build_idx);
                if (!build_key)
                    continue;
                int build_value = build_key->value;

                uint32_t probe_row_id = 0;
                for (auto *probe_page_obj : probe_col.pages) {
                    auto *probe_page = probe_page_obj->data;
                    auto probe_num_rows =
                        *reinterpret_cast<const uint16_t *>(probe_page);
                    auto probe_num_values =
                        *reinterpret_cast<const uint16_t *>(probe_page + 2);
                    auto *probe_data =
                        reinterpret_cast<const int32_t *>(probe_page + 4);

                    if (probe_num_rows == probe_num_values) {
                        for (uint16_t j = 0; j < probe_num_rows; ++j) {
                            if (probe_data[j] == build_value) {
                                collector.add_match(build_idx, probe_row_id);
                            }
                            probe_row_id++;
                        }
                    } else {
                        auto *probe_bitmap = reinterpret_cast<const uint8_t *>(
                            probe_page + PAGE_SIZE - (probe_num_rows + 7) / 8);
                        uint16_t data_idx = 0;
                        for (uint16_t j = 0; j < probe_num_rows; ++j) {
                            bool probe_valid =
                                probe_bitmap[j / 8] & (1u << (j % 8));
                            if (probe_valid) {
                                if (probe_data[data_idx] == build_value) {
                                    collector.add_match(build_idx,
                                                        probe_row_id);
                                }
                                data_idx++;
                            }
                            probe_row_id++;
                        }
                    }
                }
            }
        }
    }
}
} // namespace Contest

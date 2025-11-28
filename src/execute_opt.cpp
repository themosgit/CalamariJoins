#if defined(__APPLE__) && defined(__aarch64__)
#include <hardware_darwin.h>
#else
#include <hardware.h>
#endif

#include <columnar_structs.h>
#include <hashtable.h>

namespace Contest {

using ExecuteResult = std::vector<mema::column_t>;

ExecuteResult execute_impl(const Plan &plan, size_t node_idx);

struct JoinAlgorithm {
    ExecuteResult &build;
    ExecuteResult &probe;
    ExecuteResult &results;
    size_t build_col, probe_col;
    bool swap;
    const std::vector<std::tuple<size_t, DataType>> &output_attrs;

    ExecuteResult *left_table_ref;
    ExecuteResult *right_table_ref;
    size_t left_table_size;

    /* keeps  metadata for each output column */
    struct Accessor {
        const mema::column_t *column;
        bool is_left;
    };
    std::vector<Accessor> accessors;

    void run() {
        left_table_ref = swap ? &build : &probe;
        right_table_ref = swap ? &probe : &build;
        left_table_size = left_table_ref->size();

        /* initialize  accessor for each output column */
        accessors.reserve(output_attrs.size());
        for (auto [col_idx, _] : output_attrs) {
            if (col_idx < left_table_size) {
                accessors.push_back({&(*left_table_ref)[col_idx], true});
            } else {
                accessors.push_back(
                    {&(*right_table_ref)[col_idx - left_table_size], false});
            }
        }

        /* Use nested loop join for small tables, hash join otherwise */
        const size_t HASH_TABLE_THRESHOLD = 4;
        if (build[build_col].row_count() < HASH_TABLE_THRESHOLD) {
            nested_loop_join();
        } else {
            hash_join();
        }
    }

    /**
     *
     * Crazy code duplication follows this is done
     * in order to minimise branches within the hot path.
     *
     **/
  private:
    struct MatchBatch {
        /* packed: lower 32 bits = left row, upper 32 bits = right row */
        std::vector<uint64_t> matches;
        MatchBatch() { matches.reserve(mema::CAP_PER_PAGE); }

        inline void add_match(uint32_t left, uint32_t right) {
            matches.push_back(static_cast<uint64_t>(left) |
                              (static_cast<uint64_t>(right) << 32));
        }

        inline bool is_full() const {
            return matches.size() >= mema::CAP_PER_PAGE;
        }

        inline void clear() { matches.clear(); }

        inline size_t size() const { return matches.size(); }
    };

    void hash_join() {
        mema::column_t &build_column = build[build_col];
        mema::column_t &probe_column = probe[probe_col];

        UnchainedHashtable hash_table(build_column.size());
        hash_table.build(build_column);

        const size_t probe_count = probe_column.row_count();
        const bool probe_direct = probe_column.has_direct_access();

        MatchBatch batch;

        if (swap) {
            if (probe_direct) {
                for (size_t idx = 0; idx < probe_count; ++idx) {
                    int32_t key_val = probe_column[idx].value;
                    auto range = hash_table.find(key_val);
                    const auto *start = range.first;
                    const auto *end = range.second;

                    for (const auto *ptr = start; ptr < end; ++ptr) {
                        if (ptr->key == key_val) {
                            batch.add_match(ptr->row_id, idx);
                            if (batch.is_full()) {
                                flush_batch(batch);
                            }
                        }
                    }
                }
            } else {
                for (size_t idx = 0; idx < probe_count; ++idx) {
                    const mema::value_t *key = probe_column.get_by_row(idx);
                    if (key != nullptr) {
                        int32_t key_val = key->value;
                        auto range = hash_table.find(key_val);
                        const auto *start = range.first;
                        const auto *end = range.second;

                        for (const auto *ptr = start; ptr < end; ++ptr) {
                            if (ptr->key == key_val) {
                                batch.add_match(ptr->row_id, idx);
                                if (batch.is_full()) {
                                    flush_batch(batch);
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (probe_direct) {
                for (size_t idx = 0; idx < probe_count; ++idx) {
                    int32_t key_val = probe_column[idx].value;
                    auto range = hash_table.find(key_val);
                    const auto *start = range.first;
                    const auto *end = range.second;

                    for (const auto *ptr = start; ptr < end; ++ptr) {
                        if (ptr->key == key_val) {
                            batch.add_match(idx, ptr->row_id);
                            if (batch.is_full()) {
                                flush_batch(batch);
                            }
                        }
                    }
                }
            } else {
                for (size_t idx = 0; idx < probe_count; ++idx) {
                    const mema::value_t *key = probe_column.get_by_row(idx);
                    if (key != nullptr) {
                        int32_t key_val = key->value;
                        auto range = hash_table.find(key_val);
                        const auto *start = range.first;
                        const auto *end = range.second;

                        for (const auto *ptr = start; ptr < end; ++ptr) {
                            if (ptr->key == key_val) {
                                batch.add_match(idx, ptr->row_id);
                                if (batch.is_full()) {
                                    flush_batch(batch);
                                }
                            }
                        }
                    }
                }
            }
        }

        /* flush any remaining matches */
        if (batch.size() > 0) {
            flush_batch(batch);
        }
    }

    void nested_loop_join() {
        if (build.size() == 0 || probe.size() == 0 ||
            build_col >= build.size() || probe_col >= probe.size()) {
            return;
        }

        mema::column_t &build_column = build[build_col];
        mema::column_t &probe_column = probe[probe_col];
        const size_t build_count = build_column.row_count();
        const size_t probe_count = probe_column.row_count();

        const bool build_direct = build_column.has_direct_access();
        const bool probe_direct = probe_column.has_direct_access();

        MatchBatch batch;

        if (build_direct && probe_direct) {
            if (swap) {
                for (size_t build_idx = 0; build_idx < build_count;
                     build_idx++) {
                    int build_value = build_column[build_idx].value;
                    for (size_t probe_idx = 0; probe_idx < probe_count;
                         probe_idx++) {
                        if (probe_column[probe_idx].value == build_value) {
                            batch.add_match(build_idx, probe_idx);
                            if (batch.is_full()) {
                                flush_batch(batch);
                            }
                        }
                    }
                }
            } else {
                for (size_t build_idx = 0; build_idx < build_count;
                     build_idx++) {
                    int build_value = build_column[build_idx].value;
                    for (size_t probe_idx = 0; probe_idx < probe_count;
                         probe_idx++) {
                        if (probe_column[probe_idx].value == build_value) {
                            batch.add_match(probe_idx, build_idx);
                            if (batch.is_full()) {
                                flush_batch(batch);
                            }
                        }
                    }
                }
            }
        } else {
            if (swap) {
                for (size_t build_idx = 0; build_idx < build_count;
                     build_idx++) {
                    const mema::value_t *build_key =
                        build_column.get_by_row(build_idx);
                    if (!build_key)
                        continue;

                    for (size_t probe_idx = 0; probe_idx < probe_count;
                         probe_idx++) {
                        const mema::value_t *probe_key =
                            probe_column.get_by_row(probe_idx);
                        if (probe_key && probe_key->value == build_key->value) {
                            batch.add_match(build_idx, probe_idx);
                            if (batch.is_full()) {
                                flush_batch(batch);
                            }
                        }
                    }
                }
            } else {
                for (size_t build_idx = 0; build_idx < build_count;
                     build_idx++) {
                    const mema::value_t *build_key =
                        build_column.get_by_row(build_idx);
                    if (!build_key)
                        continue;

                    for (size_t probe_idx = 0; probe_idx < probe_count;
                         probe_idx++) {
                        const mema::value_t *probe_key =
                            probe_column.get_by_row(probe_idx);
                        if (probe_key && probe_key->value == build_key->value) {
                            batch.add_match(probe_idx, build_idx);
                            if (batch.is_full()) {
                                flush_batch(batch);
                            }
                        }
                    }
                }
            }
        }

        /* flush any remaining matches */
        if (batch.size() > 0) {
            flush_batch(batch);
        }
    }

    inline void flush_batch(MatchBatch &batch) {
        if (batch.size() == 0)
            return;
        /* process each output column for all matches in the batch */
        for (size_t col_idx = 0; col_idx < accessors.size(); ++col_idx) {
            const auto &acc = accessors[col_idx];
            const uint32_t shift = acc.is_left ? 0 : 32;
            const uint32_t mask = 0xFFFFFFFF;

            if (acc.column->has_direct_access()) {
                /* direct access column */
                for (uint64_t packed : batch.matches) {
                    uint32_t row_id = (packed >> shift) & mask;
                    results[col_idx].append((*acc.column)[row_id]);
                }
            } else {
                /* sparse column with nulls */
                for (uint64_t packed : batch.matches) {
                    uint32_t row_id = (packed >> shift) & mask;
                    const mema::value_t *value = acc.column->get_by_row(row_id);
                    if (value != nullptr) {
                        results[col_idx].append(*value);
                    } else {
                        results[col_idx].append_null();
                    }
                }
            }
        }

        batch.clear();
    }
};

ExecuteResult execute_hash_join(
    const Plan &plan, const JoinNode &join,
    const std::vector<std::tuple<size_t, DataType>> &output_attrs) {
    auto left_idx = join.left;
    auto right_idx = join.right;
    auto left = execute_impl(plan, left_idx);
    auto right = execute_impl(plan, right_idx);
    ExecuteResult results(output_attrs.size());

    const auto &left_col = left[join.left_attr];
    const auto &right_col = right[join.right_attr];

    bool build_left = left_col.row_count() <= right_col.row_count();
    size_t build_rows =
        build_left ? left_col.row_count() : right_col.row_count();
    for (size_t i = 0; i < output_attrs.size(); ++i) {
        results[i].reserve(build_rows);

        auto [col_idx, _] = output_attrs[i];
        if (col_idx < left.size()) {
            results[i].source_table = left[col_idx].source_table;
            results[i].source_column = left[col_idx].source_column;
        } else {
            results[i].source_table = right[col_idx - left.size()].source_table;
            results[i].source_column =
                right[col_idx - left.size()].source_column;
        }
    }

    JoinAlgorithm join_algorithm{
        .build = build_left ? left : right,
        .probe = build_left ? right : left,
        .results = results,
        .build_col = build_left ? join.left_attr : join.right_attr,
        .probe_col = build_left ? join.right_attr : join.left_attr,
        .swap = build_left,
        .output_attrs = output_attrs,
        .left_table_ref = nullptr,
        .right_table_ref = nullptr,
        .left_table_size = 0};

    join_algorithm.run();

    for (auto &col : results) {
        col.build_cache();
    }

    return results;
}

ExecuteResult
execute_scan(const Plan &plan, const ScanNode &scan,
             const std::vector<std::tuple<size_t, DataType>> &output_attrs) {
    auto table_id = scan.base_table_id;
    auto &input = plan.inputs[table_id];
    return mema::copy_scan(input, table_id, output_attrs);
}

ExecuteResult execute_impl(const Plan &plan, size_t node_idx) {
    auto &node = plan.nodes[node_idx];
    return std::visit(
        [&](const auto &value) -> ExecuteResult {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, JoinNode>) {
                return execute_hash_join(plan, value, node.output_attrs);
            } else {
                return execute_scan(plan, value, node.output_attrs);
            }
        },
        node.data);
}

ColumnarTable execute(const Plan &plan, void *context) {
    ExecuteResult ret = execute_impl(plan, plan.root);
    return mema::to_columnar(ret, plan);
}

void *build_context() { return nullptr; }

void destroy_context(void *context) {}

} // namespace Contest

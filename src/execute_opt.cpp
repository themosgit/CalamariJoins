#if defined(__APPLE__) && defined(__aarch64__)
#include <hardware_darwin.h>
#else
#include <hardware.h>
#endif

#include <hashtable.h>
#include <value_t_builders.h>

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

  void run() {
    left_table_ref = swap ? &build : &probe;
    right_table_ref = swap ? &probe : &build;
    left_table_size = left_table_ref->size();
    /* when the build table is small enough we do a loop */
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
  void hash_join() {
    mema::column_t &build_column = build[build_col];
    mema::column_t &probe_column = probe[probe_col];

    UnchainedHashtable hash_table(build_column.size());
    hash_table.build(build_column);

    const size_t probe_count = probe_column.row_count();
    const bool probe_direct = probe_column.has_direct_access();

    if (swap) {
      if (probe_direct) {
        for (size_t idx = 0; idx < probe_count; ++idx) {
          int32_t key_val = probe_column[idx].value;
          auto range = hash_table.find(key_val);
          const auto *start = range.first;
          const auto *end = range.second;

          for (const auto *ptr = start; ptr < end; ++ptr) {
            if (ptr->key == key_val) {
              construct_result(ptr->row_id, idx);
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
                construct_result(ptr->row_id, idx);
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
              construct_result(idx, ptr->row_id);
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
                construct_result(idx, ptr->row_id);
              }
            }
          }
        }
      }
    }
  }

  void nested_loop_join() {
    if (build.size() == 0 || probe.size() == 0 || build_col >= build.size() ||
        probe_col >= probe.size()) {
      return;
    }

    mema::column_t &build_column = build[build_col];
    mema::column_t &probe_column = probe[probe_col];
    const size_t build_count = build_column.row_count();
    const size_t probe_count = probe_column.row_count();

    const bool build_direct = build_column.has_direct_access();
    const bool probe_direct = probe_column.has_direct_access();

    if (build_direct && probe_direct) {
      if (swap) {
        for (size_t build_idx = 0; build_idx < build_count; build_idx++) {
          int build_value = build_column[build_idx].value;
          for (size_t probe_idx = 0; probe_idx < probe_count; probe_idx++) {
            if (probe_column[probe_idx].value == build_value) {
              construct_result(build_idx, probe_idx);
            }
          }
        }
      } else {
        for (size_t build_idx = 0; build_idx < build_count; build_idx++) {
          int build_value = build_column[build_idx].value;
          for (size_t probe_idx = 0; probe_idx < probe_count; probe_idx++) {
            if (probe_column[probe_idx].value == build_value) {
              construct_result(probe_idx, build_idx);
            }
          }
        }
      }
    } else {
      if (swap) {
        for (size_t build_idx = 0; build_idx < build_count; build_idx++) {
          const mema::value_t *build_key = build_column.get_by_row(build_idx);
          if (!build_key)
            continue;

          for (size_t probe_idx = 0; probe_idx < probe_count; probe_idx++) {
            const mema::value_t *probe_key = probe_column.get_by_row(probe_idx);
            if (probe_key && probe_key->value == build_key->value) {
              construct_result(build_idx, probe_idx);
            }
          }
        }
      } else {
        for (size_t build_idx = 0; build_idx < build_count; build_idx++) {
          const mema::value_t *build_key = build_column.get_by_row(build_idx);
          if (!build_key)
            continue;

          for (size_t probe_idx = 0; probe_idx < probe_count; probe_idx++) {
            const mema::value_t *probe_key = probe_column.get_by_row(probe_idx);
            if (probe_key && probe_key->value == build_key->value) {
              construct_result(probe_idx, build_idx);
            }
          }
        }
      }
    }
  }

  inline void construct_result(size_t left_row, size_t right_row) {
    size_t column_counter = 0;

    for (auto [col_idx, _] : output_attrs) {
      const mema::value_t *value;

      if (col_idx < left_table_size) {
        value = (*left_table_ref)[col_idx].get_by_row(left_row);
      } else {
        value =
            (*right_table_ref)[col_idx - left_table_size].get_by_row(right_row);
      }

      if (value != nullptr) {
        results[column_counter++].append(*value);
      } else {
        results[column_counter++].append_null();
      }
    }
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
  size_t build_rows = build_left ? left_col.row_count() : right_col.row_count();
  for (size_t i = 0; i < output_attrs.size(); ++i) {
    results[i].reserve(build_rows);
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

ColumnarTable execute(const Plan &plan, void* context) {
  ExecuteResult ret = execute_impl(plan, plan.root);
  return mema::to_columnar(ret, plan);
}

void *build_context() { 
    return nullptr;
}

void destroy_context(void* context) {}

} // namespace Contest

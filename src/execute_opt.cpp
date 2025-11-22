#include <cstdint>
#if defined(__APPLE__) && defined(__aarch64__)
    #include <hardware_darwin.h>
#else
    #include <hardware.h>
#endif

#include <hashtable.h>
#include <value_t_builders.h>

namespace Contest {

using ExecuteResult = std::vector<std::vector<value_t>>;

ExecuteResult execute_impl(const Plan& plan, size_t node_idx);

/**
 *
 *  join algorithm refactored for build table and probe table
 *  swap is previous built left maintains valid result format
 *
 **/
struct JoinAlgorithm {
    ExecuteResult&                                   build;
    ExecuteResult&                                   probe;
    ExecuteResult&                                   results;
    size_t                                           build_col, probe_col;
    bool                                             swap;
    const std::vector<std::tuple<size_t, DataType>>& output_attrs;

/**
 *
 * new function that calls nested_loop_join
 * to prevent creating hash tables for small tables
 *
 **/
public:
    template <class T>
    auto run() {
        /* 4 was chosen as an optimal value from tests */
        const size_t HASH_TABLE_THRESHOLD = 0;
        if (build.size() < HASH_TABLE_THRESHOLD) {
            nested_loop_join<T>();
        } else {
            hash_join<T>();
        }
    }

private: 

    template <class T>
    void hash_join() {
        namespace views = ranges::views;
        RobinHoodTable<T> hash_table(build.size() * 2);
        /* build hash table from build table */
        for (auto&& [idx, record]: build | views::enumerate) {
            __builtin_prefetch(&record + 128, 0, 1);
            auto key = &record[build_col].value;
            if (*key == INT32_MIN) {
                continue;
            }
            hash_table.insert(*key, idx);
        }

        const size_t probe_size = probe.size();
        for (size_t i = 0; i < probe_size; i++) {
            auto& probe_record = probe[i];
            __builtin_prefetch(&probe_record + 128, 0, 1);
            auto key = &probe_record[probe_col].value;
            if (*key == INT32_MIN) {
                continue;
            }
            auto indices = hash_table.find(*key);
            for (size_t build_index : indices) {
                swap ? construct_result(build[build_index], probe_record) :
                       construct_result(probe_record, build[build_index]);
            }
        }
    }


    template <class T>
    void nested_loop_join() {
        __builtin_prefetch(&build);
        for (auto& probe_record: probe) {
            __builtin_prefetch(&probe_record + 128);
            auto probe_key = probe_record[probe_col].value;
            if (probe_key == INT32_MIN)
                continue;
            for (size_t i = 0; i < build.size(); i++) {
                auto& build_record = build[i];
                auto build_key = build_record[build_col].value;
                if (build_key == INT32_MIN)
                    continue;
                if (build_key == probe_key) {
                    swap ? construct_result(build_record, probe_record) :
                           construct_result(probe_record, build_record);
                }
            }
        }
    }

    /**
     *
     *  constucts final result keeps left/right name because
     *  its relevant for constructing a valid result
     *
     **/
    void construct_result(const std::vector<value_t>& left_record, 
        const std::vector<value_t>& right_record) {
        std::vector<value_t> new_record;
        new_record.reserve(output_attrs.size());
        for (auto [col_idx, _]: output_attrs) {
            if (col_idx < left_record.size()) {
                new_record.emplace_back(left_record[col_idx]);
            } else {
                new_record.emplace_back(
                    right_record[col_idx - left_record.size()]);
            }
        }
        results.emplace_back(std::move(new_record));
    }
};

/** 
 *
 *  this is the function which calls our join algorithm it takes
 *  its arguments from execute_impl which takes its arguments from
 *  execute thats all i know for now
 *
 **/
ExecuteResult execute_hash_join(const Plan&          plan,
    const JoinNode&                                  join,
    const std::vector<std::tuple<size_t, DataType>>& output_attrs) {
    auto                           left_idx    = join.left;
    auto                           right_idx   = join.right;
    auto&                          left_node   = plan.nodes[left_idx];
    auto&                          right_node  = plan.nodes[right_idx];
    auto&                          left_types  = left_node.output_attrs;
    auto&                          right_types = right_node.output_attrs;
    auto                           left        = execute_impl(plan, left_idx);
    auto                           right       = execute_impl(plan, right_idx);
    std::vector<std::vector<value_t>> results;

    /**
     *
     *  it chooses which table will be built and which will be
     *  be probed the smaller one is chosen to be built to minimize
     *  hash table size
     *
     **/
    bool determine_build_left = (left.size() < right.size()) ? true : false;
    /* here we initialize the join algorithm with the desired tables */
    JoinAlgorithm join_algorithm{
    .build         = determine_build_left ? left  : right,
    .probe         = determine_build_left ? right : left,
    .results       = results,
    .build_col     = determine_build_left ? join.left_attr : join.right_attr,
    .probe_col     = determine_build_left ? join.right_attr : join.left_attr,
    .swap          = determine_build_left,
    .output_attrs  = output_attrs};

    join_algorithm.run<int32_t>();
    return results;
}

/* i have not investigated optimisations in the following functions */

ExecuteResult execute_scan(const Plan&               plan,
    const ScanNode&                                  scan,
    const std::vector<std::tuple<size_t, DataType>>& output_attrs) {
    auto                           table_id = scan.base_table_id;
    auto&                          input    = plan.inputs[table_id];
    return manolates::copy_scan(input, table_id, output_attrs);
}

ExecuteResult execute_impl(const Plan& plan, size_t node_idx) {
    auto& node = plan.nodes[node_idx];
    return std::visit(
        [&](const auto& value) -> ExecuteResult {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, JoinNode>) {
                return execute_hash_join(plan, value, node.output_attrs);
            } else {
                return execute_scan(plan, value, node.output_attrs);
            }
        },
        node.data);
}

ColumnarTable execute(const Plan& plan, void* context) {
    ExecuteResult ret = execute_impl(plan, plan.root);
    return manolates::to_columnar(ret, plan);
}

void* build_context() {
    return nullptr;
}

void destroy_context(void* context) {}

} // namespace Contest

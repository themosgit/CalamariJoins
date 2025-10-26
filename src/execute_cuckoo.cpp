#include <type_traits>
#include <variant>
#include <cstddef>
#if defined(__APPLE__) && defined(__aarch64__)
    #include <hardware_darwin.h>
#else
    #include <hardware.h>
#endif

#include <plan.h>
#include <table.h>
#include <cuckoo.h>

namespace Contest {

using ExecuteResult = std::vector<std::vector<Data>>;

ExecuteResult execute_impl(const Plan& plan, size_t node_idx);

/*join algorithm refactored for build table and probe table
 *swap is previous built left maintains valid result format*/
struct JoinAlgorithm {
    ExecuteResult&                                   build;
    ExecuteResult&                                   probe;
    ExecuteResult&                                   results;
    size_t                                           build_col, probe_col;
    bool                                             swap;
    const std::vector<std::tuple<size_t, DataType>>& output_attrs;

/* new function that calls nested_loop_join
 * to prevent creating hash tables for small tables*/
public:
    template <class T>
    auto run() {
        /* 4 was chosen as an optimal value from tests*/
        const size_t HASH_TABLE_THRESHOLD = 4;
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
        CuckooTable<T> hash_table(build.size() * 1.8);
        /*build hash table from build table*/
        for (auto&& [idx, record]: build | views::enumerate) {
            std::visit([&hash_table, idx = idx](const auto& key) {
                using Tk = std::decay_t<decltype(key)>;
                if constexpr (std::is_same_v<Tk, T>) {
                    hash_table.insert(key, idx);
                } else if constexpr (not std::is_same_v<Tk, std::monostate>) {
                    throw std::runtime_error("wrong type of field on build");
                }
            },
            record[build_col]);
        }

        // Removed prefetch logic for CuckooTable
        for (size_t i = 0; i < probe.size(); i++ ){
            auto& probe_record = probe[i];
            std::visit([&](const auto& key) {
                using Tk = std::decay_t<decltype(key)>;
                if constexpr (std::is_same_v<Tk, T>) {
                    // Changed to use CuckooTable's search interface
                    if (auto indices_opt = hash_table.search(key)) {
                        for (auto build_idx : *indices_opt.value()) {
                            swap ? construct_result(build[build_idx], probe_record):
                                construct_result(probe_record, build[build_idx]);
                        }
                    }
                } else if constexpr (not std::is_same_v<Tk, std::monostate>) {
                    throw std::runtime_error("wrong type of field on probe");
                }
            },
            probe_record[probe_col]);
        }
    }

   template <class T>
    void nested_loop_join() {
        for (auto& build_record: build) {
        for (auto& probe_record: probe) {
            std::visit([&](const auto& build_key, const auto& probe_key) {
                using Tb = std::decay_t<decltype(build_key)>;
                using Tp = std::decay_t<decltype(probe_key)>;
                if constexpr (std::is_same_v<Tb, T> && std::is_same_v<Tp, T>) {
                    if (build_key == probe_key) {
                        swap ? construct_result(build_record, probe_record) :
                               construct_result(probe_record, build_record);
                    }
                } else if constexpr (not std::is_same_v<Tb, std::monostate>
                                  || not std::is_same_v<Tp, std::monostate>) {
                    throw std::runtime_error("wrong type of field on probe");
                }
            }, build_record[build_col], probe_record[probe_col]);
        }
        }
    }

    /*constucts final result keeps left/right name because
     * its relevant for constructing a valid result*/
    void construct_result(const std::vector<Data>& left_record, 
        const std::vector<Data>& right_record) {
        std::vector<Data> new_record;
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

/* this is the function which calls our join algorithm it takes
 * its arguments from execute_impl which takes its arguments from
 * execute thats all i know for now */
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
    std::vector<std::vector<Data>> results;

    /*it chooses which table will be built and which will be
     *be probed the smaller one is chosen to be built to minimize
     *hash table size */
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

    if (determine_build_left) {
        switch (std::get<1>(left_types[join.left_attr])) {
        case DataType::INT32:   join_algorithm.run<int32_t>(); break;
        case DataType::INT64:   join_algorithm.run<int64_t>(); break;
        case DataType::FP64:    join_algorithm.run<double>(); break;
        case DataType::VARCHAR: join_algorithm.run<std::string>(); break;
        }
    } else {
        switch (std::get<1>(right_types[join.right_attr])) {
        case DataType::INT32:   join_algorithm.run<int32_t>(); break;
        case DataType::INT64:   join_algorithm.run<int64_t>(); break;
        case DataType::FP64:    join_algorithm.run<double>(); break;
        case DataType::VARCHAR: join_algorithm.run<std::string>(); break;
        }
    }
    return results;
}

/* i have not investigated optimisations in the following functions */

ExecuteResult execute_scan(const Plan&               plan,
    const ScanNode&                                  scan,
    const std::vector<std::tuple<size_t, DataType>>& output_attrs) {
    auto                           table_id = scan.base_table_id;
auto&                          input    = plan.inputs[table_id];
    return Table::copy_scan(input, output_attrs);
}

ExecuteResult execute_impl(const Plan& plan, size_t node_idx) {
    auto& node = plan.nodes[node_idx];
    return std::visit(
        [&](const auto& value) {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, JoinNode>) {
                return execute_hash_join(plan, value, node.output_attrs);
            } else {
                return execute_scan(plan, value, node.output_attrs);
            }
        },
        node.data);
}

ColumnarTable execute(const Plan& plan, [[maybe_unused]] void* context) {
    namespace views = ranges::views;
    auto ret        = execute_impl(plan, plan.root);
    auto ret_types  = plan.nodes[plan.root].output_attrs
                   | views::transform([](const auto& v) { return std::get<1>(v); })
                   | ranges::to<std::vector<DataType>>();
    Table table{std::move(ret), std::move(ret_types)};
    return table.to_columnar();
}

void* build_context() {
    return nullptr;
}

void destroy_context([[maybe_unused]] void* context) {}

} // namespace Contest

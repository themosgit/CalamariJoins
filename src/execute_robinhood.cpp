#if defined(__APPLE__) && defined(__aarch64__)
    #include <hardware_darwin.h>
#else
    #include <hardware.h>
#endif

#include <plan.h>
#include <table.h>
#include <iostream>
#include <robinhood.h>

namespace Contest {

using ExecuteResult = std::vector<std::vector<Data>>;

ExecuteResult execute_impl(const Plan& plan, size_t node_idx);

/*this is teh modified join algorithm most of the logic consistent
 * with the old implementation while improving on some obvious shortcomings
 * the left and right tables have been changed to build and probe
 * build is the table that we will hash this table is chosen to be the 
 * smaller of the two while probe is the table that we will then iterate
 * through and query our hash table on matching data the logic for chosing
 * the proper table to build or probe can be found in the execute_hash_join
 * function further down this file results is where we return our resutlts
 * build_col and probe_col are the columns in our tables from which we fetch
 * the join keys basically the relation and swap tells us whether they have
 * been swapped from the normal left for build right for probe to left for
 * probe left for build this info had to be done because the result expects
 * for the left table results to be first then the right table no matter their
 * size for which we swap them for to optimise our algorithm*/

struct JoinAlgorithm {
    ExecuteResult&                                   build; //smaller table in join
    ExecuteResult&                                   probe; //bigger table
    ExecuteResult&                                   results; // vector of vectors
    size_t                                           build_col, probe_col;
    bool                                             swap; 
    const std::vector<std::tuple<size_t, DataType>>& output_attrs; //

/* new function that calls nested_loop_join
 * to prevent creating hash tables for small tables
 * adjust hash table threshold for tests*/
public:
    template <class T>
    auto run() {
        const size_t HASH_TABLE_THRESHOLD = 4; //if build col has < 4 rows nested_loop_join
        if (build.size() < HASH_TABLE_THRESHOLD) {
            nested_loop_join<T>();
        } else {
            hash_join<T>(); // else hash_join
        }
    }

private: 
    /* our beloved hash join function changed ALOT but NOT ENOUGH!!!*/
    template <class T>
    void hash_join() {
        namespace views = ranges::views;
        /* the deafault unordered map hash table T is for the datatype key
         * see execute_hash_join for further details and the vector is there
         * to allow for buckets on collisions*/
        size_t build_size= build.size();
        build_size = build_size ? build_size : 1;//handle empty table
        size_t power_of_2 = 1; //use it to keep powerof2's
        while(power_of_2 * 0.60 < build_size){ // at least 40% empty but if build_table > result <<= 1 and us this result to build the hash_table
            power_of_2 <<= 1;
        }
        RobinHoodTable<T> hash_table(power_of_2);

        /*build hash table with the smaller table
         * iterate through table with record, idx is
         * created by views::enumerate as an index*/
        for (auto&& [idx, record]: build | views::enumerate) {
            std::visit(
                /* lambda function is called on the hash table and idx
                 *key is the join key extracted from record[build_col] down there*/
                [&hash_table, idx = idx](const auto& key) {
                    /*checks whether the type we called the function
                     * with T and the key in the column match Tk*/
                    using Tk = std::decay_t<decltype(key)>;
                    if constexpr (std::is_same_v<Tk, T>) {
                        /*if the key is not found in the hash table place insert it*/
                        hash_table.insert(key,idx);
                    /*exception for missmatching key types T and Tk*/
                    } else if constexpr (not std::is_same_v<Tk, std::monostate>) {
                        throw std::runtime_error("wrong type of field on build");
                    }
                },
                /*this is where the lambda extracts the key from*/
                record[build_col]
            );
        }
        /*probe larger table while doing
         *lookups on the prebuild hash table*/
        for (auto& probe_record: probe) {
            std::visit(
                /*lambda function that captures all values by reference
                 * !!READ ABOUT LAMBDAS take the data variant of the key as a 
                 * parameter thats why it is iterated on in probe record */
                [&](const auto& key) {
                    /*check whether the key datatype which the run fucntion was called T
                     * matches the key datatype used in the probe table Tk*/
                    using Tk = std::decay_t<decltype(key)>;
                    /*if the are the same continue else throw exception */
                    if constexpr (std::is_same_v<Tk, T>) {
                        /*get index from the hash table*/
                        auto result = hash_table.search(key);
                        /*if the key is not found exit the lambda*/
                        if (!result.has_value()) return;
                        /*because of linear probing we iterate
                         * through the vector*/
                        for (auto build_idx: **result) {
                            /*swap left and right if the have
                             * been swapped on function call 
                             * this is done to provide a valid result*/
                            switch(swap) {
                                case true: construct_result(build[build_idx], probe_record); break;
                                case false: construct_result(probe_record, build[build_idx]); break;
                            }
                        }
                    /*just an exception for missmatching Tk and T types*/
                    } else if constexpr (not std::is_same_v<Tk, std::monostate>) {
                        throw std::runtime_error("wrong type of field on probe");
                    }
                },
            probe_record[probe_col]);
        }
    }

    /*this is used for a nested loop join
     * we do nested loop joins when the
     * build table is of size 1
     * NOTE i dont like this piece
     * of code it will for sure be
     * changed in the future lil bro is nested to infinity
     * take this easy function as test to see if you
     * have understood anything at all */
   template <class T>
    void nested_loop_join() {
        for (auto& probe_record: probe) {
            for (auto& build_record: build) {
                std::visit([&](const auto& build_key) {
                    std::visit([&](const auto& probe_key) {
                        using Tb = std::decay_t<decltype(build_key)>;
                        using Tp = std::decay_t<decltype(probe_key)>;
                        if constexpr (std::is_same_v<Tb, T> && std::is_same_v<Tp, T>) {
                            if (build_key == probe_key) {
                                switch(swap) {
                                    case true: construct_result(build_record, probe_record); break;
                                    case false: construct_result(probe_record, build_record); break;
                                }
                            }
                        }
                    }, probe_record[probe_col]);
                }, build_record[build_col]);
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

    /*this is not the most readable piece of code but it works
     *it chooses which table will be built and which will be
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

    /* here we call the algorithm with the proper T type 
     * this is the type that will then be checked with Tk*/
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
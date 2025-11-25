#if defined(__APPLE__) && defined(__aarch64__)
    #include <hardware_darwin.h>
#else
    #include <hardware.h>
#endif

#include <hashtable.h>
#include <value_t_builders.h>

namespace Contest {

using ExecuteResult = std::vector<mema::column_t>;

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

    ExecuteResult* left_table_ref;
    ExecuteResult* right_table_ref;
    size_t left_table_size;

    auto run() {
        left_table_ref = swap ? &build : &probe;
        right_table_ref = swap ? &probe : &build;
        left_table_size = left_table_ref->size();
        
        const size_t HASH_TABLE_THRESHOLD = 4;
        if (build[build_col].size() < HASH_TABLE_THRESHOLD) {
            nested_loop_join();
        } else {
            hash_join();
        }
    }

private: 
    void hash_join() {
        RobinHoodTable hash_table(build[build_col].size() * 2);
        mema::column_t& build_column = build[build_col];
        const size_t build_count = build_column.row_count();
        for (size_t idx = 0; idx < build_count; idx++) {
            const mema::value_t* key = build_column.get_by_row(idx);
            if (key) hash_table.insert(key->value, idx);
        }
        
        mema::column_t& probe_column = probe[probe_col];
        const size_t probe_count = probe_column.row_count();
        if (swap) {
            for (size_t idx = 0; idx < probe_count; idx++) {
                const mema::value_t* key = probe_column.get_by_row(idx);
                if (!key) continue;
                
                auto indices = hash_table.find(key->value);
                for (size_t build_row_idx : indices) {
                    construct_result(build_row_idx, idx);
                }
            }
        } else {
            for (size_t idx = 0; idx < probe_count; idx++) {
                const mema::value_t* key = probe_column.get_by_row(idx);
                if (!key) continue;
                
                auto indices = hash_table.find(key->value);
                for (size_t build_row_idx : indices) {
                    construct_result(idx, build_row_idx);
                }
            }
        }
    }

    void nested_loop_join() {
        if (build.size() == 0 || probe.size() == 0 || 
            build_col >= build.size() || probe_col >= probe.size()) {
            return;
        }
        
        mema::column_t& build_column = build[build_col];
        mema::column_t& probe_column = probe[probe_col];
        const size_t build_count = build_column.row_count();
        const size_t probe_count = probe_column.row_count();
        
        if (swap) {
            for (size_t build_idx = 0; build_idx < build_count; build_idx++) {
                const mema::value_t* build_key = build_column.get_by_row(build_idx);
                if (!build_key) continue;
                
                for (size_t probe_idx = 0; probe_idx < probe_count; probe_idx++) {
                    const mema::value_t* probe_key = probe_column.get_by_row(probe_idx);
                    if (probe_key && probe_key->value == build_key->value) {
                        construct_result(build_idx, probe_idx);
                    }
                }
            }
        } else {
            for (size_t build_idx = 0; build_idx < build_count; build_idx++) {
                const mema::value_t* build_key = build_column.get_by_row(build_idx);
                if (!build_key) continue;
                
                for (size_t probe_idx = 0; probe_idx < probe_count; probe_idx++) {
                    const mema::value_t* probe_key = probe_column.get_by_row(probe_idx);
                    if (probe_key && probe_key->value == build_key->value) {
                        construct_result(probe_idx, build_idx);
                    }
                }
            }
        }
    }

    /**
     *
     *  improved result construction using references from
     *  initially set when run is called.
     *
     **/
    void construct_result(size_t left_row, size_t right_row) {
        size_t column_counter = 0;
        
        for (auto [col_idx, _] : output_attrs) {
            const mema::value_t* value;
            
            if (col_idx < left_table_size) {
                value = (*left_table_ref)[col_idx].get_by_row(left_row);
            } else {
                value = (*right_table_ref)[col_idx - left_table_size].get_by_row(right_row);
            }
            
            if (value) {
                results[column_counter++].append(*value);
            } else {
                results[column_counter++].append_null();
            }
        }
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
    ExecuteResult results(output_attrs.size());
    /**
     *
     *  it chooses which table will be built and which will be
     *  be probed the smaller one is chosen to be built to minimize
     *  hash table size
     *
     **/

    bool determine_build_left = (left.size() < right.size()) ? true : false;
    for (int i = 0; i < output_attrs.size(); i++)
        results[i].reserve(determine_build_left ? right.size() : left.size());

    JoinAlgorithm join_algorithm{
    .build          = determine_build_left ? left  : right,
    .probe         = determine_build_left ? right : left,
    .results       = results,
    .build_col     = determine_build_left ? join.left_attr : join.right_attr,
    .probe_col     = determine_build_left ? join.right_attr : join.left_attr,
    .swap          = determine_build_left,
    .output_attrs  = output_attrs};
    

    join_algorithm.run();
    for (auto& col : results)
        col.build_cache();

    return results;
}


ExecuteResult execute_scan(const Plan&               plan,
    const ScanNode&                                  scan,
    const std::vector<std::tuple<size_t, DataType>>& output_attrs) {
    auto                           table_id = scan.base_table_id;
    auto&                          input    = plan.inputs[table_id];
    return mema::copy_scan(input, table_id, output_attrs);
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
    return mema::to_columnar(ret, plan);
}

void* build_context() {
    return nullptr;
}

void destroy_context(void* context) {}
};

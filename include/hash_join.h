#pragma once

#include <construct_intermediate.h>
#include <hashtable.h>
#include <intermediate.h>
#include <join_setup.h>
#include <worker_pool.h>
#include <match_collector.h>

/**
 *
 *  hash join operations for building and probing hash tables
 *
 *  this header contains the build and probe phases of hash joins
 *  supports both ColumnarTable columnar and column_t intermediate inputs
 *
 *  build phase constructs unchained hash table with bloom filters
 *  probe phase searches hash table and collects matches
 *
 **/

namespace Contest {

/**
 *
 *  builds hash table from ColumnarTable columnar input
 *  reads directly from original page format
 *  constructs unchained hash table with bloom filters
 *
 **/
inline UnchainedHashtable build_from_columnar(const JoinInput &input,
                                              size_t attr_idx) {
    auto *table = std::get<const ColumnarTable *>(input.data);
    auto [actual_col_idx, _] = input.node->output_attrs[attr_idx];
    const Column &column = table->columns[actual_col_idx];

    size_t row_count = input.row_count(attr_idx);
    UnchainedHashtable hash_table(row_count);
    hash_table.build_columnar(column, 8);

    return hash_table;
}

inline UnchainedHashtable build_from_columnar_parallel(const JoinInput &input,
                                                       size_t attr_idx) {
    auto *table = std::get<const ColumnarTable *>(input.data);
    auto [actual_col_idx, _] = input.node->output_attrs[attr_idx];
    const Column &column = table->columns[actual_col_idx];

    size_t row_count = input.row_count(attr_idx);
    UnchainedHashtable hash_table(row_count);
    hash_table.build_columnar(column);

    return hash_table;
}

/**
 *
 *  builds hash table from column_t intermediate results
 *  works with intermediate results from previous joins
 *  constructs unchained hash table with bloom filters
 *
 **/
inline UnchainedHashtable build_from_intermediate(const JoinInput &input,
                                                  size_t attr_idx) {
    const auto &result = std::get<ExecuteResult>(input.data);
    const auto &column = result[attr_idx];

    size_t row_count = input.row_count(attr_idx);
    UnchainedHashtable hash_table(row_count);
    hash_table.build_intermediate(column, 8);

    return hash_table;
}

inline UnchainedHashtable build_from_intermediate_parallel(const JoinInput &input,
                                                           size_t attr_idx) {
    const auto &result = std::get<ExecuteResult>(input.data);
    const auto &column = result[attr_idx];

    size_t row_count = input.row_count(attr_idx);
    UnchainedHashtable hash_table(row_count);
    hash_table.build_intermediate(column);

    return hash_table;
}

/**
 *
 *  merges thread-local match collectors into global collector
 *  single-threaded aggregation after parallel probing phase
 *  appends all local matches sequentially
 *
 **/

inline void merge_local_collectors(
    std::vector<ThreadLocalMatchBuffer>& local_buffers,
    MatchCollector& global_collector) {
    for(auto& buf : local_buffers){
        global_collector.merge_thread_buffer(buf);
    }
}

/**
 *
 *  parallel probing of hash table with intermediate column_t input using work stealing
 *  each thread processes pages via atomic counter for dynamic load balancing
 *  leverages fixed-size pages for simple row offset calculation
 *  thread-local collectors merged after join for lock-free execution
 *  skips null values during probing
 *
 **/
inline void probe_intermediate(const UnchainedHashtable& hash_table,
                               const mema::column_t &probe_column,
                               MatchCollector& collector) {
    const auto* keys = hash_table.keys();
    const auto* row_ids = hash_table.row_ids();

    size_t pool_size = worker_pool.thread_count();
    // Use the new ThreadLocalMatchBuffer
    std::vector<ThreadLocalMatchBuffer> local_buffers(pool_size);
    
    const size_t num_pages = probe_column.pages.size();
    const size_t probe_count = probe_column.row_count();
    std::atomic<size_t> page_counter(0);

    worker_pool.execute([&](size_t thread_id, size_t /* total_threads */) {
        auto& local_buf = local_buffers[thread_id];
        
        while(true){
            size_t page_idx = page_counter.fetch_add(1);
            if(page_idx >= num_pages) break;

            size_t base_row = page_idx * mema::CAP_PER_PAGE;
            size_t page_end = std::min(base_row + mema::CAP_PER_PAGE, probe_count);

            for(size_t idx = base_row; idx < page_end; ++idx){
                const mema::value_t& key = probe_column[idx];

                if(!key.is_null()){
                    int32_t key_val = key.value;
                    auto[start_idx,end_idx] = hash_table.find_indices(key_val);

                    for(uint64_t i = start_idx; i < end_idx; ++i){
                        if(keys[i] == key_val){
                            // Optimized add
                            local_buf.add_match(row_ids[i], idx);
                        }
                    }
                }
            }
        }
    });

    merge_local_collectors(local_buffers, collector);
}

inline void probe_columnar(const UnchainedHashtable& hash_table,
                           const JoinInput& probe_input,
                           size_t probe_attr,
                           MatchCollector& collector) {

    const auto* keys = hash_table.keys();
    const auto* row_ids = hash_table.row_ids();
                                
    auto* table = std::get<const ColumnarTable*>(probe_input.data);
    auto [actual_idx_col,_] = probe_input.node->output_attrs[probe_attr];
    const Column& probe_col = table->columns[actual_idx_col];
    size_t num_pages = probe_col.pages.size();

    std::vector<uint32_t> page_offsets;
    page_offsets.reserve(num_pages);
    uint32_t running_offset = 0;

    for(const auto* page_obj : probe_col.pages){
        page_offsets.push_back(running_offset);
        auto num_rows = *reinterpret_cast<const uint16_t*>(page_obj->data);
        running_offset += num_rows;
    }

    size_t pool_size = worker_pool.thread_count();
    std::vector<ThreadLocalMatchBuffer> local_buffers(pool_size);

    std::atomic<size_t> page_counter(0);
    worker_pool.execute([&](size_t thread_id, size_t /* total_threads */) {
        auto& local_buf = local_buffers[thread_id];

        while(true){
            size_t page_idx = page_counter.fetch_add(1);
            if(page_idx >= num_pages) break;
            
            auto* page = probe_col.pages[page_idx]->data;
            auto num_rows = *reinterpret_cast<const uint16_t*>(page);
            auto num_values = *reinterpret_cast<const uint16_t*>(page + 2);
            auto *data_begin = reinterpret_cast<const int32_t *>(page + 4);
            uint32_t probe_row_id = page_offsets[page_idx];

            if(num_rows == num_values) { 
                for(uint16_t  i = 0; i < num_rows; ++i){
                    int32_t key_val = data_begin[i];
                    auto [start_idx, end_idx] = hash_table.find_indices(key_val);

                    for(uint64_t j = start_idx; j < end_idx; ++j){
                        if(keys[j] == key_val){
                            local_buf.add_match(row_ids[j], probe_row_id);
                        }
                    }
                    probe_row_id++;
                }
            } else {
                auto *bitmap = reinterpret_cast<const uint8_t *>(
                    page + PAGE_SIZE - (num_rows + 7) / 8 );
                uint16_t data_idx = 0;
                for(uint16_t i = 0; i < num_rows; ++i){
                    bool is_valid = bitmap[i/8] & (1u << (i % 8));
                    if(is_valid){
                        int32_t key_val = data_begin[data_idx++];
                        auto[start_idx, end_idx] = 
                            hash_table.find_indices(key_val);

                        for(uint64_t j = start_idx; j < end_idx; ++j){
                            if(keys[j] == key_val){
                                local_buf.add_match(row_ids[j], probe_row_id);
                            }
                        }
                    }
                    probe_row_id++;
                }
            }
        }
    });

    merge_local_collectors(local_buffers, collector);
}
} // namespace Contest

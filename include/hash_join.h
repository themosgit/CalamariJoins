#pragma once

#include <construct_intermediate.h>
// #if defined(__aarch64__)
//     #include <hardware_darwin.h>
// #else
//     #include <hardware.h>
// #endif
#include <hashtable.h>
#include <intermediate.h>
#include <join_setup.h>

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
 *  probes hash table with values from ColumnarTable columnar input
 *  reads directly from original page format
 *  handles both dense and sparse pages with bitmaps
 *  collects matches into MatchCollector for materialization
 *
 **/
inline void probe_columnar(const UnchainedHashtable &hash_table,
                           const JoinInput &probe_input, size_t probe_attr,
                           MatchCollector &collector) {

    const auto *keys = hash_table.keys();
    const auto *row_ids = hash_table.row_ids();


    auto *table = std::get<const ColumnarTable *>(probe_input.data);
    auto [actual_col_idx, _] = probe_input.node->output_attrs[probe_attr];
    const Column &probe_col = table->columns[actual_col_idx];

    uint32_t probe_row_id = 0;
    for (auto *page_obj : probe_col.pages) {
        auto *page = page_obj->data;
        auto num_rows = *reinterpret_cast<const uint16_t *>(page);
        auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
        auto *data_begin = reinterpret_cast<const int32_t *>(page + 4);

        if (num_rows == num_values) {
            for (uint16_t i = 0; i < num_rows; ++i) {
                int32_t key_val = data_begin[i];
                auto [start_idx, end_idx] = hash_table.find_indices(key_val);

                for (uint64_t j = start_idx; j < end_idx; ++j) {
                    if (keys[j] == key_val) {
                        collector.add_match(row_ids[j], probe_row_id);
                    }
                }
                probe_row_id++;
            }
        } else {
            auto *bitmap = reinterpret_cast<const uint8_t *>(
                page + PAGE_SIZE - (num_rows + 7) / 8);
            uint16_t data_idx = 0;
            for (uint16_t i = 0; i < num_rows; ++i) {
                bool is_valid = bitmap[i / 8] & (1u << (i % 8));
                if (is_valid) {
                    int32_t key_val = data_begin[data_idx++];
                    auto [start_idx, end_idx] =
                        hash_table.find_indices(key_val);

                    for (uint64_t j = start_idx; j < end_idx; ++j) {
                        if (keys[j] == key_val) {
                            collector.add_match(row_ids[j], probe_row_id);
                        }
                    }
                }
                probe_row_id++;
            }
        }
    }
}

/**
 *
 *  probes hash table with values from column_t intermediate results
 *  handles both direct access and sparse column_t formats
 *  skips null values in sparse columns
 *  collects matches into MatchCollector for materialization
 *
 **/
inline void probe_intermediate(const UnchainedHashtable &hash_table,
                               const mema::column_t &probe_column,
                               MatchCollector &collector) {
    const auto *keys = hash_table.keys();
    const auto *row_ids = hash_table.row_ids();

    const size_t probe_count = probe_column.row_count();

    for (size_t idx = 0; idx < probe_count; ++idx) {
        const mema::value_t &key = probe_column[idx];
        if (!key.is_null()) {
            int32_t key_val = key.value;
            auto [start_idx, end_idx] = hash_table.find_indices(key_val);

            for (uint64_t i = start_idx; i < end_idx; ++i) {
                if (keys[i] == key_val) {
                    collector.add_match(row_ids[i], idx);
                }
            }
        }
    }
}
inline void merge_local_collectors(
    const std::vector<MatchCollector>& local_collectors,
    MatchCollector& global_collector)
{
    size_t total = 0;
    for(const auto& lc : local_collectors) total += lc.size();

    global_collector.reserve(total);

    for(const auto& lc : local_collectors){
        global_collector.matches.insert(
            global_collector.matches.end(),
            lc.matches.begin(),
            lc.matches.end()
        );

    }
}

inline void probe_columnar_parallel(const UnchainedHashtable& hash_table,
                                    const JoinInput& probe_input,
                                    size_t probe_attr,
                                    MatchCollector& collector,int num_threads = NUM_CORES){

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

    std::vector<MatchCollector> local_collectors(num_threads);

    std::atomic<size_t> page_counter(0);

    std::vector<std::thread> workers;
    for(int t = 0; t < num_threads; ++t){
        workers.emplace_back([&,t](){
            while(true){
                size_t page_idx = page_counter.fetch_add(1);
                if(page_idx >= num_pages) break;
                auto* page = probe_col.pages[page_idx]->data; //gets only one page
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
                                local_collectors[t].add_match(row_ids[j],probe_row_id);
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
                                    local_collectors[t].add_match(row_ids[j],probe_row_id);
                                }
                            }
                        }
                        probe_row_id++;
                    }
                }
            }
        });
    }
    for(auto& w: workers) w.join();
    merge_local_collectors(local_collectors,collector);

}
} // namespace Contest

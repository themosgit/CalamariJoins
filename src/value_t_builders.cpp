#include <cstring>
#include <value_t_builders.h>
#include <inner_column.h>

namespace manolates {

    using row_store = std::vector<std::vector<value_t>>;

    inline bool get_bitmap(const uint8_t* bitmap, uint16_t idx) {
        return bitmap[idx / 8] & (1u << (idx % 8));
    }

    inline void set_bitmap(std::vector<uint8_t>& bitmap, uint16_t idx) {
        while (bitmap.size() < idx / 8 + 1) {
            bitmap.emplace_back(0);
        }
        auto byte_idx     = idx / 8;
        auto bit          = idx % 8;
        bitmap[byte_idx] |= (1u << bit);
    }

    inline void unset_bitmap(std::vector<uint8_t>& bitmap, uint16_t idx) {
        while (bitmap.size() < idx / 8 + 1) {
            bitmap.emplace_back(0);
        }
        auto byte_idx     = idx / 8;
        auto bit          = idx % 8;
        bitmap[byte_idx] &= ~(1u << bit);
    }


    row_store copy_scan(const ColumnarTable& table, uint8_t table_idx, 
                       const std::vector<std::tuple<size_t, DataType>>& output_attrs) {
        row_store results(table.num_rows, std::vector<value_t>(output_attrs.size()));
        
        /**
         *
         *  reads from output_attrs the columns we need and materializes
         *  only those. for ints we store by value for strings
         *  we store the necessary fields talked about in the handout
         *
         **/
        for (size_t out_col_idx = 0; out_col_idx < output_attrs.size(); ++out_col_idx) {
            auto [col_idx, col_type] = output_attrs[out_col_idx];
            auto& column = table.columns[col_idx];
            size_t row_idx = 0;
            
            switch (column.type) {
                case DataType::INT32: {
                    for (auto* page_obj : column.pages) {
                        /* page arithmetic */
                        auto* page = page_obj->data;
                        auto num_rows = *reinterpret_cast<uint16_t*>(page);
                        auto* data_begin = reinterpret_cast<int32_t*>(page + 4);
                        auto* bitmap = reinterpret_cast<uint8_t*>
                            (page + PAGE_SIZE - (num_rows + 7) / 8);
                        uint16_t data_idx = 0;
                        
                        for (uint16_t i = 0; i < num_rows; ++i) {
                            if (get_bitmap(bitmap, i)) {
                                results[row_idx][out_col_idx]
                                    = {data_begin[data_idx++], 0, 0, 0};
                            }
                            row_idx++;
                        }
                    }
                    break;
                }
                /**
                 *
                 *  for strings we store the necessary metadata
                 *  if it is a long string we only store
                 *  the page_idx and we have a barrier value
                 *  to identify them
                 *
                 **/
                case DataType::VARCHAR: {
                    int page_idx = 0;
                    for (auto* page_obj : column.pages) {
                        auto* page = page_obj->data;
                        auto num_rows = *reinterpret_cast<uint16_t*>(page);
                        if (num_rows == 0xffff) {
                            results[row_idx++][out_col_idx] = 
                                {page_idx, table_idx, static_cast<uint8_t>(col_idx), UINT16_MAX};
                        } 
                        else if (num_rows == 0xfffe) {
                        } 
                        else {
                            auto* bitmap = reinterpret_cast<uint8_t*>
                                (page + PAGE_SIZE - (num_rows + 7) / 8);
                            uint16_t offset_idx = 0;
                            for (uint16_t i = 0; i < num_rows; ++i) {
                                if (get_bitmap(bitmap, i)) {
                                    results[row_idx][out_col_idx] =
                                        {page_idx, table_idx, static_cast<uint8_t>(col_idx), offset_idx++};
                                }
                                row_idx++;
                            }
                        }
                        page_idx++;
                    }
                    break;
                }
                
                default:
                    break;
            }
        }
        return results;
    }



    ColumnarTable to_columnar(const row_store& table, const Plan& plan) {
        ColumnarTable ret;
        ret.num_rows = table.size();
        
        auto& root_node = plan.nodes[plan.root];
        size_t num_cols = root_node.output_attrs.size();

        /* when the table is empty it still requires empty columns */
        if (table.empty()) {
            for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
                DataType data_type = std::get<1>(root_node.output_attrs[col_idx]);
                ret.columns.emplace_back(data_type);
            }
            return ret;
        }

        /**
         *
         *  to materialize values the strategy is similar
         *  to the one used in to_columnar the difference
         *  being that we need to materialize strings
         *  for int the values our stored directly fo the process
         *  is identical.
         *
         **/
        for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
            DataType data_type = std::get<1>(root_node.output_attrs[col_idx]);
            ret.columns.emplace_back(data_type);
            auto& column = ret.columns.back();
            
            switch (data_type) {
            case DataType::INT32: {
                uint16_t num_rows = 0;
                std::vector<int32_t> data;
                std::vector<uint8_t> bitmap;
                data.reserve(2048);
                bitmap.reserve(256);
                
                auto save_page = [&column, &num_rows, &data, &bitmap]() {
                    auto* page = column.new_page()->data;
                    *reinterpret_cast<uint16_t*>(page) = num_rows;
                    *reinterpret_cast<uint16_t*>(page + 2) = static_cast<uint16_t>(data.size());
                    memcpy(page + 4, data.data(), data.size() * 4);
                    memcpy(page + PAGE_SIZE - bitmap.size(), bitmap.data(), bitmap.size());
                    num_rows = 0;
                    data.clear();
                    bitmap.clear();
                };
                
                for (const auto& record : table) {
                    const auto& val = record[col_idx];
                    if (val.value == INT32_MIN) {
                        if (4 + data.size() * 4 + (num_rows / 8 + 1) > PAGE_SIZE) {
                            save_page();
                        }
                        unset_bitmap(bitmap, num_rows);
                        ++num_rows;
                    } else {
                        if (4 + (data.size() + 1) * 4 + (num_rows / 8 + 1) > PAGE_SIZE) {
                            save_page();
                        }
                        set_bitmap(bitmap, num_rows);
                        data.emplace_back(val.value);
                        ++num_rows;
                    }
                }
                if (num_rows != 0) save_page();
                break;
            }

            /**
             *
             *  to return to columnar
             *  we first materialize the string
             *  and then use the method shown in the original
             *  to_columnar.
             *
             *  For the materialization we use all the 
             *  metadata stored in value_t
             *
             *  we first index the table then the column
             *  and then the page.
             *
             *  To reconstuct small strings we use the offset
             *  and look at the offset previous to ours to find
             *  our range within the page.
             *  (end_of_prev_string -- end_of_current_string).
             *
             *  for big strings we detect via the sentinel
             *  UINT16_MAX value we copy the contents and
             *  then read subsequent pages until they are
             *  not 0xfffe
             *
             **/
            case DataType::VARCHAR: {
                uint16_t num_rows = 0;
                std::vector<char> data;
                std::vector<uint16_t> offsets;
                std::vector<uint8_t> bitmap;
                data.reserve(8192);
                offsets.reserve(4096);
                bitmap.reserve(512);
                
                auto save_long_string = [&column](const std::string& str) {
                    size_t offset = 0;
                    auto first_page = true;
                    while (offset < str.size()) {
                        auto* page = column.new_page()->data;
                        if (first_page) {
                            *reinterpret_cast<uint16_t*>(page) = 0xffff;
                            first_page = false;
                        } else {
                            *reinterpret_cast<uint16_t*>(page) = 0xfffe;
                        }
                        auto page_data_len = std::min(str.size() - offset, PAGE_SIZE - 4);
                        *reinterpret_cast<uint16_t*>(page + 2) = static_cast<uint16_t>(page_data_len);
                        memcpy(page + 4, str.data() + offset, page_data_len);
                        offset += page_data_len;
                    }
                };
                
                auto save_page = [&column, &num_rows, &data, &offsets, &bitmap]() {
                    auto* page = column.new_page()->data;
                    *reinterpret_cast<uint16_t*>(page) = num_rows;
                    *reinterpret_cast<uint16_t*>(page + 2) = static_cast<uint16_t>(offsets.size());
                    memcpy(page + 4, offsets.data(), offsets.size() * 2);
                    memcpy(page + 4 + offsets.size() * 2, data.data(), data.size());
                    memcpy(page + PAGE_SIZE - bitmap.size(), bitmap.data(), bitmap.size());
                    num_rows = 0;
                    data.clear();
                    offsets.clear();
                    bitmap.clear();
                };
                
                for (const auto& record : table) {
                    const auto& val = record[col_idx];
                    
                    if (val.value == INT32_MIN) {
                        if (4 + offsets.size() * 2 + data.size() + (num_rows / 8 + 1) > PAGE_SIZE) {
                            save_page();
                        }
                        unset_bitmap(bitmap, num_rows);
                        ++num_rows;
                    } else {
                        std::string str;
                        const auto& src_table = plan.inputs[val.table];
                        const auto& src_col = src_table.columns[val.column];
                        
                        if (val.offset == UINT16_MAX) {
                            size_t page_idx = val.value;
                            auto* p = reinterpret_cast<uint8_t*>(src_col.pages[page_idx]->data);
                            uint16_t len = *reinterpret_cast<uint16_t*>(p + 2);
                            str.append(reinterpret_cast<char*>(p + 4), len);
                            page_idx++;
                            
                            while (page_idx < src_col.pages.size()) {
                                p = reinterpret_cast<uint8_t*>(src_col.pages[page_idx]->data);
                                if (*reinterpret_cast<uint16_t*>(p) != 0xfffe) break;
                                
                                len = *reinterpret_cast<uint16_t*>(p + 2);
                                str.append(reinterpret_cast<char*>(p + 4), len);
                                page_idx++;
                            }
                        } else {
                            int page_idx = val.value;
                            auto* page = reinterpret_cast<uint8_t*>(src_col.pages[page_idx]->data);
                            auto num_valid = *reinterpret_cast<uint16_t*>(page + 2);
                            auto* offset_array = reinterpret_cast<uint16_t*>(page + 4);
                            char* char_begin = reinterpret_cast<char*>(page + 4 + num_valid * 2);
                            
                            uint16_t end_off = offset_array[val.offset];
                            uint16_t start_off = (val.offset == 0) ? 0 : offset_array[val.offset - 1];
                            
                            str.assign(char_begin + start_off, end_off - start_off);
                        }
                        
                        if (str.size() > PAGE_SIZE - 7) {
                            if (num_rows > 0) {
                                save_page();
                            }
                            save_long_string(str);
                        } else {
                            if (4 + (offsets.size() + 1) * 2 + (data.size() + str.size()) 
                                + (num_rows / 8 + 1) > PAGE_SIZE) {
                                save_page();
                            }
                            set_bitmap(bitmap, num_rows);
                            data.insert(data.end(), str.begin(), str.end());
                            offsets.emplace_back(data.size());
                            ++num_rows;
                        }
                    }
                }
                
                if (num_rows != 0) {
                    save_page();
                }
                break;
            }
            
            default:
                break;
            }
        }
        return ret;
    }
}

#include <columnar_structs.h>
#include <cstring>
#include <inner_column.h>

namespace mema {

using columnar = std::vector<column_t>;

static void set_bitmap_range(std::vector<uint8_t> &bitmap, uint16_t start_row,
                             uint16_t end_row) {
    size_t start_byte = start_row / 8;
    size_t end_byte = (end_row - 1) / 8;

    while (bitmap.size() <= end_byte) {
        bitmap.push_back(0);
    }

    if (start_byte == end_byte) {
        for (uint16_t bit = start_row % 8; bit < end_row % 8; ++bit) {
            bitmap[start_byte] |= (1 << bit);
        }
    } else {
        for (uint16_t bit = start_row % 8; bit < 8; ++bit) {
            bitmap[start_byte] |= (1 << bit);
        }
        for (size_t b = start_byte + 1; b < end_byte; ++b) {
            bitmap[b] = 0xFF;
        }
        for (uint16_t bit = 0; bit < end_row % 8; ++bit) {
            bitmap[end_byte] |= (1 << bit);
        }
    }
}

static std::string read_normal_string(const Column &src_col, int32_t page_idx,
                                      int32_t offset_idx) {
    auto *page = reinterpret_cast<uint8_t *>(src_col.pages[page_idx]->data);
    auto num_valid = *reinterpret_cast<uint16_t *>(page + 2);
    auto *offset_array = reinterpret_cast<uint16_t *>(page + 4);
    char *char_begin = reinterpret_cast<char *>(page + 4 + num_valid * 2);

    uint16_t end_off = offset_array[offset_idx];
    uint16_t start_off = (offset_idx == 0) ? 0 : offset_array[offset_idx - 1];

    return std::string(char_begin + start_off, end_off - start_off);
}

inline bool get_bitmap(const uint8_t *bitmap, uint16_t idx) {
    return bitmap[idx / 8] & (1u << (idx % 8));
}

inline void ensure_bitmap_size(std::vector<uint8_t> &bitmap, uint16_t idx) {
    while (bitmap.size() < idx / 8 + 1) {
        bitmap.emplace_back(0);
    }
}

inline void set_bitmap(std::vector<uint8_t> &bitmap, uint16_t idx) {
    ensure_bitmap_size(bitmap, idx);
    auto byte_idx = idx / 8;
    auto bit = idx % 8;
    bitmap[byte_idx] |= (1u << bit);
}

inline void unset_bitmap(std::vector<uint8_t> &bitmap, uint16_t idx) {
    ensure_bitmap_size(bitmap, idx);
    auto byte_idx = idx / 8;
    auto bit = idx % 8;
    bitmap[byte_idx] &= ~(1u << bit);
}

Columnar
copy_scan(const ColumnarTable &table, uint8_t table_idx,
          const std::vector<std::tuple<size_t, DataType>> &output_attrs) {
    Columnar results;
    /**
     *
     * reads from output_attrs the columns we need and materializes
     * only those. for ints we store by value for strings
     * we store the necessary fields talked about in the handout
     *
     **/
    results.resize(output_attrs.size());

    for (size_t out_col_idx = 0; out_col_idx < output_attrs.size();
         ++out_col_idx) {
        auto [col_idx, col_type] = output_attrs[out_col_idx];
        auto &column = table.columns[col_idx];
        auto &result_col = results[out_col_idx];

        result_col.source_table = table_idx;
        result_col.source_column = static_cast<uint8_t>(col_idx);
        result_col.reserve(table.num_rows);
        switch (column.type) {
        case DataType::INT32: {
            for (auto *page_obj : column.pages) {
                /* page arithmetic */
                auto *page = page_obj->data;
                auto num_rows = *reinterpret_cast<uint16_t *>(page);
                auto num_values = *reinterpret_cast<uint16_t *>(page + 2);
                auto *data_begin = reinterpret_cast<int32_t *>(page + 4);

                /* check if dense */
                if (num_rows == num_values) {
                    result_col.append_bulk(data_begin, num_rows);
                } else {
                    auto *bitmap = reinterpret_cast<uint8_t *>(
                        page + PAGE_SIZE - (num_rows + 7) / 8);
                    uint16_t data_idx = 0;
                    for (uint16_t i = 0; i < num_rows; ++i) {
                        if (get_bitmap(bitmap, i)) {
                            result_col.append({data_begin[data_idx++]});
                        } else {
                            result_col.append_null();
                        }
                    }
                }
            }
            break;
        }
        /**
         *
         * for strings we store the necessary metadata
         * if it is a long string we only store
         * the page_idx and we have a barrier value
         * to identify them
         *
         **/
        case DataType::VARCHAR: {
            int page_idx = 0;
            for (auto *page_obj : column.pages) {
                auto *page = page_obj->data;
                auto num_rows = *reinterpret_cast<uint16_t *>(page);
                if (num_rows == 0xffff) {
                    result_col.append(value_t::encode_string(
                        page_idx, value_t::LONG_STRING_OFFSET));
                } else if (num_rows == 0xfffe) {
                } else {
                    /* check if dense */
                    auto num_values = *reinterpret_cast<uint16_t *>(page + 2);

                    if (num_rows == num_values) {
                        result_col.append_bulk_string(page_idx, num_rows);
                    } else {
                        auto *bitmap = reinterpret_cast<uint8_t *>(
                            page + PAGE_SIZE - (num_rows + 7) / 8);
                        uint16_t offset_idx = 0;
                        for (uint16_t i = 0; i < num_rows; ++i) {
                            if (get_bitmap(bitmap, i)) {
                                result_col.append(value_t::encode_string(
                                    page_idx, offset_idx++));
                            } else {
                                result_col.append_null();
                            }
                        }
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

    for (auto &col : results) {
        col.build_cache();
    }

    return results;
}

ColumnarTable to_columnar(const columnar &table, const Plan &plan) {
    ColumnarTable ret;
    auto &root_node = plan.nodes[plan.root];
    size_t num_cols = root_node.output_attrs.size();

    /* when the table is empty it still requires empty columns */
    if (table.empty()) {
        ret.num_rows = 0;
        for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
            DataType data_type = std::get<1>(root_node.output_attrs[col_idx]);
            ret.columns.emplace_back(data_type);
        }
        return ret;
    }
    ret.num_rows = table[0].row_count();

    /**
     *
     * to materialize values the strategy is similar
     * to the one used in to_columnar the difference
     * being that we need to materialize strings
     * for int the values our stored directly for the process
     * is identical.
     *
     **/
    for (size_t col_idx = 0; col_idx < table.size(); ++col_idx) {
        DataType data_type = std::get<1>(root_node.output_attrs[col_idx]);
        const auto &src_column = table[col_idx];
        ret.columns.emplace_back(data_type);
        auto &column = ret.columns.back();

        switch (data_type) {
        case DataType::INT32: {
            uint16_t num_rows = 0;
            std::vector<int32_t> data;
            std::vector<uint8_t> bitmap;
            data.reserve(2048);
            bitmap.reserve(256);

            auto save_page = [&]() {
                auto *page = column.new_page()->data;
                *reinterpret_cast<uint16_t *>(page) = num_rows;
                *reinterpret_cast<uint16_t *>(page + 2) =
                    static_cast<uint16_t>(data.size());
                memcpy(page + 4, data.data(), data.size() * 4);
                memcpy(page + PAGE_SIZE - bitmap.size(), bitmap.data(),
                       bitmap.size());
                num_rows = 0;
                data.clear();
                bitmap.clear();
            };

            if (src_column.has_direct_access()) {
                /* dense column batchs copies all values then page */
                const size_t total_rows = src_column.row_count();
                std::vector<int32_t> all_data;
                all_data.resize(total_rows);

                size_t row_idx = 0;
                size_t dest_idx = 0;
                while (row_idx < total_rows) {
                    size_t src_offset = row_idx % CAP_PER_PAGE;
                    size_t src_available = CAP_PER_PAGE - src_offset;
                    size_t chunk =
                        std::min(src_available, total_rows - row_idx);

                    /* batch copy int32_t values from contiguous page segment */
                    const value_t *src_ptr = &src_column[row_idx];
                    memcpy(&all_data[dest_idx], src_ptr,
                           chunk * sizeof(int32_t));

                    row_idx += chunk;
                    dest_idx += chunk;
                }

                size_t data_idx = 0;
                while (data_idx < total_rows) {
                    size_t available = (PAGE_SIZE - 4 - 1) / 4;
                    size_t batch_size =
                        std::min(available, total_rows - data_idx);

                    data.assign(all_data.begin() + data_idx,
                                all_data.begin() + data_idx + batch_size);
                    num_rows = batch_size;
                    set_bitmap_range(bitmap, 0, num_rows);
                    save_page();
                    data_idx += batch_size;
                }
            } else {
                /*sparse column checkig for nulls */
                for (size_t row_idx = 0; row_idx < src_column.row_count();
                     ++row_idx) {
                    const value_t *val = src_column.get_by_row(row_idx);

                    if (4 + (val ? data.size() + 1 : data.size()) * 4 +
                            (num_rows / 8 + 1) >
                        PAGE_SIZE) {
                        save_page();
                    }

                    if (val) {
                        set_bitmap(bitmap, num_rows);
                        data.emplace_back(val->value);
                    } else {
                        unset_bitmap(bitmap, num_rows);
                    }
                    ++num_rows;
                }
            }

            if (num_rows != 0)
                save_page();
            break;
        }

        /**
         *
         * to return to columnar
         * we first materialize the string
         * and then use the method shown in the original
         * to_columnar.
         *
         * For the materialization we use all the
         * metadata stored in value_t
         *
         * we first index the table then the column
         * and then the page.
         *
         * To reconstuct small strings we use the offset
         * and look at the offset previous to ours to find
         * our range within the page.
         * (end_of_prev_string -- end_of_current_string).
         *
         * for big strings we detect via the sentinel
         * the  value we copy the contents and
         * then read subsequent pages until they are
         * not 0xfffe
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

            const auto &src_table = plan.inputs[src_column.source_table];
            const auto &src_col = src_table.columns[src_column.source_column];

            auto save_long_string_buffer = [&](const std::string &str) {
                size_t offset = 0;
                bool first_page = true;
                while (offset < str.size()) {
                    auto *page = column.new_page()->data;
                    *reinterpret_cast<uint16_t *>(page) =
                        first_page ? 0xffff : 0xfffe;
                    first_page = false;
                    size_t len = std::min(str.size() - offset, PAGE_SIZE - 4);
                    *reinterpret_cast<uint16_t *>(page + 2) =
                        static_cast<uint16_t>(len);
                    memcpy(page + 4, str.data() + offset, len);
                    offset += len;
                }
            };

            auto copy_long_string_pages = [&](int32_t start_page_idx) {
                int32_t curr_idx = start_page_idx;
                while (true) {
                    auto *src_page_data = src_col.pages[curr_idx]->data;
                    auto *dest_page_data = column.new_page()->data;
                    std::memcpy(dest_page_data, src_page_data, PAGE_SIZE);

                    curr_idx++;
                    if (curr_idx >= src_col.pages.size())
                        break;
                    auto *next_p = src_col.pages[curr_idx]->data;
                    if (*reinterpret_cast<uint16_t *>(next_p) != 0xfffe)
                        break;
                }
            };

            auto save_page = [&]() {
                auto *page = column.new_page()->data;
                *reinterpret_cast<uint16_t *>(page) = num_rows;
                *reinterpret_cast<uint16_t *>(page + 2) =
                    static_cast<uint16_t>(offsets.size());
                memcpy(page + 4, offsets.data(), offsets.size() * 2);
                memcpy(page + 4 + offsets.size() * 2, data.data(), data.size());
                memcpy(page + PAGE_SIZE - bitmap.size(), bitmap.data(),
                       bitmap.size());
                num_rows = 0;
                data.clear();
                offsets.clear();
                bitmap.clear();
            };

            auto add_normal_string = [&](int32_t page_idx, int32_t offset_idx) {
                std::string str =
                    read_normal_string(src_col, page_idx, offset_idx);

                if (str.size() > PAGE_SIZE - 7) {
                    if (num_rows > 0)
                        save_page();
                    save_long_string_buffer(str);
                    return false;
                }

                /* check if fits in current page */
                size_t needed = 4 + (offsets.size() + 1) * 2 +
                                (data.size() + str.size()) + (num_rows / 8 + 1);
                if (needed > PAGE_SIZE)
                    save_page();

                data.insert(data.end(), str.begin(), str.end());
                offsets.emplace_back(data.size());
                /* normal string increments num_rows */
                return true;
            };

            if (src_column.has_direct_access()) {
                /* dense string column */
                for (size_t row_idx = 0; row_idx < src_column.row_count();
                     ++row_idx) {
                    int32_t page_idx, offset_idx;
                    value_t::decode_string(src_column[row_idx].value, page_idx,
                                           offset_idx);

                    if (offset_idx == value_t::LONG_STRING_OFFSET) {
                        if (num_rows > 0)
                            save_page();
                        copy_long_string_pages(page_idx);
                    } else {
                        if (add_normal_string(page_idx, offset_idx)) {
                            set_bitmap(bitmap, num_rows++);
                        }
                    }
                }
            } else {
                /* sparse string column */
                for (size_t row_idx = 0; row_idx < src_column.row_count();
                     ++row_idx) {
                    const value_t *val = src_column.get_by_row(row_idx);

                    if (!val) {
                        if (4 + offsets.size() * 2 + data.size() +
                                (num_rows / 8 + 1) >
                            PAGE_SIZE) {
                            save_page();
                        }
                        unset_bitmap(bitmap, num_rows++);
                    } else {
                        int32_t page_idx, offset_idx;
                        value_t::decode_string(val->value, page_idx,
                                               offset_idx);

                        if (offset_idx == value_t::LONG_STRING_OFFSET) {
                            if (num_rows > 0)
                                save_page();
                            copy_long_string_pages(page_idx);
                        } else {
                            if (add_normal_string(page_idx, offset_idx)) {
                                set_bitmap(bitmap, num_rows++);
                            }
                        }
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
} // namespace mema

#include <plan.h>
#include <cstdint>
#include <common.h>
#include <table.h>

struct alignas(8) value_t {
    int value;
    uint8_t table;
    uint8_t column;
    uint16_t offset;

    value_t() : value(INT32_MIN), table(0), column(0), offset(0) {}
    constexpr value_t(int val, uint8_t table_idx, uint8_t column_idx, uint16_t page_idx) noexcept
        : value(val), table(table_idx), column(column_idx), offset(page_idx) {}
}; 

namespace manolates {
    using row_store = std::vector<std::vector<value_t>>;
    row_store copy_scan(const ColumnarTable& table, uint8_t table_idx,
            const std::vector<std::tuple<size_t, DataType>>& output_attrs);

    ColumnarTable to_columnar(const row_store& table, const Plan& plan);
}

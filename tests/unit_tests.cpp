#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <data_model/plan.h>
#include <data_access/table.h>

void sort(std::vector<std::vector<Data>> &table) {
    std::sort(table.begin(), table.end());
}

/*
 *
 * Default tests
 *
 */
TEST_CASE("Empty join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    ColumnarTable table1, table2;
    table1.columns.emplace_back(DataType::INT32);
    table2.columns.emplace_back(DataType::INT32);
    plan.inputs.emplace_back(std::move(table1));
    plan.inputs.emplace_back(std::move(table2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 0);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
}

TEST_CASE("One line join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data{
        {
            1,
        },
    };
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable table1 = table.to_columnar();
    ColumnarTable table2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(table1));
    plan.inputs.emplace_back(std::move(table2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 1);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {
            1,
            1,
        },
    };
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("Simple join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data{
        {
            1,
        },
        {
            2,
        },
        {
            3,
        },
    };
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable table1 = table.to_columnar();
    ColumnarTable table2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(table1));
    plan.inputs.emplace_back(std::move(table2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 3);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {
            1,
            1,
        },
        {
            2,
            2,
        },
        {
            3,
            3,
        },
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("Empty Result", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{
        {
            1,
        },
        {
            2,
        },
        {
            3,
        },
    };
    std::vector<std::vector<Data>> data2{
        {
            4,
        },
        {
            5,
        },
        {
            6,
        },
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), std::move(types));
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 0);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
}

TEST_CASE("Multiple same keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{
        {
            1,
        },
        {
            1,
        },
        {
            2,
        },
        {
            3,
        },
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), std::move(types));
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 6);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {
            1,
            1,
        },
        {
            1,
            1,
        },
        {
            1,
            1,
        },
        {
            1,
            1,
        },
        {
            2,
            2,
        },
        {
            3,
            3,
        },
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("NULL keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{
        {
            1,
        },
        {
            1,
        },
        {
            std::monostate{},
        },
        {
            2,
        },
        {
            3,
        },
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), std::move(types));
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 6);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {
            1,
            1,
        },
        {
            1,
            1,
        },
        {
            1,
            1,
        },
        {
            1,
            1,
        },
        {
            2,
            2,
        },
        {
            3,
            3,
        },
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("Multiple columns", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{1, DataType::VARCHAR}, {0, DataType::INT32}});
    plan.new_join_node(
        true, 0, 1, 0, 1,
        {{0, DataType::INT32}, {2, DataType::INT32}, {1, DataType::VARCHAR}});
    using namespace std::string_literals;
    std::vector<std::vector<Data>> data1{
        {
            1,
            "xxx"s,
        },
        {
            1,
            "yyy"s,
        },
        {
            std::monostate{},
            "zzz"s,
        },
        {
            2,
            "uuu"s,
        },
        {
            3,
            "vvv"s,
        },
    };
    std::vector<DataType> types{DataType::INT32, DataType::VARCHAR};
    Table table1(std::move(data1), std::move(types));
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 6);
    REQUIRE(result.columns.size() == 3);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    REQUIRE(result.columns[2].type == DataType::VARCHAR);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {1, 1, "xxx"s}, {1, 1, "xxx"s}, {1, 1, "yyy"s},
        {1, 1, "yyy"s}, {2, 2, "uuu"s}, {3, 3, "vvv"s},
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("Build on right", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{1, DataType::VARCHAR}, {0, DataType::INT32}});
    plan.new_join_node(
        false, 0, 1, 0, 1,
        {{0, DataType::INT32}, {2, DataType::INT32}, {1, DataType::VARCHAR}});
    using namespace std::string_literals;
    std::vector<std::vector<Data>> data1{
        {
            1,
            "xxx"s,
        },
        {
            1,
            "yyy"s,
        },
        {
            std::monostate{},
            "zzz"s,
        },
        {
            2,
            "uuu"s,
        },
        {
            3,
            "vvv"s,
        },
    };
    std::vector<DataType> types{DataType::INT32, DataType::VARCHAR};
    Table table1(std::move(data1), std::move(types));
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 6);
    REQUIRE(result.columns.size() == 3);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    REQUIRE(result.columns[2].type == DataType::VARCHAR);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {1, 1, "xxx"s}, {1, 1, "xxx"s}, {1, 1, "yyy"s},
        {1, 1, "yyy"s}, {2, 2, "uuu"s}, {3, 3, "vvv"s},
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("leftdeep 2-level join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_scan_node(2, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    plan.new_join_node(
        false, 3, 2, 0, 0,
        {{0, DataType::INT32}, {1, DataType::INT32}, {2, DataType::INT32}});
    std::vector<std::vector<Data>> data{
        {
            1,
        },
        {
            2,
        },
        {
            3,
        },
    };
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable table1 = table.to_columnar();
    ColumnarTable table2 = table.to_columnar();
    ColumnarTable table3 = table.to_columnar();
    plan.inputs.emplace_back(std::move(table1));
    plan.inputs.emplace_back(std::move(table2));
    plan.inputs.emplace_back(std::move(table3));
    plan.root = 4;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 3);
    REQUIRE(result.columns.size() == 3);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    REQUIRE(result.columns[2].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {
            1,
            1,
            1,
        },
        {
            2,
            2,
            2,
        },
        {
            3,
            3,
            3,
        },
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("3-way join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}, {1, DataType::VARCHAR}});
    plan.new_scan_node(1, {{0, DataType::INT32}, {1, DataType::VARCHAR}});
    plan.new_scan_node(2, {{0, DataType::INT32}, {1, DataType::VARCHAR}});
    plan.new_join_node(false, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::VARCHAR}});
    plan.new_join_node(
        false, 3, 2, 0, 0,
        {{0, DataType::INT32}, {1, DataType::VARCHAR}, {3, DataType::VARCHAR}});
    using namespace std::string_literals;
    std::vector<std::vector<Data>> data1{
        {1, "a"s},
        {2, "b"s},
        {3, "c"s},
    };
    std::vector<std::vector<Data>> data2{
        {1, "x"s},
        {2, "y"s},
    };
    std::vector<std::vector<Data>> data3{
        {1, "u"s},
        {2, "v"s},
        {3, "w"s},
    };
    std::vector<DataType> types{DataType::INT32, DataType::VARCHAR};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    Table table3(std::move(data3), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    ColumnarTable input3 = table3.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.inputs.emplace_back(std::move(input3));
    plan.root = 4;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 2);
    REQUIRE(result.columns.size() == 3);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::VARCHAR);
    REQUIRE(result.columns[2].type == DataType::VARCHAR);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {1, "a"s, "u"s},
        {2, "b"s, "v"s},
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("Large Duplicate keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    // 100 rows with key=1
    std::vector<std::vector<Data>> data1;
    for (int i = 0; i < 100; i++) {
        data1.push_back({1});
    }
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));

    plan.root = 2;

    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);

    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 10000); // 100 * 100
}
TEST_CASE("Sequential keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data;
    for (int i = 1; i <= 1000; i++) {
        data.push_back({i});
    }
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable input1 = table.to_columnar();
    ColumnarTable input2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 1000);
}

TEST_CASE("All NULL keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data{
        {std::monostate{}},
        {std::monostate{}},
        {std::monostate{}},
    };
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable input1 = table.to_columnar();
    ColumnarTable input2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 0); // NULLs don't match
}

TEST_CASE("All keys different", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{{1}, {2}, {3}};
    std::vector<std::vector<Data>> data2{{4}, {5}, {6}};
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 0);
}
TEST_CASE("Negative keys join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{{-1}, {-2}, {-3}};
    std::vector<std::vector<Data>> data2{{-2}, {-3}, {-4}};
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 2); // -2 and -3
}

TEST_CASE("Heavy hash collisions", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    // Choose keys that would hash to the same bucket in a power-of-2 sized
    // table. The Cuckoo table uses a fixed capacity based on the initial size
    // hint, so this is an indirect test.
    std::vector<std::vector<Data>> data1{{0}, {1024}, {2048}, {3072}};
    std::vector<std::vector<Data>> data2{{0}, {1024}, {4096}};
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 2); // 0 and 1024 match. 2048 is in T1 only, 3072
                                   // is in T1 only. 4096 is in T2 only.
}

/*
 *
 * Threaded hash build tests
 * These tests use large datasets (>= 10000 rows) to trigger
 * multi-threaded hash table building
 *
 */
TEST_CASE("Threaded build large sequential keys", "[join][threaded]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    // 15000 rows to trigger threaded build (>= 10000 threshold)
    std::vector<std::vector<Data>> data;
    for (int i = 1; i <= 15000; i++) {
        data.push_back({i});
    }
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable input1 = table.to_columnar();
    ColumnarTable input2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 15000);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    // Verify all keys match correctly
    auto table_data = result_table.table();
    sort(table_data);
    REQUIRE(table_data.size() == 15000);
    for (size_t i = 0; i < 15000; i++) {
        REQUIRE(std::get<int32_t>(table_data[i][0]) == static_cast<int32_t>(i + 1));
        REQUIRE(std::get<int32_t>(table_data[i][1]) == static_cast<int32_t>(i + 1));
    }
}

TEST_CASE("Threaded build large duplicate keys", "[join][threaded]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    // 12000 rows with only 10 distinct keys to stress partitioning
    std::vector<std::vector<Data>> data1;
    for (int i = 0; i < 1200; i++) {
        data1.push_back({(i % 10) + 1});
    }
    std::vector<std::vector<Data>> data2;
    for (int i = 0; i < 1200; i++) {
        data2.push_back({(i % 10) + 1});
    }
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    // Each key appears 1200 times in each table, so 1200 * 1200 = 1,440,000 matches per key
    // 10 keys total = 144,000 matches
    REQUIRE(result.num_rows == 144000);
    REQUIRE(result.columns.size() == 2);
}

TEST_CASE("Threaded build multi-level join", "[join][threaded]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_scan_node(2, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    plan.new_join_node(
        true, 3, 2, 0, 0,
        {{0, DataType::INT32}, {1, DataType::INT32}, {2, DataType::INT32}});
    // First join creates intermediate result with >= 10000 rows to trigger threaded build
    std::vector<std::vector<Data>> data;
    for (int i = 1; i <= 12000; i++) {
        data.push_back({i});
    }
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable table1 = table.to_columnar();
    ColumnarTable table2 = table.to_columnar();
    ColumnarTable table3 = table.to_columnar();
    plan.inputs.emplace_back(std::move(table1));
    plan.inputs.emplace_back(std::move(table2));
    plan.inputs.emplace_back(std::move(table3));
    plan.root = 4;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 12000);
    REQUIRE(result.columns.size() == 3);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    REQUIRE(result.columns[2].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    auto table_data = result_table.table();
    sort(table_data);
    REQUIRE(table_data.size() == 12000);
    for (size_t i = 0; i < 12000; i++) {
        int32_t expected = static_cast<int32_t>(i + 1);
        REQUIRE(std::get<int32_t>(table_data[i][0]) == expected);
        REQUIRE(std::get<int32_t>(table_data[i][1]) == expected);
        REQUIRE(std::get<int32_t>(table_data[i][2]) == expected);
    }
}

TEST_CASE("Threaded build with NULLs", "[join][threaded]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    // 15000 rows with some NULLs mixed in
    std::vector<std::vector<Data>> data1;
    for (int i = 0; i < 15000; i++) {
        if (i % 7 == 0) {
            data1.push_back({std::monostate{}});
        } else {
            data1.push_back({(i % 100) + 1});
        }
    }
    std::vector<std::vector<Data>> data2;
    for (int i = 0; i < 15000; i++) {
        if (i % 11 == 0) {
            data2.push_back({std::monostate{}});
        } else {
            data2.push_back({(i % 100) + 1});
        }
    }
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows > 0); // Should have matches for non-NULL keys
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    // Verify no NULLs in result (NULLs don't match)
    auto result_table = Table::from_columnar(result);
    for (const auto& row : result_table.table()) {
        REQUIRE(!std::holds_alternative<std::monostate>(row[0]));
        REQUIRE(!std::holds_alternative<std::monostate>(row[1]));
    }
}

TEST_CASE("Threaded build sparse distribution", "[join][threaded]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0,
                       {{0, DataType::INT32}, {1, DataType::INT32}});
    // 20000 rows with sparse key distribution to test partitioning
    // Keys range from 1 to 20000, but only every 10th key appears
    std::vector<std::vector<Data>> data1;
    for (int i = 1; i <= 20000; i += 10) {
        data1.push_back({i});
    }
    std::vector<std::vector<Data>> data2;
    for (int i = 1; i <= 20000; i += 10) {
        data2.push_back({i});
    }
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto *context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    // 20000 / 10 = 2000 distinct keys, all should match
    REQUIRE(result.num_rows == 2000);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    auto table_data = result_table.table();
    sort(table_data);
    REQUIRE(table_data.size() == 2000);
    // Verify all keys are multiples of 10
    for (size_t i = 0; i < 2000; i++) {
        int32_t key = std::get<int32_t>(table_data[i][0]);
        REQUIRE(key == std::get<int32_t>(table_data[i][1]));
        REQUIRE(key % 10 == 1); // Keys are 1, 11, 21, ..., 19991
    }
}

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <table.h>
#include <plan.h>
#include <robinhood.h>

void sort(std::vector<std::vector<Data>>& table) {
    std::sort(table.begin(), table.end());
}

// =================================================================================
// New CuckooTable Specific Tests
// =================================================================================
/*
TEST_CASE("CuckooTable Basic Insert and Search", "[cuckoo]") {
    CuckooTable<TableEntity> table(10);
    TableEntity key1{"t1", 1};
    TableEntity key2{"t2", 2};
    
    // Insert first key
    table.insert(key1, 100);
    
    // Search for first key
    auto result1 = table.search(key1);
    REQUIRE(result1.has_value());
    REQUIRE(result1.value()->size() == 1);
    REQUIRE(result1.value()->at(0) == 100);

    // Insert second key
    table.insert(key2, 200);

    // Search for second key
    auto result2 = table.search(key2);
    REQUIRE(result2.has_value());
    REQUIRE(result2.value()->size() == 1);
    REQUIRE(result2.value()->at(0) == 200);

    // Search for non-existent key
    TableEntity key3{"t3", 3};
    auto result3 = table.search(key3);
    REQUIRE_FALSE(result3.has_value());
}

TEST_CASE("CuckooTable Duplicate Key Handling", "[cuckoo]") {
    CuckooTable<TableEntity> table(10);
    TableEntity key1{"t1", 1};
    
    // Insert with first index
    table.insert(key1, 100);
    
    // Insert with second index (should append)
    table.insert(key1, 101);
    
    // Insert with third index (should append)
    table.insert(key1, 102);

    auto result = table.search(key1);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->size() == 3);
    REQUIRE(result.value()->at(0) == 100);
    REQUIRE(result.value()->at(1) == 101);
    REQUIRE(result.value()->at(2) == 102);
}

TEST_CASE("CuckooTable Rehash Implicit Test (Many Inserts)", "[cuckoo]") {
    // Start with a small table, forcing rehashes. Capacity will be max(1, 10/2) = 5 per table.
    CuckooTable<int> table(10); 
    size_t num_inserts = 100;
    
    // Insert many unique keys. If the insert logic or rehash is broken, this will fail.
    for (int i = 0; i < num_inserts; ++i) {
        table.insert(i, i);
    }
    
    // Verify all keys are present
    for (int i = 0; i < num_inserts; ++i) {
        auto result = table.search(i);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->size() == 1);
        REQUIRE(result.value()->at(0) == (size_t)i);
    }
}
*/
// =================================================================================
// Existing Join Tests
// =================================================================================

TEST_CASE("Empty join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    ColumnarTable table1, table2;
    table1.columns.emplace_back(DataType::INT32);
    table2.columns.emplace_back(DataType::INT32);
    plan.inputs.emplace_back(std::move(table1));
    plan.inputs.emplace_back(std::move(table2));
    plan.root = 2;
    auto* context = Contest::build_context();
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
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data{
        {1, },
    };
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable table1 = table.to_columnar();
    ColumnarTable table2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(table1));
    plan.inputs.emplace_back(std::move(table2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 1);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {1, 1,},
    };
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("Simple join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data{
        {1,},
        {2,},
        {3,},
    };
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable table1 = table.to_columnar();
    ColumnarTable table2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(table1));
    plan.inputs.emplace_back(std::move(table2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 3);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {1, 1,},
        {2, 2,},
        {3, 3,},
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("Empty Result", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{
        {1,},
        {2,},
        {3,},
    };
    std::vector<std::vector<Data>> data2{
        {4,},
        {5,},
        {6,},
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), std::move(types));
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
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
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{
        {1,},
        {1,},
        {2,},
        {3,},
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), std::move(types));
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 6);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {1, 1,},
        {1, 1,},
        {1, 1,},
        {1, 1,},
        {2, 2,},
        {3, 3,},
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("NULL keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{
        {1,               },
        {1,               },
        {std::monostate{},},
        {2,               },
        {3,               },
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), std::move(types));
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 6);
    REQUIRE(result.columns.size() == 2);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {1, 1,},
        {1, 1,},
        {1, 1,},
        {1, 1,},
        {2, 2,},
        {3, 3,},
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("Multiple columns", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{1, DataType::VARCHAR}, {0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 1, {{0, DataType::INT32}, {2, DataType::INT32}, {1, DataType::VARCHAR}});
    using namespace std::string_literals;
    std::vector<std::vector<Data>> data1{
        {1               , "xxx"s,},
        {1               , "yyy"s,},
        {std::monostate{}, "zzz"s,},
        {2               , "uuu"s,},
        {3               , "vvv"s,},
    };
    std::vector<DataType> types{DataType::INT32, DataType::VARCHAR};
    Table table1(std::move(data1), std::move(types));
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 6);
    REQUIRE(result.columns.size() == 3);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    REQUIRE(result.columns[2].type == DataType::VARCHAR);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {1, 1, "xxx"s},
        {1, 1, "xxx"s},
        {1, 1, "yyy"s},
        {1, 1, "yyy"s},
        {2, 2, "uuu"s},
        {3, 3, "vvv"s},
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("Build on right", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{1, DataType::VARCHAR}, {0, DataType::INT32}});
    plan.new_join_node(false, 0, 1, 0, 1, {{0, DataType::INT32}, {2, DataType::INT32}, {1, DataType::VARCHAR}});
    using namespace std::string_literals;
    std::vector<std::vector<Data>> data1{
        {1               , "xxx"s,},
        {1               , "yyy"s,},
        {std::monostate{}, "zzz"s,},
        {2               , "uuu"s,},
        {3               , "vvv"s,},
    };
    std::vector<DataType> types{DataType::INT32, DataType::VARCHAR};
    Table table1(std::move(data1), std::move(types));
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 6);
    REQUIRE(result.columns.size() == 3);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    REQUIRE(result.columns[2].type == DataType::VARCHAR);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {1, 1, "xxx"s},
        {1, 1, "xxx"s},
        {1, 1, "yyy"s},
        {1, 1, "yyy"s},
        {2, 2, "uuu"s},
        {3, 3, "vvv"s},
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("leftdeep 2-level join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_scan_node(2, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    plan.new_join_node(false, 3, 2, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}, {2, DataType::INT32}});
    std::vector<std::vector<Data>> data{
        {1,},
        {2,},
        {3,},
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
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 3);
    REQUIRE(result.columns.size() == 3);
    REQUIRE(result.columns[0].type == DataType::INT32);
    REQUIRE(result.columns[1].type == DataType::INT32);
    REQUIRE(result.columns[2].type == DataType::INT32);
    auto result_table = Table::from_columnar(result);
    std::vector<std::vector<Data>> ground_truth{
        {1, 1, 1,},
        {2, 2, 2,},
        {3, 3, 3,},
    };
    sort(result_table.table());
    REQUIRE(result_table.table() == ground_truth);
}

TEST_CASE("3-way join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}, {1, DataType::VARCHAR}});
    plan.new_scan_node(1, {{0, DataType::INT32}, {1, DataType::VARCHAR}});
    plan.new_scan_node(2, {{0, DataType::INT32}, {1, DataType::VARCHAR}});
    plan.new_join_node(false, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::VARCHAR}});
    plan.new_join_node(false, 3, 2, 0, 0, {{0, DataType::INT32}, {1, DataType::VARCHAR}, {3, DataType::VARCHAR}});
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
    auto* context = Contest::build_context();
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
/*
TEST_CASE("Large Duplicate keys","[join]"){
    Plan plan;
    plan.new_scan_node(0,{{0, DataType::INT32}});
    plan.new_scan_node(1,{{0, DataType:: INT32}});
    plan.new_join_node(true, 0, 1, 0 ,0, {{0, DataType::INT32}, {1,DataType::INT32}});
    //100 rows with key=1
    std::vector<std::vector<Data>> data1;
    for(int i = 0; i < 100; i++){
        data1.push_back({1});
    }
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1),types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));

    plan.root = 2;

    auto* context = Contest::build_context();
    auto result = Contest::execute(plan,context);

    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 10000); //100 * 100
}
TEST_CASE("Sequential keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
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
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 1000);
}

TEST_CASE("All NULL keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
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
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 0); // NULLs don't match
}

TEST_CASE("INT64 keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT64}});
    plan.new_scan_node(1, {{0, DataType::INT64}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT64}, {1, DataType::INT64}});
    std::vector<std::vector<Data>> data{
        {(int64_t)1000000000000LL},
        {(int64_t)2000000000000LL},
        {(int64_t)3000000000000LL},
    };
    std::vector<DataType> types{DataType::INT64};
    Table table(std::move(data), std::move(types));
    ColumnarTable input1 = table.to_columnar();
    ColumnarTable input2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 3);
}

TEST_CASE("Negative keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data{
        {-100},
        {-50},
        {0},
        {50},
        {100},
    };
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable input1 = table.to_columnar();
    ColumnarTable input2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 5);
}
TEST_CASE("Sparse keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT64}});
    plan.new_scan_node(1, {{0, DataType::INT64}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT64}, {1, DataType::INT64}});
    std::vector<std::vector<Data>> data{
        {(int64_t)1},
        {(int64_t)1000},
        {(int64_t)1000000},
        {(int64_t)1000000000LL},
        {(int64_t)1000000000000LL},
    };
    std::vector<DataType> types{DataType::INT64};
    Table table(std::move(data), std::move(types));
    ColumnarTable input1 = table.to_columnar();
    ColumnarTable input2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 5);
}
TEST_CASE("All keys different", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{
        {1}, {2}, {3}
    };
    std::vector<std::vector<Data>> data2{
        {4}, {5}, {6}
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 0);
}
TEST_CASE("Negative keys join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{
        {-1}, {-2}, {-3}
    };
    std::vector<std::vector<Data>> data2{
        {-2}, {-3}, {-4}
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 2); // -2 and -3
}
TEST_CASE("Large string keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::VARCHAR}});
    plan.new_scan_node(1, {{0, DataType::VARCHAR}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::VARCHAR}, {1, DataType::VARCHAR}});
    using namespace std::string_literals;
    std::string bigA(1000, 'a');
    std::string bigB(1000, 'b');
    std::string bigC(1000, 'c');
    std::vector<std::vector<Data>> data1{
        {bigA}, {bigB}
    };
    std::vector<std::vector<Data>> data2{
        {bigA}, {bigC}
    };
    std::vector<DataType> types{DataType::VARCHAR};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 1); // Only bigA matches
}
TEST_CASE("Heavy hash collisions", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    // Choose keys that would hash to the same bucket in a power-of-2 sized table.
    // The Cuckoo table uses a fixed capacity based on the initial size hint, so this is an indirect test.
    std::vector<std::vector<Data>> data1{
        {0}, {1024}, {2048}, {3072}
    };
    std::vector<std::vector<Data>> data2{
        {0}, {1024}, {4096}
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 2); // 0 and 1024 match. 2048 is in T1 only, 3072 is in T1 only. 4096 is in T2 only.
}

/*
TEST_CASE("Large Duplicate keys","[join]"){
    Plan plan;
    plan.new_scan_node(0,{{0, DataType::INT32}});
    plan.new_scan_node(1,{{0, DataType:: INT32}});
    plan.new_join_node(true, 0, 1, 0 ,0, {{0, DataType::INT32}, {1,DataType::INT32}});
    //100 rows with key=1
    std::vector<std::vector<Data>> data1;
    for(int i = 0; i < 100; i++){
        data1.push_back({1});
    }
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1),types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table1.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));

    plan.root = 2;

    auto* context = Contest::build_context();
    auto result = Contest::execute(plan,context);

    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 10000); //100 * 100
}
TEST_CASE("Sequential keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
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
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 1000);
}

TEST_CASE("All NULL keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
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
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 0); // NULLs don't match
}

TEST_CASE("INT64 keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT64}});
    plan.new_scan_node(1, {{0, DataType::INT64}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT64}, {1, DataType::INT64}});
    std::vector<std::vector<Data>> data{
        {(int64_t)1000000000000LL},
        {(int64_t)2000000000000LL},
        {(int64_t)3000000000000LL},
    };
    std::vector<DataType> types{DataType::INT64};
    Table table(std::move(data), std::move(types));
    ColumnarTable input1 = table.to_columnar();
    ColumnarTable input2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 3);
}

TEST_CASE("Negative keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data{
        {-100},
        {-50},
        {0},
        {50},
        {100},
    };
    std::vector<DataType> types{DataType::INT32};
    Table table(std::move(data), std::move(types));
    ColumnarTable input1 = table.to_columnar();
    ColumnarTable input2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 5);
}
TEST_CASE("Sparse keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT64}});
    plan.new_scan_node(1, {{0, DataType::INT64}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT64}, {1, DataType::INT64}});
    std::vector<std::vector<Data>> data{
        {(int64_t)1},
        {(int64_t)1000},
        {(int64_t)1000000},
        {(int64_t)1000000000LL},
        {(int64_t)1000000000000LL},
    };
    std::vector<DataType> types{DataType::INT64};
    Table table(std::move(data), std::move(types));
    ColumnarTable input1 = table.to_columnar();
    ColumnarTable input2 = table.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 5);
}
TEST_CASE("All keys different", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{
        {1}, {2}, {3}
    };
    std::vector<std::vector<Data>> data2{
        {4}, {5}, {6}
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 0);
}
TEST_CASE("Negative keys join", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    std::vector<std::vector<Data>> data1{
        {-1}, {-2}, {-3}
    };
    std::vector<std::vector<Data>> data2{
        {-2}, {-3}, {-4}
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 2); // -2 and -3
}
TEST_CASE("Large string keys", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::VARCHAR}});
    plan.new_scan_node(1, {{0, DataType::VARCHAR}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::VARCHAR}, {1, DataType::VARCHAR}});
    using namespace std::string_literals;
    std::string bigA(1000, 'a');
    std::string bigB(1000, 'b');
    std::string bigC(1000, 'c');
    std::vector<std::vector<Data>> data1{
        {bigA}, {bigB}
    };
    std::vector<std::vector<Data>> data2{
        {bigA}, {bigC}
    };
    std::vector<DataType> types{DataType::VARCHAR};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 1); // μόνο το bigA ταιριάζει
}
TEST_CASE("Heavy hash collisions", "[join]") {
    Plan plan;
    plan.new_scan_node(0, {{0, DataType::INT32}});
    plan.new_scan_node(1, {{0, DataType::INT32}});
    plan.new_join_node(true, 0, 1, 0, 0, {{0, DataType::INT32}, {1, DataType::INT32}});
    // Επιλέγουμε keys που δίνουν το ίδιο hash
    std::vector<std::vector<Data>> data1{
        {0}, {1024}, {2048}, {3072}
    };
    std::vector<std::vector<Data>> data2{
        {0}, {1024}, {4096}
    };
    std::vector<DataType> types{DataType::INT32};
    Table table1(std::move(data1), types);
    Table table2(std::move(data2), types);
    ColumnarTable input1 = table1.to_columnar();
    ColumnarTable input2 = table2.to_columnar();
    plan.inputs.emplace_back(std::move(input1));
    plan.inputs.emplace_back(std::move(input2));
    plan.root = 2;
    auto* context = Contest::build_context();
    auto result = Contest::execute(plan, context);
    Contest::destroy_context(context);
    REQUIRE(result.num_rows == 2);
}


TEST_CASE("Hopscotch creation", "[hopscotch]") {
    HopscotchHashTable<int32_t> hash_table(10);
    REQUIRE(hash_table.size() >= 10);
    hash_table.diagnostic();
}

TEST_CASE("Hopscotch basic insertion", "[hopscotch]") {
    HopscotchHashTable<int32_t> hash_table(10);
    for(int i = 0; i < hash_table.size(); i++) {
        hash_table.insert(i, i+10);
    }
}

TEST_CASE("Hopscotch basic find", "[hopscotch]") {
    HopscotchHashTable<int32_t> hash_table(10);

    for(int i = 0; i < 5; i++) {
        hash_table.insert(i, i+10);
    }

    for (int i = 0; i < 5; i++) {
        auto bucket = hash_table.find(i);
        for (auto idx : bucket) {
            REQUIRE(idx == i + 10);
        }
    }
}

TEST_CASE("Hopscotch test bucket functionality", "[hopscotch]") {
    HopscotchHashTable<int32_t> hash_table(10);
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 100; j++ ) {
            hash_table.insert(i, j);
        }
    }
    for (int i = 0; i < 10; i++) {
        auto bucket =  hash_table.find(i);
        int j = 0;
        for (auto idx: bucket)
            REQUIRE(idx == (j++));
    }
}
*/

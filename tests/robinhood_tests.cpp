#include <robinhood.h>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <table.h>
#include <plan.h>

TEST_CASE("RobinHood creating", "[robinhood]")
{
    RobinHoodTable<int32_t> hash_table(10);
    REQUIRE((hash_table.capacity()) >= 10);
    // hash_table.diagnostic();
}

TEST_CASE("RobinHood sequential numbers", "[robinhood]")
{
    RobinHoodTable<int32_t> hash_table(256); //
    for (size_t i = 0; i < 150; ++i)
    {
        hash_table.insert(i, i);
    }
    for (size_t i = 0; i < 150; ++i)
    {
        auto result = hash_table.find(i);
        REQUIRE(result!=nullptr);
        REQUIRE(result->size() == 1);
        REQUIRE((*result)[0] == i);
    }
    // hash_table.diagnostic();
}
TEST_CASE("RobinHood functionality", "[robinhood]")
{ // check if indices are saved for each key and in the right order.
    RobinHoodTable<int32_t> hash_table(128);
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 100; j++)
        {
            hash_table.insert(i, j);
        }
    }
    for (int i = 0; i < 10; i++)
    {
        auto bucket = hash_table.find(i);
        int j = 0;
        REQUIRE(bucket != nullptr);
        for (auto idx : *bucket)
            REQUIRE(idx == (j++));
    }
    // hash_table.diagnostic();
}
TEST_CASE("RobinHood PSL functionality", "[robinhood]")
{
    RobinHoodTable<uint32_t> hash_table(8); // use small table for more collisions
    hash_table.insert(1, 1);
    hash_table.insert(9, 9);
    hash_table.insert(5, 5);
    hash_table.insert(2, 2);

    hash_table.insert(17, 17);
    hash_table.insert(12, 12);

    // hash_table.diagnostic();
}
TEST_CASE("RobinHood duplicate keys","[robinhood]"){
    RobinHoodTable<int32_t> hash_table(16);
    for(int i = 0; i < 5; ++i){
        for(int j = 0; j <3; ++j){
            hash_table.insert(i,i*100+j); //insert same i different idx
        }
    }
    for(int i = 0; i < 5; ++i){
        auto result = hash_table.find(i);
        REQUIRE(result!=nullptr);
        REQUIRE(result->size()==3);
        for(int j = 0; j < 3; ++j)  REQUIRE((*result)[j] == i* 100+ j);
    }
    // hash_table.diagnostic();
}

TEST_CASE("RobinHood Missing key", "[robinhood]"){
    RobinHoodTable<int32_t> hash_table(8);
    hash_table.insert(1,10);
    hash_table.insert(2,20);

    auto result = hash_table.find(99);
    // hash_table.diagnostic();
}
TEST_CASE("RobinHoodTable insert/find micro-benchmark", "[robinhood]") {
    constexpr size_t N = 100000;
    RobinHoodTable<int> table(1 << 18);

    // Insert benchmark
    auto start_insert = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        table.insert(i, i);
    }
    auto end_insert = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> insert_time = end_insert - start_insert;
    std::cout << "Insert time for " << N << " elements: " << insert_time.count() << " seconds\n";

    // find benchmark
    auto start_find = std::chrono::high_resolution_clock::now();
    size_t found = 0;
    for (size_t i = 0; i < N; ++i) {
        auto result = table.find(i);
        if (result!=nullptr) ++found;
    }
    auto end_find = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> find_time = end_find - start_find;
    std::cout << "find time for " << N << " elements: " << find_time.count() << " seconds\n";
    std::cout << "Found: " << found << "\n";

    REQUIRE(found == N);
}


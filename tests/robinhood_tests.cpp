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

TEST_CASE("RobinHood sequintial numbers", "[robinhood]")
{
    RobinHoodTable<int32_t> hash_table(256); // use only power of 2 cause of hash_func
    for (size_t i = 0; i < 150; ++i)
    {
        hash_table.insert(i, i);
    }
    for (size_t i = 0; i < 150; ++i)
    {
        auto result = hash_table.search(i);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->size() == 1);
        REQUIRE((*result.value())[0] == i);
    }
    // hash_table.diagnostic();
}
TEST_CASE("RobinHOod functionality", "[robinhood]")
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
        auto bucket = hash_table.search(i);
        int j = 0;
        REQUIRE(bucket.has_value());
        for (auto idx : (*bucket.value()))
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
        auto result = hash_table.search(i);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->size()==3);
        for(int j = 0; j < 3; ++j)  REQUIRE((*result.value())[j] == i* 100+ j);
    }
    // hash_table.diagnostic();
}

TEST_CASE("RobinHood Missing key", "[robinhood]"){
    RobinHoodTable<int32_t> hash_table(8);
    hash_table.insert(1,10);
    hash_table.insert(2,20);

    auto result = hash_table.search(99);
    // hash_table.diagnostic();
}


#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <table.h>
#include <plan.h>
#include <hopscotch.h>

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

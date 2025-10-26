#include <catch2/catch_test_macros.hpp>

#include <cuckoo.h>
#include <table_entity.h>
#include <string>
#include <fmt/core.h>

TEST_CASE("CuckooTable Basic Insert and Search", "[cuckoo]") {
    fmt::println("--- Testing Basic Insert and Search ---");
    CuckooTable<TableEntity> table(10);
    TableEntity key1{"t1", 1};
    TableEntity key2{"t2", 2};
    
    // Insert first key
    fmt::println("Inserting key1: {} with index 100", key1);
    table.insert(key1, 100);
    
    // Search for first key
    auto result1 = table.search(key1);
    REQUIRE(result1.has_value());
    REQUIRE(result1.value()->size() == 1);
    REQUIRE(result1.value()->at(0) == 100);
    fmt::println("Search for key1 successful. Found index: {}", result1.value()->at(0));

    // Insert second key
    fmt::println("Inserting key2: {} with index 200", key2);
    table.insert(key2, 200);

    // Search for second key
    auto result2 = table.search(key2);
    REQUIRE(result2.has_value());
    REQUIRE(result2.value()->size() == 1);
    REQUIRE(result2.value()->at(0) == 200);
    fmt::println("Search for key2 successful. Found index: {}", result2.value()->at(0));

    // Search for non-existent key
    TableEntity key3{"t3", 3};
    fmt::println("Searching for non-existent key3: {}", key3);
    auto result3 = table.search(key3);
    REQUIRE_FALSE(result3.has_value());
    fmt::println("Search for key3 failed as expected.");
    fmt::println("----------------------------------------");
}

TEST_CASE("CuckooTable Duplicate Key Handling", "[cuckoo]") {
    fmt::println("--- Testing Duplicate Key Handling ---");
    CuckooTable<TableEntity> table(10);
    TableEntity key1{"t1", 1};
    
    // Insert with multiple indices
    fmt::println("Inserting key1: {} with indices 100, 101, 102", key1);
    table.insert(key1, 100);
    table.insert(key1, 101);
    table.insert(key1, 102);

    auto result = table.search(key1);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->size() == 3);
    REQUIRE(result.value()->at(0) == 100);
    REQUIRE(result.value()->at(1) == 101);
    REQUIRE(result.value()->at(2) == 102);
    fmt::println("Search for key1 successful. Found indices: {}, {}, {}", 
        result.value()->at(0), result.value()->at(1), result.value()->at(2));
    fmt::println("----------------------------------------");
}

TEST_CASE("CuckooTable Rehash Implicit Test (Many Inserts)", "[cuckoo]") {
    fmt::println("--- Testing Implicit Rehash (100 Inserts) ---");
    CuckooTable<int> table(10); 
    size_t num_inserts = 100;
    
    fmt::println("Inserting {} unique integer keys to force multiple rehashes...", num_inserts);
    for (int i = 0; i < num_inserts; ++i) {
        table.insert(i, i);
    }
    
    // Verify all keys are present
    fmt::println("Verifying all keys...");
    for (int i = 0; i < num_inserts; ++i) {
        auto result = table.search(i);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->size() == 1);
        REQUIRE(result.value()->at(0) == (size_t)i);
    }
    fmt::println("All {} keys verified successfully.", num_inserts);
    fmt::println("----------------------------------------");
}

// Edge case: Test with int64_t keys
TEST_CASE("CuckooTable int64_t Keys", "[cuckoo]") {
    fmt::println("--- Testing int64_t Keys ---");
    CuckooTable<int64_t> table(10);
    int64_t key1 = 123456789012345LL;
    int64_t key2 = 987654321098765LL;
    
    fmt::println("Inserting key1: {} with index 1", key1);
    table.insert(key1, 1);
    fmt::println("Inserting key2: {} with index 2", key2);
    table.insert(key2, 2);
    
    auto result1 = table.search(key1);
    REQUIRE(result1.has_value());
    REQUIRE(result1.value()->size() == 1);
    fmt::println("Search for key1 successful. Found index: {}", result1.value()->at(0));
    
    // Duplicate index for key1
    fmt::println("Inserting duplicate index 3 for key1: {}", key1);
    table.insert(key1, 3);
    auto result1_dup = table.search(key1);
    REQUIRE(result1_dup.has_value());
    REQUIRE(result1_dup.value()->size() == 2);
    REQUIRE(result1_dup.value()->at(1) == 3);
    fmt::println("Key1 now has indices: {}, {}", result1_dup.value()->at(0), result1_dup.value()->at(1));
    fmt::println("----------------------------------------");
}

// Edge case: Test with std::string keys including longer strings
TEST_CASE("CuckooTable std::string Keys", "[cuckoo]") {
    fmt::println("--- Testing std::string Keys ---");
    CuckooTable<std::string> table(10);
    std::string key1 = "hello";
    std::string key2 = "world";
    
    fmt::println("Inserting key1: '{}' with index 10", key1);
    table.insert(key1, 10);
    fmt::println("Inserting key2: '{}' with index 20", key2);
    table.insert(key2, 20);
    
    auto result1 = table.search(key1);
    REQUIRE(result1.has_value());
    REQUIRE(result1.value()->size() == 1);
    fmt::println("Search for key1 successful. Found index: {}", result1.value()->at(0));

    // Non-existent search
    fmt::println("Searching for non-existent key 'test'");
    auto result_non_existent = table.search("test");
    REQUIRE_FALSE(result_non_existent.has_value());
    fmt::println("Search for 'test' failed as expected.");
    
    // Test a longer string and duplicate indices
    std::string key3 = "a very long key to ensure hashing covers the whole string";
    fmt::println("Inserting long key: '{}' with indices 30, 31", key3);
    table.insert(key3, 30);
    table.insert(key3, 31);

    auto result3 = table.search(key3);
    REQUIRE(result3.has_value());
    REQUIRE(result3.value()->size() == 2);
    REQUIRE(result3.value()->at(0) == 30);
    REQUIRE(result3.value()->at(1) == 31);
    fmt::println("Search for long key successful. Found indices: {}, {}", result3.value()->at(0), result3.value()->at(1));
    fmt::println("----------------------------------------");
}

// Edge case: Test with double keys
TEST_CASE("CuckooTable double Keys", "[cuckoo]") {
    fmt::println("--- Testing double Keys ---");
    CuckooTable<double> table(10);
    double key1 = 3.14159;
    double key2 = 2.71828;
    
    fmt::println("Inserting key1: {} with index 10", key1);
    table.insert(key1, 10);
    fmt::println("Inserting key2: {} with index 20", key2);
    table.insert(key2, 20);
    
    auto result1 = table.search(key1);
    REQUIRE(result1.has_value());
    REQUIRE(result1.value()->size() == 1);
    
    // Duplicate index for key1
    fmt::println("Inserting duplicate index 11 for key1: {}", key1);
    table.insert(key1, 11);
    auto result1_dup = table.search(key1);
    REQUIRE(result1_dup.has_value());
    REQUIRE(result1_dup.value()->size() == 2);
    fmt::println("Key1 now has indices: {}, {}", result1_dup.value()->at(0), result1_dup.value()->at(1));

    // Search for a close but different value (floating point comparison edge case)
    double different_key = 3.1416;
    fmt::println("Searching for close but different key: {}", different_key);
    REQUIRE_FALSE(table.search(different_key).has_value());
    fmt::println("Search for {} failed as expected.", different_key);
    fmt::println("----------------------------------------");
}

// Edge case: Test search on empty table and minimum capacity rehash
TEST_CASE("CuckooTable Edge Cases", "[cuckoo]") {
    fmt::println("--- Testing Edge Cases (Empty/Min Capacity) ---");
    // Test searching on an empty table
    CuckooTable<int> empty_table(10);
    fmt::println("Searching empty table for key 5.");
    REQUIRE_FALSE(empty_table.search(5).has_value());
    fmt::println("Search failed as expected.");

    // Test minimum capacity (capacity is initially min(1, s/2)=1) and multiple rehashes
    CuckooTable<int> min_capacity_table(1); 
    size_t num_min_inserts = 20;
    fmt::println("Inserting {} elements into a min-capacity table (initial capacity 1 per table) to force rehashes.", num_min_inserts);
    for (int i = 0; i < num_min_inserts; ++i) {
        min_capacity_table.insert(i, i * 10);
    }
    
    // Verify results after multiple rehashes
    fmt::println("Verifying all keys in min-capacity table...");
    for (int i = 0; i < num_min_inserts; ++i) {
        auto result = min_capacity_table.search(i);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->size() == 1);
        REQUIRE(result.value()->at(0) == (size_t)i * 10);
    }
    fmt::println("All {} keys verified successfully.", num_min_inserts);
    fmt::println("----------------------------------------");
}

// Test with heavy hash collisions to stress the kick/rehash mechanism
TEST_CASE("CuckooTable Heavy Collisions", "[cuckoo]") {
    fmt::println("--- Testing Heavy Collisions ---");
    // With a small capacity, these keys are likely to cause collisions.
    CuckooTable<int> table(4); 
    
    // Keys that are multiples of the capacity are good candidates for collisions.
    int keys[] = {0, 4, 8, 12, 16, 20, 24, 28};
    size_t num_keys = sizeof(keys)/sizeof(int);

    fmt::println("Inserting {} keys that are likely to collide...", num_keys);
    for (size_t i = 0; i < num_keys; ++i) {
        table.insert(keys[i], i);
    }

    fmt::println("Verifying all keys after insertions and potential rehashes...");
    for (size_t i = 0; i < num_keys; ++i) {
        auto result = table.search(keys[i]);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->size() == 1);
        REQUIRE(result.value()->at(0) == i);
    }
    fmt::println("All {} colliding keys verified successfully.", num_keys);
    fmt::println("----------------------------------------");
}

// Test inserting a large number of items with many duplicate keys
TEST_CASE("CuckooTable High Duplicate Rate", "[cuckoo]") {
    fmt::println("--- Testing High Duplicate Rate ---");
    CuckooTable<int> table(50);
    int num_unique_keys = 10;
    int duplicates_per_key = 20;

    fmt::println("Inserting {} unique keys, each with {} duplicates...", num_unique_keys, duplicates_per_key);
    for (int i = 0; i < num_unique_keys; ++i) {
        for (int j = 0; j < duplicates_per_key; ++j) {
            table.insert(i, (size_t)(i * duplicates_per_key + j));
        }
    }

    fmt::println("Verifying all keys and their indices...");
    for (int i = 0; i < num_unique_keys; ++i) {
        auto result = table.search(i);
        REQUIRE(result.has_value());
        REQUIRE(result.value()->size() == (size_t)duplicates_per_key);
        for (int j = 0; j < duplicates_per_key; ++j) {
            REQUIRE(result.value()->at(j) == (size_t)(i * duplicates_per_key + j));
        }
    }
    fmt::println("All keys with high duplication verified successfully.");
    fmt::println("----------------------------------------");
}

// Test with a custom struct that requires a std::hash specialization
struct CustomKey {
    int id;
    std::string name;

    bool operator==(const CustomKey& other) const {
        return id == other.id && name == other.name;
    }
};

namespace std {
    template <>
    struct hash<CustomKey> {
        size_t operator()(const CustomKey& k) const {
            return hash<int>()(k.id) ^ (hash<string>()(k.name) << 1);
        }
    };
}

TEST_CASE("CuckooTable Custom Key Struct", "[cuckoo]") {
    fmt::println("--- Testing Custom Key Struct ---");
    CuckooTable<CustomKey> table(10);
    CustomKey key1 = {1, "one"};
    CustomKey key2 = {2, "two"};

    table.insert(key1, 101);
    table.insert(key2, 202);

    auto result1 = table.search(key1);
    REQUIRE(result1.has_value());
    REQUIRE(result1.value()->at(0) == 101);

    auto result2 = table.search(key2);
    REQUIRE(result2.has_value());
    REQUIRE(result2.value()->at(0) == 202);

    // Test duplicate insert
    table.insert(key1, 1010);
    auto result1_dup = table.search(key1);
    REQUIRE(result1_dup.value()->size() == 2);
    REQUIRE(result1_dup.value()->at(1) == 1010);
    fmt::println("Custom key tests passed.");
    fmt::println("----------------------------------------");
}

TEST_CASE("CuckooTable Insert After Search Miss", "[cuckoo]") {
    fmt::println("--- Testing Insert After Search Miss ---");
    CuckooTable<int> table(10);
    REQUIRE_FALSE(table.search(42).has_value());
    table.insert(42, 100);
    auto result = table.search(42);
    REQUIRE(result.has_value());
    REQUIRE(result.value()->at(0) == 100);
    fmt::println("Insert after search miss verified successfully.");
    fmt::println("----------------------------------------");
}

TEST_CASE("CuckooTable Zero-Sized Initialization", "[cuckoo]") {
    fmt::println("--- Testing Zero-Sized Initialization ---");
    CuckooTable<int> table(0); // Should default to capacity of 1
    table.insert(1, 10);
    table.insert(2, 20); // This will force a rehash
    auto result1 = table.search(1);
    auto result2 = table.search(2);
    REQUIRE(result1.has_value());
    REQUIRE(result2.has_value());
    fmt::println("Zero-sized initialization and subsequent rehash verified.");
    fmt::println("----------------------------------------");
}
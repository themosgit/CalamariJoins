#pragma once
#include <hashtable_structs.h>
#include <vector>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

template<typename Key>
struct CuckooBucket {
    Key key;
    uint32_t first_segment;
    uint32_t last_segment;
    uint16_t count;
    bool occupied;

    CuckooBucket() 
        : key(), first_segment(UINT32_MAX), last_segment(0), 
          count(0), occupied(false) {}
};

template<typename Key>
class CuckooTable {
private:
    // Two hash tables of the same capacity
    std::vector<CuckooBucket<Key>> table1;
    std::vector<CuckooBucket<Key>> table2;
    
    // Shared storage for all buckets
    std::vector<uint32_t> value_store;
    std::vector<Segment> segments;
    
    size_t capacity;
    std::hash<Key> key_hasher;

    // Maximum number of displacements before forcing a rehash
    static constexpr size_t MAX_KICKS = 500; 
    
    // Hash function 1
    size_t h1(const Key& key) const {
        return key_hasher(key) % capacity;
    }

    // Hash function 2 (simple bit rotation of h1 result)
    size_t h2(const Key& key) const {
        size_t h = key_hasher(key);
        // Rotates the hash value left by 1 bit, then modulates
        return ((h << 1) | (h >> (sizeof(size_t) * 8 - 1))) % capacity;
    }

    // Tries to insert the bucket by kicking other elements. 
    // Returns true on success, false if MAX_KICKS exceeded.
    // If false, the bucket parameter contains the entry that couldn't be placed.
    bool insert_internal(CuckooBucket<Key>& bucket) {
        for (size_t current_kicks = 0; current_kicks < MAX_KICKS; ++current_kicks) {
            
            // Try h1 slot in table1
            size_t idx1 = h1(bucket.key);
            if (!table1[idx1].occupied) {
                table1[idx1] = bucket;
                return true; // Success
            }
            
            // Kick the existing bucket from table1 to table2
            std::swap(bucket, table1[idx1]);
            
            // Try h2 slot in table2 for the kicked bucket
            size_t idx2 = h2(bucket.key);
            if (!table2[idx2].occupied) {
                table2[idx2] = bucket;
                return true; // Success
            }
            
            // Kick the existing bucket from table2 to table1 (as its next step)
            std::swap(bucket, table2[idx2]);
        }
        
        // Failed to insert after MAX_KICKS attempts
        return false;
    }
    
    // Doubles the capacity and rebuilds the tables
    void rehash() {
        capacity *= 2;
        
        std::vector<CuckooBucket<Key>> old_table1 = std::move(table1);
        std::vector<CuckooBucket<Key>> old_table2 = std::move(table2);
        
        table1.assign(capacity, CuckooBucket<Key>());
        table2.assign(capacity, CuckooBucket<Key>());

        // Reinsert all old buckets
        for (auto& bucket : old_table1) {
            if (bucket.occupied) {
                if (!insert_internal(bucket)) {
                    throw std::runtime_error("Cuckoo rehash failed: rehash cycle detected (T1)");
                }
            }
        }
        for (auto& bucket : old_table2) {
            if (bucket.occupied) {
                if (!insert_internal(bucket)) {
                    throw std::runtime_error("Cuckoo rehash failed: rehash cycle detected (T2)");
                }
            }
        }
    }

public:
    // Cuckoo hashing is most efficient with a low load factor. 
    // Capacity is set to s/2 for each table, giving 's' total slots.
    CuckooTable(size_t s) {
        // capacity is the size of one table.
        capacity = std::max((size_t)1, s / 2); 
        table1.assign(capacity, CuckooBucket<Key>());
        table2.assign(capacity, CuckooBucket<Key>());
    }
    
    // Inserts a key-index pair. If the key exists, the index is appended.
    void insert(const Key& key, uint32_t item) {
        // 1. Check if key already exists in either table
        size_t idx1 = h1(key);
        if (table1[idx1].occupied && table1[idx1].key == key) {
            insert_duplicate(table1[idx1], item, value_store, segments);
            return;
        }

        size_t idx2 = h2(key);
        if (table2[idx2].occupied && table2[idx2].key == key) {
            insert_duplicate(table2[idx2], item, value_store, segments);
            return;
        }

        // 2. Prepare new bucket with inline value optimization
        CuckooBucket<Key> new_bucket;
        new_bucket.key = key;
        new_bucket.occupied = true;
        new_bucket.count = 1;
        new_bucket.first_segment = UINT32_MAX;
        new_bucket.last_segment = item;  // Store inline for single value

        // 3. Insert/Kick, rehash if cycle is detected
        while (!insert_internal(new_bucket)) {
            // Insertion failed, rehash and try again
            rehash(); 
        }
    }

    // Searches for a key. Returns a ValueSpan if found, otherwise returns empty ValueSpan.
    ValueSpan<Key> find(const Key& key) const {
        // Check h1 slot in table1
        size_t idx1 = h1(key);
        if (table1[idx1].occupied && table1[idx1].key == key) {
            const auto& bucket = table1[idx1];
            ValueSpan<Key> span;
            
            // Handle inline value optimization
            if (bucket.count == 1) {
                span.value_store = nullptr;  // Signal inline value
                span.segments = nullptr;
                span.first_segment = bucket.last_segment;  // The inline value
            } else {
                span.value_store = &value_store;
                span.segments = &segments;
                span.first_segment = bucket.first_segment;
            }
            span.total_count = bucket.count;
            
            return span;
        }

        // Check h2 slot in table2
        size_t idx2 = h2(key);
        if (table2[idx2].occupied && table2[idx2].key == key) {
            const auto& bucket = table2[idx2];
            ValueSpan<Key> span;
            
            // Handle inline value optimization
            if (bucket.count == 1) {
                span.value_store = nullptr;  // Signal inline value
                span.segments = nullptr;
                span.first_segment = bucket.last_segment;  // The inline value
            } else {
                span.value_store = &value_store;
                span.segments = &segments;
                span.first_segment = bucket.first_segment;
            }
            span.total_count = bucket.count;
            
            return span;
        }

        // Return empty span
        ValueSpan<Key> empty_span;
        empty_span.value_store = nullptr;
        empty_span.segments = nullptr;
        empty_span.first_segment = UINT32_MAX;
        empty_span.total_count = 0;
        return empty_span;
    }

    // Legacy search method for backwards compatibility
    ValueSpan<Key> search(const Key& key) const {
        return find(key);
    }
};

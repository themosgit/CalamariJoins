#pragma once
#include <vector>
#include <optional>
#include <cstddef>
#include <functional>
#include <algorithm>
#include <stdexcept>

template<typename Key>
struct CuckooEntry {
    Key key;
    std::vector<size_t> indices;
};

template<typename Key>
class CuckooTable {
private:
    // Two hash tables of the same capacity
    std::vector<std::optional<CuckooEntry<Key>>> table1;
    std::vector<std::optional<CuckooEntry<Key>>> table2;
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

    // Tries to insert the entry by kicking other elements. 
    // Returns the entry that failed to be placed on a cycle/kick limit, or std::nullopt on success.
    std::optional<CuckooEntry<Key>> insert_internal(CuckooEntry<Key> entry) {
        for (size_t current_kicks = 0; current_kicks < MAX_KICKS; ++current_kicks) {
            
            // Try h1 slot in table1
            size_t idx1 = h1(entry.key);
            if (!table1[idx1].has_value()) {
                table1[idx1] = std::move(entry);
                return std::nullopt; // Success
            }
            
            // Kick the existing entry from table1 to table2
            std::swap(entry, *table1[idx1]);
            
            // Try h2 slot in table2 for the kicked entry
            size_t idx2 = h2(entry.key);
            if (!table2[idx2].has_value()) {
                table2[idx2] = std::move(entry);
                return std::nullopt; // Success
            }
            
            // Kick the existing entry from table2 to table1 (as its next step)
            std::swap(entry, *table2[idx2]);
        }
        
        // Failed to insert after MAX_KICKS attempts, return the entry that couldn't be placed.
        return std::move(entry);
    }
    
    // Doubles the capacity and rebuilds the tables
    void rehash() {
        capacity *= 2;
        
        std::vector<std::optional<CuckooEntry<Key>>> old_table1 = std::move(table1);
        std::vector<std::optional<CuckooEntry<Key>>> old_table2 = std::move(table2);
        
        table1.assign(capacity, std::nullopt);
        table2.assign(capacity, std::nullopt);

        // Reinsert all old entries
        for (auto& optional_entry : old_table1) {
            if (optional_entry.has_value()) {
                if (insert_internal(std::move(*optional_entry)).has_value()) {
                    throw std::runtime_error("Cuckoo rehash failed: rehash cycle detected (T1)");
                }
            }
        }
        for (auto& optional_entry : old_table2) {
            if (optional_entry.has_value()) {
                if (insert_internal(std::move(*optional_entry)).has_value()) {
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
        table1.assign(capacity, std::nullopt);
        table2.assign(capacity, std::nullopt);
    }
    
    // Inserts a key-index pair. If the key exists, the index is appended.
    void insert(const Key& key, size_t idx) {
        // 1. Check if key already exists in either table
        if (auto existing = search(key)) {
            existing.value()->push_back(idx);
            return;
        }

        // 2. Prepare new entry
        CuckooEntry<Key> new_entry = {key, {idx}};

        // 3. Insert/Kick, rehash if cycle is detected
        while (true) {
            if (auto failed_entry_opt = insert_internal(std::move(new_entry))) {
                // Insertion failed, get the failed entry, rehash, and try again.
                new_entry = std::move(failed_entry_opt.value());
                rehash(); 
            } else {
                // Success
                break;
            }
        }
    }

    // Searches for a key. Returns a pointer to the vector of indices if found.
    std::optional<std::vector<size_t>*> search(const Key& key) {
        // Check h1 slot in table1
        size_t idx1 = h1(key);
        if (table1[idx1].has_value() && table1[idx1]->key == key) {
            return &table1[idx1]->indices;
        }

        // Check h2 slot in table2
        size_t idx2 = h2(key);
        if (table2[idx2].has_value() && table2[idx2]->key == key) {
            return &table2[idx2]->indices;
        }

        return std::nullopt;
    }
};

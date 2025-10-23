#ifndef HOPSCOTCH_HASH_TABLE_HPP
#define HOPSCOTCH_HASH_TABLE_HPP

#include <cstdlib>
#include <type_traits>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <bitset>

template <typename Key, typename Hash = std::hash<Key>>
class HopscotchHashTable {
private:
    /* neighborhood size - single cacheline */
    static constexpr size_t H = 64;
    /*neighborhood mask*/
    static constexpr uint64_t EMPTY_MASK = 0;
    
    struct Bucket {
        /* key corresponding to bucket */
        Key key;
        /* vector containing same key idx*/
        std::vector<size_t> indices;
        /* bit mask of the bucket*/
        uint64_t hop_info;
        bool occupied;
        
        Bucket() : hop_info(EMPTY_MASK), occupied(false) {
        }
    };

    /*optimized hash function for int32_t keys*/
    uint32_t murmur3_fmix32(uint32_t h) const {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }

    /*  ultra complicated hash expanding to 64 bit */
    size_t hash_int32(int32_t key) const {
        uint32_t h = static_cast<uint32_t>(key);
        
        h = murmur3_fmix32(h);
        
        uint64_t h64 = h;
        h64 = (h64 ^ 0xdeadbeef) + (h64 << 5);
        h64 ^= h64 >> 23;
        h64 *= 0x2127599bf4325c37ULL;
        h64 ^= h64 >> 47;
        
        h64 *= 0xbf58476d1ce4e5b9ULL;
        h64 ^= h64 >> 27;
        h64 *= 0x94d049bb133111ebULL;
        h64 ^= h64 >> 31;
        
        return h64;
    }
    
    size_t hash_key(const Key& key) const {
        if constexpr (std::is_same_v<Key, int32_t>) {
            return hash_int32(key);
       } else { 
            size_t h = hasher(key);
        }
    }
    
    std::vector<Bucket> table;
    size_t capacity;
    size_t num_items;
    Hash hasher;
    
    /*find free bucket within max distance from ideal position*/
    size_t find_free_bucket(size_t start_idx) {
        for (size_t i = 0; i < capacity; ++i) {
            size_t idx = (start_idx + i) % capacity;
            if (!table[idx].occupied) {
                return idx;
            }
        }
        return capacity; /*No free bucket found*/
    }
    
    /* Move items to get a free bucket within H positions*/
    bool relocate(size_t &free_idx, size_t target_idx) {
        const size_t MAX_DEPTH = 128;
        
        for (size_t depth = 0; depth < MAX_DEPTH; ++depth) {
            size_t dist = (free_idx + capacity - target_idx) % capacity;
            if (dist < H) {
                return true;
            }
            
            bool found = false;
            
            for (size_t offset = 1; offset < H && !found; ++offset) {
                size_t check_idx = (free_idx + capacity - offset) % capacity;
                
                if (!table[check_idx].occupied) continue;
                
                uint64_t hop = table[check_idx].hop_info;
                
                for (size_t j = H - 1; j > 0 && !found; --j) {
                    if (hop & (1ULL << j)) {
                        size_t item_idx = (check_idx + j) % capacity;
                        size_t new_dist = (free_idx + capacity - check_idx) % capacity;
                        
                        if (new_dist < H && item_idx != free_idx) {
                            table[free_idx] = std::move(table[item_idx]);
                            table[item_idx].occupied = false;
                            table[item_idx].hop_info = EMPTY_MASK;
                            
                            table[check_idx].hop_info &= ~(1ULL << j);
                            table[check_idx].hop_info |= (1ULL << new_dist);
                            
                            free_idx = item_idx;
                            found = true;
                        }
                    }
                }
            }
            
            if (!found) {
                return false;
            }
        }
        
        return false;
    }
    
public:
    explicit HopscotchHashTable(size_t size, const Hash& hash = Hash()) 
        : capacity(0), hasher(hash) {
        size_t PRIME_SIZES[] = {
            53ul, 97ul, 193ul, 389ul, 769ul, 1543ul, 3079ul, 6151ul,
            12289ul, 24593ul, 49157ul, 98317ul, 196613ul, 393241ul,
            786433ul, 1572869ul, 3145739ul, 6291469ul, 12582917ul,
            25165843ul, 50331653ul, 100663319ul, 201326611ul,
            402653189ul, 805306457ul, 1610612741ul, 3221225473ul,
            6442450939ul, 12884901893ul, 25769803799ul,
            51539607551ul, 103079215111ul, 206158430209ul,
            412316860441ul, 824633720831ul, 1649267441651ul,
        };
        
        for (size_t candidate_size : PRIME_SIZES) {
            if (candidate_size > size) {
                capacity = candidate_size;
                break;
            }
        }
        
        if (capacity == 0) {
            throw std::invalid_argument("Requested size exceeds maximum table capacity");
        }

        table.resize(capacity);
    }
    
    void insert(const Key& key, size_t idx) {
        size_t hash_val = hash_key(key);
        size_t base_idx = hash_val % capacity;
        
        /* check if key has already been inserted*/ 
        uint64_t hop = table[base_idx].hop_info;
        for (size_t i = 0; i < H; ++i) {
            /* check bitmask*/
            if (hop & (1ULL << i)) {
                size_t check_idx = (base_idx + i) % capacity;
                if (table[check_idx].occupied && table[check_idx].key == key) {
                    /* key found add value to vector */ 
                    table[check_idx].indices.push_back(idx);
                    return;
                }
            }
        }
        
        /* find first free bucket */
        size_t free_idx = find_free_bucket(base_idx);
        if (free_idx == capacity) {
            throw std::runtime_error("Hash table full");
        }

        /* calculate distance from original hash position */
        size_t dist = (free_idx + capacity - base_idx) % capacity;

        /* distance is outside neighborhood, need to relocate */
        if (dist >= H) {
            if (!relocate(free_idx, base_idx)) {
                throw std::runtime_error("Failed to relocate item within neighborhood");
            }
            /* recalculate distance after relocation */
            dist = (free_idx + capacity - base_idx) % capacity;
        }
        
        /* verify distance is within bounds */
        if (dist >= H) {
            throw std::runtime_error("Distance exceeds neighborhood size after relocation");
        }
        
        table[free_idx].key = key;
        table[free_idx].indices.push_back(idx);
        table[free_idx].occupied = true;
        table[base_idx].hop_info |= (1ULL << dist);
    }
    
    const std::vector<size_t>* find(const Key& key) const {
        size_t hash_val = hash_key(key);
        size_t base_idx = hash_val % capacity;
        /* find the value based on the bitmap */
        uint64_t hop = table[base_idx].hop_info;
        for (size_t i = 0; i < H; ++i) {
            if (hop & (1ULL << i)) {
                size_t check_idx = (base_idx + i) % capacity;
                if (table[check_idx].occupied && table[check_idx].key == key) {
                    return &table[check_idx].indices;
                }
            }
        }
        return nullptr;
    }

    void diagnostic() {
        for (size_t i = 0; i < capacity; i++) {
            if (table[i].occupied) {
                std::cout << "Table index: " << i << " occupied by key: " << table[i].key << std::endl;
                std::cout << "  Bucket contains:" << std::endl;
                auto bucket = &table[i].indices;
                for (auto value : *bucket) {
                    std::cout << "____" << value << std::endl;
                }
                std::bitset<64> bitmask(table[i].hop_info);
                std::cout << "Bitmap: " << bitmask << std::endl;
            } else {
                std::cout << "Table index: " << i << " unoccupied" <<std::endl;
            }
        }
    }
    
    size_t size() const { return capacity; }
};

#endif

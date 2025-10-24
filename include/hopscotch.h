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
#include <llvm/ADT/SmallVector.h>

template <typename Key, typename Hash = std::hash<Key>>
class HopscotchHashTable {
private:
    static constexpr size_t H = 64;
    static constexpr uint64_t EMPTY_MASK = 0;

    struct Bucket {
        /* bit mask of the bucket*/
        uint64_t bitmask;
        /* vector containing same key idx*/
        llvm::SmallVector<size_t, 2> indices;
        /* key corresponding to bucket */
        Key key;
        bool occupied;
        
        Bucket() : bitmask(EMPTY_MASK), occupied(false) {}
    }__attribute__((aligned(64)));

    std::vector<Bucket> table;
    size_t capacity;
    size_t num_items;
    Hash hasher;
    

    __attribute__((always_inline))
    inline size_t hash_key(const Key& key) const noexcept {
        if constexpr (std::is_same_v<Key, int32_t>)
            return hash_int32(key);
        return hasher(key);
    }

    /* ultra complicated hash expanding to 64 bit */
    __attribute__((always_inline))
    static inline size_t hash_int32(int32_t key) noexcept {
        uint32_t h = static_cast<uint32_t>(key);
        
        h = murmur3_fmix32(h);
        
        uint64_t h64 = h;
        h64 = (h64 ^ 0x1312acab) + (h64 << 5);
        h64 ^= h64 >> 23;
        h64 *= 0x2127599bf4325c37ULL;
        h64 ^= h64 >> 47;
        
        h64 *= 0xbf58476d1ce4e5b9ULL;
        h64 ^= h64 >> 27;
        h64 *= 0x94d049bb133111ebULL;
        h64 ^= h64 >> 31;
        
        return h64;
    }

    /*optimized hash function for int32_t keys*/
    __attribute__((always_inline))
    static inline uint32_t murmur3_fmix32(uint32_t h) noexcept {
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }

    
   
    /*find free bucket within max distance from ideal position*/
    inline size_t find_free_bucket(size_t start_idx) const noexcept {
        for (size_t i = 0; i < capacity; ++i) {
            size_t idx = (start_idx + i) & (capacity - 1);
            if (!table[idx].occupied) {
                return idx;
            }
        }
        return capacity; /*no free bucket found*/
    }
    
    /* get a free bucket within H positions*/
    bool relocate(size_t &free_index, size_t target_idx) {
        /* relocation attempt limit */
        static constexpr size_t MAX_DEPTH = 128;
        
        for (size_t depth = 0; depth < MAX_DEPTH; ++depth) {
            size_t dist = (free_index + capacity - target_idx) & (capacity - 1);
            /* if we are within the neighborhood we are done */
            if (dist < H) {
                return true;
            }
            
            bool found = false;
           /* this loop looks backwards from free_index
            * and checks occupied items' bitmask*/ 
            for (size_t offset = 1; offset < H && !found; ++offset) {
                size_t check_index = (free_index + capacity - offset) & (capacity - 1);
                
                if (!table[check_index].occupied) continue;
                
                uint64_t bitmask = table[check_index].bitmask;
                
                for (size_t j = H - 1; j > 0 && !found; --j) {
                    /* check its bitmap position */
                    if (bitmask & (1ULL << j)) {
                        size_t item_idx = (check_index + j) & (capacity - 1);
                        size_t new_dist = (free_index + capacity - check_index) & (capacity - 1);
                       /* check if the item can be moved to free_index while
                        * staying in its appropiate neighborhood if this can
                        * be done move the items */
                        if (new_dist < H && item_idx != free_index) {
                            table[free_index] = std::move(table[item_idx]);
                            table[item_idx].occupied = false;
                            table[item_idx].bitmask = EMPTY_MASK;
                            /* clears and sets proper bit map positions */ 
                            table[check_index].bitmask &= ~(1ULL << j);
                            table[check_index].bitmask |= (1ULL << new_dist);
                            
                            free_index = item_idx;
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
        static constexpr size_t MIN_CAPACITY = 64;
        static constexpr size_t MAX_CAPACITY = 1ULL << 62;

        if (size == 0) {
            capacity = MIN_CAPACITY; 
        } else if (size > MAX_CAPACITY) {
            throw std::invalid_argument("Cannot build hash table of this size");
        } else {
            /* calculate next power of 2 */
            size_t clz = __builtin_clzll(size - 1);
            capacity = 1ULL << (64 - clz);
        }
        table.resize(capacity);
    }
    
    void insert(const Key& key, size_t idx) {
        size_t hash_val = hash_key(key);
        size_t base_index = hash_val & (capacity - 1);
        
        /* check if key has already been inserted*/ 
        uint64_t bitmask = table[base_index].bitmask;
        uint64_t temp_mask = bitmask;
        while (temp_mask) {
            size_t i = __builtin_ctzll(temp_mask);
            size_t check_index = (base_index + i) & (capacity - 1);
            if (table[check_index].occupied && table[check_index].key == key) {
                /* key found add value to vector */ 
                table[check_index].indices.emplace_back(idx);
                return;
            }
            /*clear bit */
            temp_mask &= (temp_mask - 1);
        }
        
        /* find first free bucket */
        size_t free_index = find_free_bucket(base_index);
        if (free_index == capacity) {
            throw std::runtime_error("Hash table full");
        }

        /* calculate distance from original hash position */
        size_t dist = (free_index + capacity - base_index) & (capacity - 1);

        /* distance is outside neighborhood, need to relocate */
        if (dist >= H) {
            if (!relocate(free_index, base_index)) {
                throw std::runtime_error("Failed to relocate item within neighborhood");
            }
            /*recalculate distance after relocation */
            dist = (free_index + capacity - base_index) & (capacity - 1);
        }
        
        /* verify distance is within bounds */
        if (dist >= H) {
            throw std::runtime_error("Distance exceeds neighborhood size after relocation");
        }
        /* update table entries with proper data */ 
        table[free_index].key = key;
        table[free_index].indices.emplace_back(idx);
        table[free_index].occupied = true;
        table[base_index].bitmask |= (1ULL << dist);
    }
    
    inline const llvm::SmallVector<size_t, 2>* find(const Key& key) const noexcept{
        size_t hash_val = hash_key(key);
        size_t base_index = hash_val & (capacity - 1);

        __builtin_prefetch(&table[base_index], 0 , 3);
        /* find the value based on the bitmap */
        uint64_t bitmask = table[base_index].bitmask;
        if (bitmask == 0)
            return nullptr;

        /* bit scanning */
        while (bitmask) {
            size_t i = __builtin_ctzll(bitmask);
            size_t check_index = (base_index + i) & (capacity - 1);
            if (table[check_index].occupied && table[check_index].key == key) {
                return &table[check_index].indices;
            }
            /* clear bit */
            bitmask &= (bitmask - 1);
        }
       return nullptr;
    }


    /* diagnostic functions that prints all data 
     * in the hash table useful for tricky debugging */
    void diagnostic() {
        for (size_t i = 0; i < capacity; i++) {
            if (table[i].occupied) {
                std::cout << "Table index: " << i << " occupied by key: " << table[i].key << std::endl;
                std::cout << "  Bucket contains:" << std::endl;
                auto bucket = &table[i].indices;
                for (auto value : *bucket) {
                    std::cout << "____" << value << std::endl;
                }
                std::bitset<64> bitmask(table[i].bitmask);
                std::cout << "Bitmap: " << bitmask << std::endl;
            } else {
                std::cout << "Table index: " << i << " unoccupied" <<std::endl;
            }
        }
    }
    
    size_t size() const { return capacity; }
};

#endif

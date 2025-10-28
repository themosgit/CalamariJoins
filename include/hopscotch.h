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
    static constexpr uint8_t H = 64;

    /**
     *
     *  first is the bit mask of the bucket
     *  with occupied state stored at the last bit
     *  or first bit 
     *
     *  then the index of the values within
     *  are value store vector the count of the
     *  items and padding to keep buckets at 
     *  32 bytes on the M1 proccesor cache line
     *  size is ta 128 bytes so we fit 4 buckets
     *  per cacheline
     *
     **/

     inline static constexpr size_t BUCKET_SIZE =
         (sizeof(Key) <=4) ? 32 :
         (sizeof(Key) <= 8) ? 32 : 64;

     struct alignas(BUCKET_SIZE) Bucket {
        uint64_t bitmask;
        Key key;
        uint32_t value_index;
        uint16_t count;
        bool occupied;

        Bucket() : bitmask(0), value_index(0), count(0), occupied(false){}
    };


    std::vector<Bucket> table;
    std::vector<size_t> value_store;
    size_t capacity;
    Hash hasher;
    

    __attribute__((always_inline))
    inline size_t hash_key(const Key& key) const noexcept {
        if constexpr (std::is_same_v<Key, int32_t>)
            return hash_int32(key);
        return hasher(key);
    }

    __attribute__((always_inline))
    static inline size_t hash_int32(int32_t key) noexcept {
        #if defined(__aarch64__)
        return __builtin_arm_crc32h(key, 0);
        #else
        uint64_t h = static_cast<uint32_t>(key);
        h ^= h << 32;
        h *= 0x9e3779b97f4a7c15ULL;
        h ^= h >> 32;
        h *= 0xc2b2ae3d27d4eb4fULL;
        h ^= h >> 29;
        return h;
        #endif
    } 
    /*find free bucket within search distance from current position*/
    inline size_t find_free_bucket(size_t start_index) const noexcept {
        static constexpr size_t MAX_SEARCH = 256;
        for (size_t i = 0; i < MAX_SEARCH && i < capacity; ++i) {
            size_t index = (start_index + i) & (capacity - 1);
            if (!table[index].occupied) {
                return index;
            }
        }
        return capacity; /*no free bucket found*/
    }
    
    /* get a free bucket within H positions*/
    bool relocate(size_t &free_index, size_t target_index) {
        /* relocation attempt limit */
        static constexpr size_t MAX_DEPTH = 128;
        
        for (size_t depth = 0; depth < MAX_DEPTH; ++depth) {
            size_t dist = (free_index + capacity - target_index) & (capacity - 1);
            /* if we are within the neighborhood we are done */
            if (dist < H) {
                return true;
            }
            
            bool found = false;

           /* looks backwards from free_index  and checks occupied items' bitmask*/ 
            for (size_t offset = 1; offset < H && !found; ++offset) {
                size_t check_index = (free_index + capacity - offset) & (capacity - 1);
                if (!table[check_index].occupied) continue;
                
                uint64_t bitmask = table[check_index].bitmask;
                for (size_t j = H - 1; j > 0 && !found; --j){
                    /* check its bitmap position */
                    if (bitmask & (1ULL << j)) {
                        size_t item_index = (check_index + j) & (capacity - 1);
                        size_t new_dist = (free_index + capacity - check_index) & (capacity - 1);
                       /**
                        *  check if the item can be moved to free_index while
                        *  staying in its appropiate neighborhood if this can
                        *  be done move the items 
                        **/
                        if (new_dist < H && item_index != free_index) {
                            table[free_index] = table[item_index];
                            table[item_index].occupied = false;
                            table[item_index].bitmask = 0;
                            /* clears and sets proper bit map positions */
                            table[check_index].bitmask &= ~(1ULL << j);
                            table[check_index].bitmask |= (1ULL << new_dist);
                            
                            free_index = item_index;
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
    /**
     *
     *  Hopscotch hash table constructor
     *  determines capacity from powers of 2
     *
     **/
    explicit HopscotchHashTable(size_t size, const Hash& hash = Hash()) 
        : capacity(0), hasher(hash) {
        static constexpr size_t MIN_CAPACITY = 64;
        static constexpr size_t MAX_CAPACITY = 1ULL << 62;

        if (size == 0 || size < MIN_CAPACITY) {
            capacity = MIN_CAPACITY;
        } else if (size <= MAX_CAPACITY){
            capacity = 1ULL << (64 - __builtin_clzll(size - 1));
        } else {
            throw std::invalid_argument("Cannot build hash table of this size");
        }
        table.resize(capacity);
    }

    /**
     *
     *  Insert first checks for same duplicates
     *  then finds first available bucket if within 
     *  H then insert otherwise attempt to relocate
     *  then insert
     *
     **/
    void insert(const Key& key, size_t item) {
        size_t hash_val = hash_key(key);
        size_t base_index = hash_val & (capacity - 1);
        
        uint64_t temp_mask = table[base_index].bitmask;
        while (temp_mask) {
            int i = __builtin_ctzll(temp_mask);
            size_t check_index = (base_index + i) & (capacity - 1);
            if (table[check_index].key == key) {

                size_t old_index = table[check_index].value_index;
                size_t old_count = table[check_index].count;

                value_store.push_back(item);

                if (old_index + old_count == value_store.size() - 1) {
                    table[check_index].count++;
                    return;
                }

                size_t new_index = value_store.size() - 1;
                for (size_t j = 0; j < old_count; j++) {
                    value_store.push_back(value_store[old_index + j]);
                }

                table[check_index].value_index = new_index;
                table[check_index].count = old_count + 1;
                return;
            }
            temp_mask &= (temp_mask - 1);
        }
        
        size_t free_index = find_free_bucket(base_index);
        if (free_index == capacity) {
            throw std::runtime_error("Hash table full");
        }

        size_t dist = (free_index + capacity - base_index) & (capacity - 1);

        if (dist >= H) {
            if (!relocate(free_index, base_index)) {
                throw std::runtime_error("Failed to relocate item within neighborhood");
            }
            dist = (free_index + capacity - base_index) & (capacity - 1);
        }
        
        if (dist >= H) {
            throw std::runtime_error("Distance exceeds neighborhood size after relocation");
        }

        table[free_index].key = key;
        table[free_index].value_index = value_store.size();
        table[free_index].count = 1;
        value_store.push_back(item);
        
        table[free_index].occupied = true;
        table[base_index].bitmask |= (1ULL << dist);
    }


    /** 
     *
     *  struct to return pointers
     *  avoids inderections 
     *  with support for iterators
     *
     **/
    struct IndexSpan {
        const size_t *data;
        size_t size;
        const size_t *begin() const {
            return data; 
        }
        const size_t *end() const {
            return data + size;
        }
    };
   
    inline IndexSpan find(const Key& key) const noexcept{
        size_t hash_val = hash_key(key);
        size_t base_index = hash_val & (capacity - 1);

        /* find the value based on the bitmap */
        uint64_t temp_mask = table[base_index].bitmask;
        if (temp_mask == 0)
            return {nullptr, 0};

        while (temp_mask) {
            uint8_t i =  __builtin_ctzll(temp_mask);
            size_t check_index = (base_index + i) & (capacity - 1);

            const Bucket& bucket = table[check_index];

            if (bucket.key == key) {
                return {
                    value_store.data() + bucket.value_index,
                    bucket.count 
                };
            }
            /* clear bit */
            temp_mask &= (temp_mask - 1);
        }
       return {nullptr, 0};
    }

    /**
     *
     *  diagnostic functions that prints all data 
     *  in the hash table useful for tricky debugging 
     *
     **/
    void diagnostic() {
        for (size_t i = 0; i < capacity; i++) {
            if (table[i].occupied) {
                std::cout << "Table index: " << i << " occupied by key: " << table[i].key << std::endl;
                std::cout << "Storage index: " << table[i].value_index << " Storage count: " << table[i].count << std::endl;
                std::bitset<64> bitmask(table[i].bitmask);
                std::cout << "Bitmap: " << bitmask << std::endl; std::cout << "  Bucket contains:" << std::endl;
                int index = table[i].value_index;
                for (int j = 0; j < table[i].count; j++) {
                    auto value = value_store[index + j]; 
                    std::cout << "____" << value << std::endl;
                }
            } else {
                std::cout << "Table index: " << i << " unoccupied" <<std::endl;
            }
        }
    }
    
    size_t size() const { return capacity; }
};

#endif

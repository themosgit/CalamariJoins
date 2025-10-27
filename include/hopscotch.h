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
    static constexpr uint8_t H = 128;

    /**
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
     **/

     inline static constexpr size_t BUCKET_SIZE =
         (sizeof(Key) <=4) ? 32 :
         (sizeof(Key) <= 8) ? 32 : 64;

     struct alignas(BUCKET_SIZE) Bucket {
        uint64_t bitmask_low;
        uint64_t bitmask_high;
        Key key;
        uint32_t value_index;
        uint16_t count;
        bool occupied;

        Bucket() : bitmask_low(0), bitmask_high(0), value_index(0), count(0), occupied(false){}
    };


    std::vector<Bucket> table;
    std::vector<size_t> value_store;
    size_t capacity;
    size_t value_store_size;
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
        uint64_t h = static_cast<uint32_t>(key);

        h ^= h << 32;
        h *= 0x9e3779b97f4a7c15ULL;
        h ^= h >> 32;
        h *= 0xc2b2ae3d27d4eb4fULL;
        h ^= h >> 29;
        return h;
    }
    __attribute__((always_inline))
    static inline int ctz128(uint64_t low, uint64_t high) noexcept {
        int low_ctz = __builtin_ctzll(low | 64);
        int high_ctz = __builtin_ctzll(high | 64);
        bool use_high = (low == 0);
        return use_high ? (64 + high_ctz) : low_ctz;
    }

    __attribute__((always_inline))
    static inline void set_bit(uint64_t& low, uint64_t& high, size_t pos) noexcept {
        uint64_t mask_low = uint64_t(pos < 64) << (pos & 63); 
        uint64_t mask_high = uint64_t(pos >= 64) << (pos & 63);
        low |= mask_low;
        high |= mask_high;
    }
 
    __attribute__((always_inline))
    static inline void clear_bit(uint64_t& low, uint64_t& high, size_t pos) noexcept {
        uint64_t mask_low = uint64_t(pos < 64) << (pos & 63); 
        uint64_t mask_high = uint64_t(pos >= 64) << (pos & 63);
        low &= ~mask_low;
        high &= ~mask_high;
    }
   
    __attribute__((always_inline))
    static inline bool test_bit(uint64_t low, uint64_t high, size_t pos) noexcept {
        uint64_t value = (pos < 64) ? low : high;
        return (value >> (pos & 63)) & 1;
    }
   
    /*find free bucket within search distance from current position*/
    inline size_t find_free_bucket(size_t start_index) const noexcept {
        static constexpr size_t MAX_SEARCH = 512;
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
        static constexpr size_t MAX_DEPTH = 256;
        
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

                if (offset + 4 < H) {
                    __builtin_prefetch(&table[(free_index + capacity - offset - 4) & (capacity - 1)], 0, 2);
                }
                
                if (!table[check_index].occupied) continue;
                
                uint64_t bitmask_low = table[check_index].bitmask_low;
                uint64_t bitmask_high = table[check_index].bitmask_high;
                
                for (size_t j = H - 1; j > 0 && !found; --j){
                    /* check its bitmap position */
                    if (test_bit(bitmask_low, bitmask_high, j)) {
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
                            table[item_index].bitmask_low = 0;
                            table[item_index].bitmask_high = 0;
                            /* clears and sets proper bit map positions */
                            clear_bit(table[check_index].bitmask_low, table[check_index].bitmask_high, j);
                            set_bit(table[check_index].bitmask_low, table[check_index].bitmask_high, new_dist);
                            
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
    explicit HopscotchHashTable(size_t size, const Hash& hash = Hash()) 
        : capacity(0), value_store_size(0), hasher(hash) {
        static constexpr size_t MIN_CAPACITY = 64;
        static constexpr size_t MAX_CAPACITY = 1ULL << 62;

        if (size == 0) {
            capacity = MIN_CAPACITY; 
        } else if (size > MAX_CAPACITY) {
            throw std::invalid_argument("Cannot build hash table of this size");
        } else if (size < MIN_CAPACITY) {
            capacity = MIN_CAPACITY;
        } else {
            /* calculate next power of 2 */
            capacity = 1ULL << (64 - __builtin_clzll(size - 1));
        }
        table.resize(capacity);
    }
    
    void insert(const Key& key, size_t item) {
        size_t hash_val = hash_key(key);
        size_t base_index = hash_val & (capacity - 1);
        
        /* check if key has already been inserted */
        uint64_t temp_mask_low = table[base_index].bitmask_low;
        uint64_t temp_mask_high = table[base_index].bitmask_high;
        while (temp_mask_low || temp_mask_high) {
            int i = temp_mask_low ? __builtin_ctzll(temp_mask_low) :
                __builtin_ctzll(temp_mask_high) + 64;

            size_t check_index = (base_index + i) & (capacity - 1);
            if (table[check_index].key == key) {

                size_t old_index = table[check_index].value_index;
                size_t old_count = table[check_index].count;

                value_store.push_back(item);

                if (old_index + old_count == value_store_size) {
                    table[check_index].count++;
                    value_store_size++;
                    return;
                }

                size_t new_index = value_store_size;
                for (size_t j = 0; j < old_count; j++) {
                    value_store.push_back(value_store[old_index + j]);
                }

                table[check_index].value_index = new_index;
                table[check_index].count = old_count + 1;
                value_store_size = value_store.size();
                return;
            }
            /* clear bit */
            temp_mask_low ? temp_mask_low &= (temp_mask_low - 1) :
                temp_mask_high &= (temp_mask_high - 1);
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
        table[free_index].value_index = value_store_size;
        table[free_index].count = 1;
        value_store.push_back(item);
        value_store_size++;
        

        table[free_index].occupied = true;
        set_bit(table[base_index].bitmask_low, table[base_index].bitmask_high, dist);
    }


    /** 
     *  struct to return pointers
     *  avoids inderections 
     *
     **/
    struct IndexSpan {
        const size_t *data;
        size_t size;

        /* support iterators */
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

        __builtin_prefetch(&table[base_index], 0 , 3);
        /* find the value based on the bitmap */
        uint64_t temp_mask_low = table[base_index].bitmask_low;
        uint64_t temp_mask_high = table[base_index].bitmask_high;

        if ((temp_mask_low | temp_mask_high) == 0)
            return {nullptr, 0};

        /* bit scanning */
        while (temp_mask_low || temp_mask_high) {
            int i = temp_mask_low ? __builtin_ctzll(temp_mask_low) :
                __builtin_ctzll(temp_mask_high) + 64;
            size_t check_index = (base_index + i) & (capacity - 1);

            const Bucket& bucket = table[check_index];
            if (bucket.count > 0) {
                __builtin_prefetch(&value_store[bucket.value_index], 0, 3);
            }

            if (bucket.key == key) {
                return {
                    value_store.data() + bucket.value_index,
                    bucket.count 
                };
            }
            /* clear bit */

            temp_mask_low ? temp_mask_low &= (temp_mask_low - 1) :
                temp_mask_high &= (temp_mask_high - 1);
        }
       return {nullptr, 0};
    }

    inline void prefetch(const Key& key) const noexcept {
        size_t hash_val = hash_key(key);
        size_t base_index = hash_val & (capacity - 1);
        __builtin_prefetch(&table[base_index], 0, 3);
    }


    /**
     *  diagnostic functions that prints all data 
     *  in the hash table useful for tricky debugging 
     *
     **/
    void diagnostic() {
        for (size_t i = 0; i < capacity; i++) {
            if (table[i].occupied) {
                std::cout << "Table index: " << i << " occupied by key: " << table[i].key << std::endl;
                std::cout << "Storage index: " << table[i].value_index << " Storage count: " << table[i].count << std::endl;
                std::bitset<128> bitmask(table[i].bitmask);
                std::cout << "Bitmap: " << bitmask << std::endl;
                std::cout << "  Bucket contains:" << std::endl;
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

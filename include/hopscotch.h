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
    static constexpr size_t H = 64;
    static constexpr uint64_t EMPTY_MASK = 0;

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
     *
     **/
    struct Bucket {
        uint64_t internal_bitmask;
        Key key;
        uint32_t value_index;
        unint16_t count;
        unint16_t _padding;

        Bucket() : internal_bitmask(EMPTY_MASK), value_index(0), count(0), _padding(0){}

        inline bool occupied() const noexcept {
            return internal_bitmask & (1ULL << 63);
        }

        inline void set_occupied(bool occupied) noexcept {
            internal_bitmask = (internal_bitmask & 0x7FFFFFFFFFFFFFFFULL) | 
                       (static_cast<uint64_t>(occupied) << 63);
        }

        inline uint64_t bitmask() const noexcept {
            return internal_bitmask & 0x7FFFFFFFFFFFFFFFULL;
        }

        inline void set_bitmask(size_t pos) noexcept {
            internal_bitmask |= (1ULL << pos);
        }

        inline void clear_bitmask(size_t pos) noexcept {
            internal_bitmask &= ~(1ULL << pos);
        }

    }__attribute__((aligned(32)));

    static_assert(sizeof(Bucket) == 32, "Bucket should be 32 bytes");

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
   
    /*find free bucket within search distance from current position*/
    inline size_t find_free_bucket(size_t start_idx) const noexcept {
        static constexpr size_t MAX_SEARCH = 512;
        for (size_t i = 0; i < MAX_SEARCH && i < capacity; ++i) {

            size_t idx = (start_idx + i) & (capacity - 1);
            if (!table[idx].occupied()) {
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

           /* looks backwards from free_index  and checks occupied items' bitmask*/ 
            for (size_t offset = 1; offset < H && !found; ++offset) {
                size_t check_index = (free_index + capacity - offset) & (capacity - 1);

                if (offset + 4 < H) {
                    __builtin_prefetch(&table[(free_index + capacity - offset - 4) & (capacity - 1)], 0, 2);
                }
                
                if (!table[check_index].occupied()) continue;
                
                uint64_t bitmask = table[check_index].bitmask();
                
                for (size_t j = H - 1; j > 0 && !found; --j){
                    /* check its bitmap position */
                    if (bitmask & (1ULL << j)) {
                        size_t item_idx = (check_index + j) & (capacity - 1);
                        size_t new_dist = (free_index + capacity - check_index) & (capacity - 1);
                       /**
                        *  check if the item can be moved to free_index while
                        *  staying in its appropiate neighborhood if this can
                        *  be done move the items 
                        **/
                        if (new_dist < H && item_idx != free_index) {
                            table[free_index] = table[item_idx];
                            table[item_idx].set_occupied(false);
                            table[item_idx].internal_bitmask = EMPTY_MASK;
                            /* clears and sets proper bit map positions */
                            table[check_index].clear_bitmask(j);
                            table[check_index].set_bitmask(new_dist);
                            
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
            capacity = 1ULL << (64 - __builtin_clzll(size - 1));
        }
        table.resize(capacity);
    }
    
    void insert(const Key& key, size_t idx) {
        size_t hash_val = hash_key(key);
        size_t base_index = hash_val & (capacity - 1);
        
        /* check if key has already been inserted */ 
        uint64_t bitmask = table[base_index].bitmask();
        uint64_t temp_mask = bitmask;
        while (temp_mask) {
            size_t i = __builtin_ctzll(temp_mask);
            size_t check_index = (base_index + i) & (capacity - 1);
            if (table[check_index].key == key) {
                /* key found add value to vector */ 
                value_storage.emplace_back(idx);
                tble[check_index].count++;
                return;
            }
            /* clear bit */
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
        table[free_index].value_index = value_vector_size;
        table[free_index].count = 1;
        value_store.emplace_back(idx);
        value_store_size++;
        

        table[free_index].set_occupied(true);
        table[base_index].set_bitmask(dist);
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
        uint64_t bitmask = table[base_index].bitmask();
        if (bitmask == 0)
            return {nullptr, 0};

        /* bit scanning */
        while (bitmask) {
            size_t i = __builtin_ctzll(bitmask);
            size_t check_index = (base_index + i) & (capacity - 1);

            const Bucket& bucket = table[check_index];
            if (bucket.count > 0) {
                __builtin_prefetch(&value_store[bucket.flat_index], 0, 3);
            }

            if (bucket.key == key) {
                return {
                    value_store.data() + bucket.value_index,
                    bucket.count 
                };
            }
            /* clear bit */
            bitmask &= (bitmask - 1);
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
            if (table[i].occupied()) {
                std::cout << "Table index: " << i << " occupied by key: " << table[i].key << std::endl;
                std::cout << "  Bucket contains:" << std::endl;
                auto bucket = &table[i].indices;
                for (auto value : *bucket) {
                    std::cout << "____" << value << std::endl;
                }
                std::bitset<64> bitmask(table[i].bitmask());
                std::cout << "Bitmap: " << bitmask << std::endl;
            } else {
                std::cout << "Table index: " << i << " unoccupied" <<std::endl;
            }
        }
    }
    
    size_t size() const { return capacity; }
};

#endif

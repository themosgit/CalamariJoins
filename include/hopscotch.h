#pragma once

#if defined(__x86_64__)
    #include  <immintrin.h>
#endif

#include <hashtable_structs.h>
#include <cstdlib>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <type_traits>

/**
 *
 *  This is the main class of the hopscotch table
 *  implementation. The hopscotch table consists 
 *  of three vectors. 
 *
 *  The first vector is the hash table itself,
 *  storing buckets. Buckets do not contain
 *  the actual values themselves, for further
 *  detail reference hopscotch_structs.h.
 *
 *  The value_store vector stores all the items,
 *  no keys just values.
 *
 *  The segments vector is a vector of segments.
 *  Segments store indices to value_store that
 *  correspond to ranges of duplicate key
 *  insertions. For more details reference
 *  hopscotch_structs.h.
 *
 *  For hashing by default std::hash is used
 *  except for when the type is uint32_t, which
 *  is something that always happens.
 *  
 **/

template <typename Key>
struct alignas(sizeof(Key) <= 8 ? 32 : 64) Bucket {
    uint64_t bitmask;
    Key key;
    uint32_t first_segment;
    uint32_t last_segment;
    uint16_t count;
    bool occupied;

    Bucket() : bitmask(0), occupied(false), first_segment(0), 
               last_segment(UINT32_MAX), count(0) {}
};


template <typename Key, typename Hash = std::hash<Key>>
class HopscotchHashTable {
private:
    /* neighbourhood size */
    static constexpr uint8_t H = 64;

    std::vector<Bucket<Key>> table;
    std::vector<uint32_t> value_store;
    std::vector<Segment> segments;
    size_t capacity;
    Hash hasher;

    inline size_t hash_key(const Key& key) const noexcept {
        if constexpr (std::is_same_v<Key, int32_t>)
            return hash_int32(key);
        return hasher(key);
    }

    static inline size_t hash_int32(int32_t key) noexcept {
        #if defined(__aarch64__)
            return __builtin_arm_crc32w(key, 0);
        #elif defined(__x86_64__)
            return __builtin_ia32_crc32di(key, 0);
        #endif
    }

    inline size_t find_free_bucket(size_t start_index) const noexcept {
        for (size_t i = 0; i < capacity; ++i) {
            size_t index = (start_index + i) & (capacity - 1);
            if (!table[index].occupied) {
                return index;
            }
        }
        return capacity;
    }

    /**
     *
     *  Relocate handles the procces of moving buckets
     *  inorder to free up a spot within the neighbourhood
     *  of target_index
     *
     *  The search is backwards starting from the free bucket
     *  and looks to move within  that neighbourhood clearing
     *  potitions closer to the target index each iteration
     *
     **/
    bool relocate(size_t &free_index, size_t target_index) {
        /* limit attempts */
        static constexpr size_t MAX_DEPTH = 128;
        const size_t capacity_mask = capacity - 1;
        for (size_t depth = 0; depth < MAX_DEPTH; ++depth) {
            /* if the free bucket is within the target H success */
            if (((free_index - target_index) & capacity_mask) < H) {
                return true;
            }
            bool moved = false;
            for (size_t offset = 1; offset < H && !moved; ++offset) {
                size_t check_index = (free_index - offset) & capacity_mask;
                uint64_t bitmask = table[check_index].bitmask;
                /*doesnt house any items of its own in  neighbourhood*/
                if (!table[check_index].occupied || bitmask == 0) continue;
                /* check buckets in neighbourhood  */
                uint64_t temp_mask = bitmask;
                while (temp_mask) {
                    size_t j = 63 - __builtin_clzll(temp_mask);
                    size_t item_index = (check_index + j) & capacity_mask;
                    size_t new_dist = (free_index - check_index) & capacity_mask;

                    if (new_dist < H && item_index != free_index) {
                        table[free_index] = table[item_index];
                        table[item_index].occupied = false;
                        table[item_index].bitmask = 0;
                        table[check_index].bitmask = (bitmask & ~(1ULL << j)) | (1ULL << new_dist);
                        
                        free_index = item_index;
                        moved = true;
                        break;
                    }
                    
                    temp_mask &= ~(1ULL << j);
                }
            }
            
            if (!moved) return false;
        }
        
        return false;
    }

public:
    /* Hopscotch table constructor */
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
        value_store.reserve(size / 2);
        segments.reserve(size / 4);
    }

    void insert(const Key& key, uint32_t item) {
        size_t hash_val = hash_key(key);
        size_t base_index = hash_val & (capacity - 1);
        uint64_t temp_mask = table[base_index].bitmask;
        /* check for pre existing bucket */
        while (temp_mask) {
            uint8_t i = __builtin_ctzll(temp_mask);
            size_t check_index = (base_index + i) & (capacity - 1);

            if (table[check_index].key != key) {
                temp_mask &= (temp_mask - 1);
            } else {
                return insert_duplicate(table[check_index],
                        item, value_store, segments); 
            }
        }
        /* new key find a free bucket for it */
        size_t free_index = find_free_bucket(base_index);
        if (free_index == capacity)
            throw std::runtime_error("Hash table full");

        /* bring free bucket distance within H if not already */
        size_t dist = (free_index + capacity - base_index) & (capacity - 1);
        if (dist >= H) {
            if (!relocate(free_index, base_index))
                throw std::runtime_error("Failed to relocate item within neighborhood");
        }
        /* insert new value/key inplace */
        table[free_index].key = key;
        table[free_index].last_segment = item;
        table[free_index].count = 1;
        table[free_index].occupied = true;
        table[base_index].bitmask |= (1ULL << dist);
    }

    /**
     *
     *  Uses bitmask to quickly find the correct bucket
     *  then return an iterator compatible struct.
     *
     *  The implementation of this struct can be found in
     *  hopscotch_structs.h
     *
     **/
    inline ValueSpan<Key> find(const Key& key) const noexcept{
        size_t hash_val = hash_key(key);
        size_t base_index = hash_val & (capacity - 1);
        uint64_t temp_mask = table[base_index].bitmask;
        if (temp_mask == 0)
            return {nullptr, nullptr, UINT32_MAX,  0};

        while (temp_mask) {
            uint8_t i =  __builtin_ctzll(temp_mask);
            size_t check_index = (base_index + i) & (capacity - 1);
            const Bucket<Key>& bucket = table[check_index];
            if (bucket.key == key) {
                if (bucket.count == 1) {
                    return { nullptr, nullptr,
                        bucket.last_segment, bucket.count };
                }
                return { &value_store, &segments,
                    bucket.first_segment, bucket.count };
            }
            temp_mask &= (temp_mask - 1);
        }
       return {nullptr, nullptr, UINT32_MAX, 0};
    }
    inline size_t size() const noexcept {
        return table.size();
    }
};

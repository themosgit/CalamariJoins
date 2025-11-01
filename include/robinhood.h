#pragma once

#if defined(__x86_64__)
    #include  <immintrin.h>
#endif

#include <vector>
#include <cstddef>
#include <functional>
#include <cstdint>
#include <hashtable_structs.h>

/*
Robin hood hashing algorithm: the idea behind this hashing algorithm is
to keep the psl(probe sequence length) as low as possible.
Psl is the distance from key's ideal position to actual position.
on collision if new key has higher psl than existing key, they swap positions
this keeps the variance low and search fast

Enhanced with value store and segment logic for efficient duplicate key handling
*/


template <typename Key>
struct Entry {
    Key key;
    size_t psl;
    uint32_t first_segment;
    uint32_t last_segment;
    uint16_t count;
    
    Entry() : psl(0), first_segment(0), last_segment(UINT32_MAX), count(0) {}
    Entry(Key k, size_t p, uint32_t item)
        : key(k), psl(p), first_segment(0), last_segment(item), count(1) {}
};

template <typename Key, typename Hash = std::hash<Key>>
class RobinHoodTable {
private:
    size_t size;
    std::vector<Entry<Key>> table;
    std::vector<uint32_t> value_store;
    std::vector<Segment> segments;
    Hash hasher;
    
    inline size_t hash(const Key& key) const noexcept {
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
public:
    RobinHoodTable(size_t build_size, const Hash& hash = Hash()) {
        build_size = build_size ? build_size : 1;
        size = 1ULL << (64 - __builtin_clzll(build_size - 1));
        table.resize(size);
        value_store.reserve(build_size / 2);
        segments.reserve(build_size / 4);
    }

    const size_t capacity(void) {
        return size;
    }

    void insert(const Key &key, uint32_t idx) {
        size_t p = hash(key) & (size - 1);
        size_t vpsl = 0;
        // Check if the key already exists
        size_t search_p = p;
        size_t search_psl = 0;
        while (table[search_p].count > 0) {
            if (table[search_p].key == key) {
                insert_duplicate(table[search_p], idx,
                        value_store, segments);
                return;
            }
            if (search_psl > table[search_p].psl) break;
            search_p = (search_p + 1) & (size - 1);
            search_psl++;
        }
        // New key - need to insert
        Key k = key;
        Entry<Key> entry_to_insert(key, vpsl, idx);
        while (table[p].count > 0) {
            if (entry_to_insert.psl > table[p].psl) {
                std::swap(entry_to_insert, table[p]);
            }
            p = (p + 1) & (size - 1);
            entry_to_insert.psl++;
        }
        table[p] = std::move(entry_to_insert);
    }

    ValueSpan<Key> find(const Key &key) const noexcept {
        size_t p = hash(key) & (size - 1);
        size_t vpsl = 0;
        while (table[p].count > 0) {
            if (table[p].key == key) {
                const Entry<Key>& entry = table[p];
                if (entry.count == 1) {
                    return { nullptr, nullptr,
                        entry.last_segment, entry.count };
                }
                return { &value_store, &segments,
                    entry.first_segment, entry.count };
            }
            
            if (vpsl > table[p].psl) break;
            p = (p + 1) & (size - 1);
            vpsl++;
        }
        return {nullptr, nullptr, UINT32_MAX, 0};
    }
};

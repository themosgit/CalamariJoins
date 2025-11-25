#pragma once

#if defined(__x86_64__)
    #include <immintrin.h>
#endif

#include <vector>
#include <cstddef>
#include <cstdint>
#include <hashtable_structs.h>

/**
 *
 *  Robin hood hashing algorithm: the idea behind this hashing
 *  algorithm is to keep the psl(probe sequence length) as low as possible.
 *  Psl is the distance from key's ideal position to actual position.
 *  on collision if new key has higher psl than existing key, they swap
 *  positions this keeps the variance low and search fast enhanced with
 *  value store and segment logic for efficient duplicate key handling
 *
 **/

using Key = int32_t;

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

class RobinHoodTable {
private:
    size_t size;
    std::vector<Entry> table;
    std::vector<uint32_t> value_store;
    std::vector<Segment> segments;
    Entry temp_entry;
    BloomFilter bloom;

    inline size_t hash(const Key& key) const noexcept {
        return (size_t)(key * 0x85ebca6b) | ((size_t)(key * 0xc2b2ae35)) << 32;
    }

    inline void SetTempEntry(const Key& key, uint32_t item) noexcept {
        temp_entry.key = key;
        temp_entry.psl = 0;
        temp_entry.first_segment = 0;
        temp_entry.last_segment = item;
        temp_entry.count = 1;
    }

public:
    RobinHoodTable(size_t build_size) : bloom(build_size) {
        build_size = build_size ? build_size : 1;
        size = 1ULL << (64 - __builtin_clzll(build_size - 1));
        table.resize(size);
        value_store.reserve(build_size / 2);
        segments.reserve(build_size / 4);
    }

    const size_t capacity(void) {
        return size;
    }

    void insert(const Key& key, uint32_t idx) {
        bloom.insert(hash(key));
        SetTempEntry(key, idx);
        size_t p = hash(key) & (size - 1);
        while (table[p].count != 0) {
            if (table[p].key == key) {
                insert_duplicate(table[p], idx,
                    value_store, segments);
                return;
            }
            if (temp_entry.psl > table[p].psl)
                std::swap(temp_entry, table[p]);

            p = (p + 1) & (size - 1);
            temp_entry.psl++;
        }
        table[p] = temp_entry;
    }

    ValueSpan<Key> find(const Key& key) const noexcept {
        if (!bloom.contains(hash(key)))
            return{ nullptr, nullptr, UINT32_MAX, 0};
        size_t p = hash(key) & (size - 1);
        size_t vpsl = 0;
        while (table[p].count > 0) {
            if (table[p].key == key) {
                const Entry& entry = table[p];
                if (entry.count == 1) {
                    return { nullptr, nullptr,
                        entry.last_segment, entry.count };
                }
                return { &value_store, &segments,
                    entry.first_segment, entry.count };
            }
            /**
             *
             *  if the psl of the key we're searching exceeds
             *  the psl of the current bucket  it means that
             *  the key cannot be present beyond this point so
             *  we stop searching.
             *
             **/
            if (vpsl > table[p].psl) break;
            p = (p + 1) & (size - 1);
            vpsl++;
        }
        return {nullptr, nullptr, UINT32_MAX, 0};
    }
};

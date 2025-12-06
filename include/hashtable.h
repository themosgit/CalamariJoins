#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <intermediate.h>
#include <vector>

class UnchainedHashtable {
  public:
    struct Tuple {
        int32_t key;
        uint32_t row_id;
    };

  private:
    std::vector<uint64_t> directory;

    /* structure of srrays layout for better cache locality */
    std::vector<int32_t> keys;
    std::vector<uint32_t> row_ids;

    int shift;

    static inline uint64_t hash_key(int32_t k) {
        uint64_t key = (uint64_t)k;
        return (key * 0x85ebca6b) ^ ((key * 0xc2b2ae35) << 32);
    }

    /**
     *
     *  a clever methods that sets
     *  sets of 4 bits with different
     *  parts of the hash allowing
     *  for more sparsity and
     *  fewer false positives
     *
     **/
    static inline uint16_t compute_bloom(uint64_t hash) {
        uint16_t mask = 0;
        mask |= (1 << (hash & 0xF));
        mask |= (1 << ((hash >> 4) & 0xF));
        mask |= (1 << ((hash >> 8) & 0xF));
        mask |= (1 << ((hash >> 12) & 0xF));
        return mask;
    }

  public:
    UnchainedHashtable(size_t build_size) {
        size_t dir_size = build_size > 0 ? build_size : 1;
        size_t pow2 = 1ULL << (64 - __builtin_clzll(dir_size - 1));

        if (pow2 < 2048)
            pow2 = 2048;

        directory.resize(pow2, 0);

        /**
         *
         *  shift determines the bits used
         *  to select a bucket be using
         *  trailing zeros etc.
         *
         *  this is the exact amout of bits
         *  we need to reference its slot
         *  in the directory
         *
         **/
        shift = 64 - __builtin_ctzll(pow2);
        keys.reserve(build_size);
        row_ids.reserve(build_size);
    }

    void build(const mema::column_t &column) {
        if (column.has_direct_access()) {
            build_dense(column);
        } else {
            build_sparse(column);
        }
    }

    void build(const Column &column) { build_from_column(column); }

  private:
    /* builds the hash table from a non-null column */
    void build_dense(const mema::column_t &column) {
        const size_t rows = column.row_count();
        std::vector<uint32_t> counts(directory.size(), 0);

        /* counts how many values go hash in each bucket */
        for (size_t i = 0; i < rows; ++i) {
            uint64_t h = hash_key(column[i].value);
            counts[h >> shift]++;
        }

        /**
         *
         *  creates prefix sum in order to index
         *  each bucket in the value store
         *
         *  after the prefix sum is calculated
         *  the counts is set to that value
         *  meaning that count temporaraly
         *  stores the indexes to value store
         *  those indexes will be stored in
         *  the directory during the next phase
         *  of building
         *
         **/
        uint32_t current_offset = 0;
        for (size_t i = 0; i < counts.size(); ++i) {
            uint32_t count = counts[i];
            counts[i] = current_offset;
            current_offset += count;
        }

        keys.resize(current_offset);
        row_ids.resize(current_offset);

        /* scatter directly from source - no intermediate buffer needed */
        for (size_t i = 0; i < rows; ++i) {
            int32_t val = column[i].value;
            uint64_t h = hash_key(val);
            size_t slot = h >> shift;
            uint32_t pos = counts[slot]++;
            keys[pos] = val;
            row_ids[pos] = (uint32_t)i;
            directory[slot] |= compute_bloom(h);
        }

        finalize_directory(counts);
    }

    /* build hash table from a column with null values */
    void build_sparse(const mema::column_t &column) {
        const size_t rows = column.row_count();
        std::vector<uint32_t> counts(directory.size(), 0);

        /**
         *
         *  two-pass approach to avoid temp_input buffer
         *  pass 1 count per bucket
         *  pass 2 directly populate
         *
         **/

        for (size_t i = 0; i < rows; ++i) {
            const mema::value_t *val = column.get_by_row(i);
            if (val) {
                uint64_t h = hash_key(val->value);
                counts[h >> shift]++;
            }
        }

        /* prefix sum */
        uint32_t current_offset = 0;
        for (size_t i = 0; i < counts.size(); ++i) {
            uint32_t count = counts[i];
            counts[i] = current_offset;
            current_offset += count;
        }

        keys.resize(current_offset);
        row_ids.resize(current_offset);

        for (size_t i = 0; i < rows; ++i) {
            const mema::value_t *val = column.get_by_row(i);
            if (val) {
                int32_t key_val = val->value;
                uint64_t h = hash_key(key_val);
                size_t slot = h >> shift;
                uint32_t pos = counts[slot]++;
                keys[pos] = key_val;
                row_ids[pos] = (uint32_t)i;
                directory[slot] |= compute_bloom(h);
            }
        }

        finalize_directory(counts);
    }

    /* build hash table from original Column format two-pass approach */
    void build_from_column(const Column &column) {
        std::vector<uint32_t> counts(directory.size(), 0);

        uint32_t global_row_id = 0;
        for (auto *page_obj : column.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page);
            auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
            auto *data_begin = reinterpret_cast<const int32_t *>(page + 4);

            if (num_rows == num_values) {
                for (uint16_t i = 0; i < num_rows; ++i) {
                    uint64_t h = hash_key(data_begin[i]);
                    counts[h >> shift]++;
                }
                global_row_id += num_rows;
            } else {
                auto *bitmap = reinterpret_cast<const uint8_t *>(
                    page + PAGE_SIZE - (num_rows + 7) / 8);
                uint16_t data_idx = 0;
                for (uint16_t i = 0; i < num_rows; ++i) {
                    bool is_valid = bitmap[i / 8] & (1u << (i % 8));
                    if (is_valid) {
                        uint64_t h = hash_key(data_begin[data_idx++]);
                        counts[h >> shift]++;
                    }
                    global_row_id++;
                }
            }
        }

        /* prefix sum to get bucket offsets */
        uint32_t current_offset = 0;
        for (size_t i = 0; i < counts.size(); ++i) {
            uint32_t count = counts[i];
            counts[i] = current_offset;
            current_offset += count;
        }

        keys.resize(current_offset);
        row_ids.resize(current_offset);

        global_row_id = 0;
        for (auto *page_obj : column.pages) {
            auto *page = page_obj->data;
            auto num_rows = *reinterpret_cast<const uint16_t *>(page);
            auto num_values = *reinterpret_cast<const uint16_t *>(page + 2);
            auto *data_begin = reinterpret_cast<const int32_t *>(page + 4);

            if (num_rows == num_values) {
                /* dense page - direct scatter */
                for (uint16_t i = 0; i < num_rows; ++i) {
                    int32_t val = data_begin[i];
                    uint64_t h = hash_key(val);
                    size_t slot = h >> shift;
                    uint32_t pos = counts[slot]++;
                    keys[pos] = val;
                    row_ids[pos] = global_row_id + i;
                    directory[slot] |= compute_bloom(h);
                }
                global_row_id += num_rows;
            } else {
                /* sparse page - selective copy */
                auto *bitmap = reinterpret_cast<const uint8_t *>(
                    page + PAGE_SIZE - (num_rows + 7) / 8);
                uint16_t data_idx = 0;
                for (uint16_t i = 0; i < num_rows; ++i) {
                    bool is_valid = bitmap[i / 8] & (1u << (i % 8));
                    if (is_valid) {
                        int32_t val = data_begin[data_idx++];
                        uint64_t h = hash_key(val);
                        size_t slot = h >> shift;
                        uint32_t pos = counts[slot]++;
                        keys[pos] = val;
                        row_ids[pos] = global_row_id;
                        directory[slot] |= compute_bloom(h);
                    }
                    global_row_id++;
                }
            }
        }

        finalize_directory(counts);
    }

    /**
     *
     *  packs blooms filter and value store index
     *  upper 48 bits are the index in value store lower 16 bits
     *  are the bloom filter
     *
     **/
    inline void finalize_directory(const std::vector<uint32_t> &final_offsets) {
        for (size_t i = 0; i < directory.size(); ++i) {
            uint64_t end_idx = final_offsets[i];
            uint64_t bloom = directory[i] & 0xFFFF;
            directory[i] = (end_idx << 16) | bloom;
        }
    }

  public:
    /* returns index range into keys/row_ids arrays for a given key */
    inline std::pair<uint64_t, uint64_t> find_indices(int32_t key) const {
        uint64_t h = hash_key(key);
        size_t slot = h >> shift;
        uint64_t entry = directory[slot];

        uint16_t filter = (uint16_t)entry;
        uint16_t key_mask = compute_bloom(h);

        if ((filter & key_mask) != key_mask) {
            return {0, 0};
        }

        uint64_t end_idx = entry >> 16;
        uint64_t start_idx = (slot == 0) ? 0 : (directory[slot - 1] >> 16);

        if (start_idx >= end_idx) {
            return {0, 0};
        }

        return {start_idx, end_idx};
    }

    inline const int32_t *get_keys() const { return keys.data(); }
    inline const uint32_t *get_row_ids() const { return row_ids.data(); }
};

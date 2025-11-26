#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <value_t_builders.h>
#include <vector>


class UnchainedHashtable {
public:
  struct Tuple {
    int32_t key;
    uint32_t row_id;
  };

private:
  std::vector<uint64_t> directory;
  std::vector<Tuple> value_store;
  size_t shift;

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
    value_store.reserve(build_size);
  }

  void build(const mema::column_t &column) {
    if (column.has_direct_access()) {
      build_dense(column);
    } else {
      build_sparse(column);
    }
  }

private:
  /* builds the hash table from a nun null column */
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

    value_store.resize(current_offset);

    /**
     *
     *  Hashes keys and stores the
     *  key index tuples within value
     *  store using the prefix sums
     *  created earlier
     *
     *  increments its prefix sum to store
     *  next tuple in the next slot by the
     *  end of the procedure counts will
     *  be brought to the original form
     *  before the prefix calculations
     *  were made.
     *
     *  also creates a bloom for that value
     *  and stores it in the directory
     *
     **/

    for (size_t i = 0; i < rows; ++i) {
      int32_t val = column[i].value;
      uint64_t h = hash_key(val);
      size_t slot = h >> shift;
      uint32_t pos = counts[slot]++;
      value_store[pos] = {val, (uint32_t)i};
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
     * this makes a temp buffer
     * so we dont call get_by_row
     * 3 times.
     *
     **/
    std::vector<Tuple> temp_input;
    temp_input.reserve(rows);
    for (size_t i = 0; i < rows; ++i) {
      const mema::value_t *val = column.get_by_row(i);
      if (val) {
        temp_input.push_back({val->value, (uint32_t)i});
      }
    }

    /* same proccess as build_dense */

    for (const auto &t : temp_input) {
      uint64_t h = hash_key(t.key);
      counts[h >> shift]++;
    }

    uint32_t current_offset = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
      uint32_t count = counts[i];
      counts[i] = current_offset;
      current_offset += count;
    }

    value_store.resize(current_offset);
    for (const auto &t : temp_input) {
      uint64_t h = hash_key(t.key);
      size_t slot = h >> shift;

      uint32_t pos = counts[slot]++;
      value_store[pos] = t;
      directory[slot] |= compute_bloom(h);
    }

    finalize_directory(counts);
  }

  /**
   *
   *  this functions packs blooms filter and value store index
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

  /**
   *
   * returns a span of entries
   * within the value_store
   * for easy iteration
   *
   * but in contrast to previous
   * implementations we have to validate
   * the output of those as they might be
   * refering to other keys as well
   *
   **/
  inline std::pair<const Tuple *, const Tuple *> find(int32_t key) const {
    uint64_t h = hash_key(key);
    size_t slot = h >> shift;
    uint64_t entry = directory[slot];

    /* checks if value is in filter */
    uint16_t filter = (uint16_t)entry;
    uint16_t key_mask = compute_bloom(h);

    if ((filter & key_mask) != key_mask) {
      return {nullptr, nullptr};
    }
    /* shifts bloom out and
     * indexes value_store */
    uint64_t end_idx = entry >> 16;
    uint64_t start_idx = (slot == 0) ? 0 : (directory[slot - 1] >> 16);

    if (start_idx >= end_idx) {
      return {nullptr, nullptr};
    }

    return {&value_store[start_idx], &value_store[end_idx]};
  }
};

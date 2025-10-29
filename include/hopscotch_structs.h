#ifndef HOPSCOTCH_STRUCTURES_HPP
#define HOPSCOTCH_STRUCTURES_HPP

#include <vector>
#include <cstdint>
#include <cstddef>

/**
 *
 *  The buckets contain the neighbourhood bitmask,
 *  the key that corresponds to the bucket and 
 *  indices to segments that  help locate values
 *  within the value_store vector found in the hash table.
 *
 *  Bucket alignment is very important and the
 *  specialized one will be chosen for the most
 *  usual case of uint32_t keys
 *
 *  All other structs are alinged as well.
 *
 *  Buckets also utilize an optimization
 *  whre the first value is stored inline
 *  via the last_segment field
 *
 **/
template <typename Key>
struct alignas(64) Bucket {
    uint64_t bitmask;
    Key key;
    uint32_t first_segment;
    uint32_t last_segment;
    uint16_t count;
    bool occupied;

    Bucket() : bitmask(0), first_segment(0), last_segment(UINT32_MAX),
               count(0), occupied(false){}
};

template<>
struct alignas(32) Bucket<int32_t> {
    uint64_t bitmask;
    int32_t key;
    uint32_t first_segment;
    uint32_t last_segment;
    uint16_t count;
    bool occupied;

    Bucket() : bitmask(0), first_segment(0), last_segment(UINT32_MAX),
               count(0), occupied(false){}
};

/**
 *
 *  Segments store an index to a contiguous
 *  range of same key values within the
 *  value store vector.
 *  
 *  They also store the position of the 
 *  next segment within their array basically
 *  acting like a linked list of indices
 *  within the value store vector.
 *
 **/
struct alignas(16) Segment {
    uint32_t start_index;
    uint16_t count;
    uint32_t next_segment;

    Segment() : start_index(0), count(0), next_segment(UINT32_MAX) {}
    Segment(uint32_t start, uint16_t cnt, uint32_t next = UINT32_MAX)
        : start_index(start), count(cnt), next_segment(next) {}
};


/**
 *
 *  Value span allows us to retrieve
 *  values associated with a key using
 *  iterators. This is the return type
 *  of find.
 *
 **/

template <typename Key>
struct alignas(32) ValueSpan {
    const std::vector<uint32_t>* value_store;
    const std::vector<Segment>* segments;
    uint32_t first_segment;
    uint16_t total_count;

    /**
     *  This struct handles the somewhat
     *  complex traversal of the linked
     *  segments to retrieve all relavant
     *  values.
     *
     *  A prefetching scheme has been used
     *  to try and offset the fragmentation
     *  of the data that will be accessed
     *  this provides a small but measurable
     *  performance impact.
     *
     *  Also handles inline value optimization
     *
     **/
    struct alignas(32) Iterator {
        const std::vector<uint32_t>* value_store;
        const std::vector<Segment>* segments;
        uint32_t current_segment;
        uint16_t offset;

        Iterator(const std::vector<uint32_t>* vs, const std::vector<Segment>* segs,
                uint32_t seg_index, uint16_t offset = 0)
            : value_store(vs), segments(segs), current_segment(seg_index),
              offset(offset) {}

        size_t operator*() const {
            /* inline value handling */
            if (value_store == nullptr) {
                return current_segment;
            }
            /* usual case */
            const Segment& seg = (*segments)[current_segment];
            size_t data_index = seg.start_index + offset;
            if (offset + 16 < seg.count) {
                __builtin_prefetch(&(*value_store)[data_index + 16], 0, 2);
            } else if (seg.next_segment != UINT32_MAX) {
                const Segment& next_seg = (*segments)[seg.next_segment];
                __builtin_prefetch(&(*value_store)[next_seg.start_index], 0, 2);
            }
            return (*value_store)[seg.start_index + offset];

        }

        Iterator& operator++() {
            /* inline value handling */
            if (value_store == nullptr) {
                current_segment = UINT32_MAX;
                return *this;
            }
            /* usual case */
            const Segment& seg = (*segments)[current_segment];
            offset++;
            if (offset >= seg.count) {
                current_segment = seg.next_segment;
                offset = 0;
            }
            return *this;
        }

        bool operator!=(const Iterator& other) const {
            return current_segment != other.current_segment ||
            offset != other.offset;
        }
    };

    Iterator begin() const {
        return Iterator(value_store, segments, first_segment, 0);
    }

    Iterator end() const {
        return Iterator(value_store, segments, UINT32_MAX, 0);
    }
    
    size_t size() const { return total_count; }
    bool empty() const { return total_count == 0; }
};

#endif

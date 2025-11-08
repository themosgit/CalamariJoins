#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>

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
 *  Templated method for all types of buckets
 *
 *  handles duplicate key insertion effeciently by using 
 *  pre stored last_segment field of Bucket
 *
 *  Enables O(1) because we can immediatly idenitfy
 *  if this bucket's last segment is at the vectors end.
 *
 *  The first branch checks if we had only one value inserted
 *  and does the appropriate operations for our optimization.
 *
 **/

template <typename BucketType>
inline void insert_duplicate(
    BucketType& bucket,
    uint32_t item,
    std::vector<uint32_t>& value_store,
    std::vector<Segment>& segments)
{
    if (bucket.count == 1) {
        /* migrate inline value to value_store */
        value_store.push_back(bucket.last_segment);
        value_store.push_back(item);
        
        uint32_t new_segment_index = segments.size();
        segments.emplace_back(value_store.size() - 2, 2, UINT32_MAX);
        
        bucket.first_segment = new_segment_index;
        bucket.last_segment = new_segment_index;
        bucket.count = 2;
        return;
    }

    if (bucket.last_segment != UINT32_MAX) {
        Segment& last_seg = segments[bucket.last_segment];
        if (last_seg.start_index + last_seg.count == value_store.size()) {
            value_store.push_back(item);
            last_seg.count++;
            bucket.count++;
            return;
        }
    }
    
    /* create new segment */
    value_store.push_back(item);
    uint32_t new_segment_index = segments.size();
    segments.emplace_back(value_store.size() - 1, 1, UINT32_MAX);

    if (bucket.last_segment != UINT32_MAX)
        segments[bucket.last_segment].next_segment = new_segment_index;
    else
        bucket.first_segment = new_segment_index;

    bucket.last_segment = new_segment_index;
    bucket.count++;
}


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
        Segment cached_segment;

        Iterator(const std::vector<uint32_t>* vs, const std::vector<Segment>* segs,
                uint32_t seg_index, uint16_t offset = 0)
            : value_store(vs), segments(segs), current_segment(seg_index),
              offset(offset) {
              if (seg_index != UINT32_MAX && segs != nullptr)
                  cached_segment = (*segs)[seg_index];
              }

        size_t operator*() const {
            /* inline value handling */
            if (value_store == nullptr) {
                return current_segment;
            }
            /* usual case */
            const Segment& seg = (*segments)[current_segment];
            size_t data_index = seg.start_index + offset;
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
                current_segment = cached_segment.next_segment;
                offset = 0;
                if (current_segment != UINT32_MAX) {
                    cached_segment = (*segments)[current_segment];
                    if (cached_segment.count > 0) {
                        __builtin_prefetch(&(*value_store)
                                [cached_segment.start_index], 0, 2);
                    }
                }
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


/**
 *
 *  Bloom filter implementation
 *
 **/
struct BloomFilter{
    /* bit array */
    std::vector<uint64_t> bits;
    /* size in bits */
    size_t m;
    size_t shift;
    size_t total_bits;

    BloomFilter(size_t n, const double p = 0.05) noexcept {
        if (n == 0) n = 1;
        /* formula for the size of the bit array */
        m = static_cast<size_t>(-(n* std::log(p)) / (std::log(2) * std::log(2)));
        /* bloom filter vector alocate and initialize */
        bits.resize((m + 63) / 64, 0);
        shift = 0;
        total_bits = bits.size() * 64;
        while((1ULL << shift) < total_bits) ++shift;
    }

    void insert(const size_t hash) noexcept {
        /* take key and shift right then mod bits */
        size_t idx = (hash >> shift) % total_bits;
        bits[idx / 64] |= (1ULL << (idx % 64));
    }

    bool contains(const size_t hash) const noexcept {
        size_t idx = (hash >> shift) % total_bits;
        /**
         *
         *  which idx in the bloom filter and which
         *  place in the idx we will "turn on"(like a switch)
         *
         **/
        return bits[idx / 64] & (1ULL << (idx % 64));
    }
};

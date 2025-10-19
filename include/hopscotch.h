#ifndef HOPSCOTCH_HASH_TABLE_HPP
#define HOPSCOTCH_HASH_TABLE_HPP

#include <cstdint>
#include <cstring>
#include <string>
#include <attribute.h>

union DataValue {
    int32_t i32;
    int64_t i64;
    double fp64;
    const char* varchar;
    
    DataValue() : i64(0) {}
};

struct HashKey {
    DataType type;
    DataValue value;
    
    HashKey() : type(DataType::INT32) {}
    HashKey(int32_t v) : type(DataType::INT32) { value.i32 = v; }
    HashKey(int64_t v) : type(DataType::INT64) { value.i64 = v; }
    HashKey(double v) : type(DataType::FP64) { value.fp64 = v; }
    HashKey(const char* v) : type(DataType::VARCHAR) { value.varchar = v; }
    
    template<typename K>
    HashKey(const K& v) {
        if constexpr (std::is_same_v<K, int32_t>) {
            type = DataType::INT32;
            value.i32 = v;
        } else if constexpr (std::is_same_v<K, int64_t>) {
            type = DataType::INT64;
            value.i64 = v;
        } else if constexpr (std::is_same_v<K, double>) {
            type = DataType::FP64;
            value.fp64 = v;
        } else if constexpr (std::is_same_v<K, float>) {
            type = DataType::FP64;
            value.fp64 = static_cast<double>(v);
        } else if constexpr (std::is_same_v<K, std::string>) {
            type = DataType::VARCHAR;
            value.varchar = v.c_str();
        } else if constexpr (std::is_same_v<K, const char*> || std::is_same_v<K, char*>) {
            type = DataType::VARCHAR;
            value.varchar = v;
        } else {
            static_assert(sizeof(K) == 0, "Unsupported key type for HashKey");
        }
    }
    
    bool operator==(const HashKey& other) const {
        if (type != other.type) return false;
        
        switch (type) {
            case DataType::INT32:
                return value.i32 == other.value.i32;
            case DataType::INT64:
                return value.i64 == other.value.i64;
            case DataType::FP64:
                return value.fp64 == other.value.fp64;
            case DataType::VARCHAR:
                return strcmp(value.varchar, other.value.varchar) == 0;
        }
        return false;
    }
};

class TypedHashFunction {
public:
    static uint64_t hash(const HashKey& key) {
        uint64_t h = 0;
        
        switch (key.type) {
            case DataType::INT32:
                h = hash_int32(key.value.i32);
                break;
            case DataType::INT64:
                h = hash_int64(key.value.i64);
                break;
            case DataType::FP64:
                h = hash_fp64(key.value.fp64);
                break;
            case DataType::VARCHAR:
                h = hash_varchar(key.value.varchar);
                break;
        }
        
        return h;
    }
    
private:
    static uint64_t hash_int32(int32_t key) {
        uint64_t h = static_cast<uint64_t>(key);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }
    
    static uint64_t hash_int64(int64_t key) {
        uint64_t h = static_cast<uint64_t>(key);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
        return h;
    }
    
    static uint64_t hash_fp64(double key) {
        uint64_t bits;
        std::memcpy(&bits, &key, sizeof(double));
        return hash_int64(bits);
    }
    
    static uint64_t hash_varchar(const char* str) {
        uint64_t hash = 0xcbf29ce484222325ULL;
        const uint64_t prime = 0x100000001b3ULL;
        
        while (*str) {
            hash ^= static_cast<uint64_t>(*str);
            hash *= prime;
            ++str;
        }
        
        return hash;
    }
};

template<typename V>
class HopscotchHashTable {
private:
    static constexpr size_t H = 64;
    
    struct Bucket {
        alignas(8) HashKey key;
        alignas(8) V value;
        alignas(8) uint64_t hop_info;
        alignas(8) bool occupied;
        
        Bucket() : hop_info(0), occupied(false) {}
    } __attribute__((aligned(64)));
    
    Bucket* table;
    size_t capacity;
    size_t size_count;
    
    // Find free bucket within max distance
    size_t find_free_bucket(size_t start_idx) const {
        for (size_t i = 0; i < capacity; ++i) {
            size_t idx = (start_idx + i) % capacity;
            if (!table[idx].occupied) {
                return idx;
            }
        }
        return capacity;
    }
    
    bool move_closer(size_t free_idx, size_t& new_free_idx) {
        size_t start = (free_idx >= H - 1) ? (free_idx - (H - 1)) : 0;
        
        for (size_t i = start; i < free_idx; ++i) {
            if (table[i].occupied) {
                uint64_t hop_info = table[i].hop_info;
                
                for (size_t j = 0; j < H && (i + j) < capacity; ++j) {
                    if ((hop_info & (1ULL << j)) && (i + j) < free_idx) {
                        size_t victim_idx = i + j;
                        
                        table[free_idx] = table[victim_idx];
                        table[victim_idx].occupied = false;
                        
                        table[i].hop_info &= ~(1ULL << j);
                        size_t new_offset = free_idx - i;
                        if (new_offset < H) {
                            table[i].hop_info |= (1ULL << new_offset);
                        }
                        
                        new_free_idx = victim_idx;
                        return true;
                    }
                }
            }
        }
        
        return false;
    }

public:
    explicit HopscotchHashTable(size_t cap) 
        : capacity(cap), size_count(0) {
        table = new Bucket[capacity];
    }
    
    ~HopscotchHashTable() {
        delete[] table;
    }
    V* find(const HashKey& key) {
        uint64_t hash = TypedHashFunction::hash(key);
        size_t home_idx = hash % capacity;
        
        uint64_t hop_info = table[home_idx].hop_info;
        for (size_t i = 0; i < H && (home_idx + i) < capacity; ++i) {
            if ((hop_info & (1ULL << i)) && table[home_idx + i].occupied) {
                if (table[home_idx + i].key == key) {
                    return &table[home_idx + i].value;
                }
            }
        }
        return nullptr;
    }
    
    bool insert(const HashKey& key, const V& value) {
        uint64_t hash = TypedHashFunction::hash(key);
        size_t home_idx = hash % capacity;
        
        uint64_t hop_info = table[home_idx].hop_info;
        for (size_t i = 0; i < H && (home_idx + i) < capacity; ++i) {
            if ((hop_info & (1ULL << i)) && table[home_idx + i].occupied) {
                if (table[home_idx + i].key == key) {
                    table[home_idx + i].value = value;
                    return true;
                }
            }
        }
        
        size_t free_idx = find_free_bucket(home_idx);
        if (free_idx == capacity) {
            return false;
        }
        
        while (free_idx >= home_idx + H) {
            size_t new_free_idx;
            if (!move_closer(free_idx, new_free_idx)) {
                return false;
            }
            free_idx = new_free_idx;
        }
        
        table[free_idx].key = key;
        table[free_idx].value = value;
        table[free_idx].occupied = true;
        
        size_t offset = free_idx - home_idx;
        table[home_idx].hop_info |= (1ULL << offset);
        
        ++size_count;
        return true;
    }
    
    bool lookup(const HashKey& key, V& value) const {
        uint64_t hash = TypedHashFunction::hash(key);
        size_t home_idx = hash % capacity;
        
        uint64_t hop_info = table[home_idx].hop_info;
        for (size_t i = 0; i < H && (home_idx + i) < capacity; ++i) {
            if ((hop_info & (1ULL << i)) && table[home_idx + i].occupied) {
                if (table[home_idx + i].key == key) {
                    value = table[home_idx + i].value;
                    return true;
                }
            }
        }
        
        return false;
    }
    
    bool remove(const HashKey& key) {
        uint64_t hash = TypedHashFunction::hash(key);
        size_t home_idx = hash % capacity;
        
        uint64_t hop_info = table[home_idx].hop_info;
        for (size_t i = 0; i < H && (home_idx + i) < capacity; ++i) {
            if ((hop_info & (1ULL << i)) && table[home_idx + i].occupied) {
                if (table[home_idx + i].key == key) {
                    table[home_idx + i].occupied = false;
                    table[home_idx].hop_info &= ~(1ULL << i);
                    --size_count;
                    return true;
                }
            }
        }
        return false;
    }
    
    size_t size() const { return size_count; }
    size_t get_capacity() const { return capacity; }
    bool empty() const { return size_count == 0; }
    double load_factor() const { return static_cast<double>(size_count) / capacity; }
};

#endif

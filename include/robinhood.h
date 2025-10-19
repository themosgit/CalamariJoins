#pragma once
#include <vector>
#include <optional>
#include <cstddef>
#include <functional>
#include <memory>

template <typename Key>

struct Entry
{
    Key key;
    size_t psl;
    std::unique_ptr<std::vector<size_t>> indices; //try
    Entry() : psl(0) , indices(nullptr){}
    Entry(Key k, size_t p, std::unique_ptr<std::vector<size_t>>idx)
        : key(k),psl(p), indices(std::move(idx)) {}
};

template <typename Key>
class RobinHoodTable
{
private:
    std::vector<std::optional<Entry<Key>>> table;
    size_t size;
    size_t num_swaps = 0; 
    size_t num_inserts = 0;
    size_t hash(const Key &key) const
    {
        size_t h =  std::hash<Key>{}(key);
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
    
    return h;
    }

public:
    RobinHoodTable(size_t s) : size(s), table(s) {}

    void insert(const Key &key, size_t idx)
    {
        size_t p = hash(key) & (size - 1);
        size_t vpsl = 0;
        Key k = key;
        auto v = std::make_unique<std::vector<size_t>>();
        v->push_back(idx);
        // check if the key already exists
        while (table[p].has_value())
        {
            if (table[p]->key == key)
            {
                table[p]->indices->push_back(idx);
                return;
            }
            // if entry has smaller psl swap
            if (vpsl > table[p]->psl)
            {
                num_swaps++;
                std::swap(k, table[p]->key);
                std::swap(v, table[p]->indices);
                std::swap(vpsl, table[p]->psl);
            }
            p = (p + 1) & (size - 1);
            vpsl++;
        }
        table[p] = Entry<Key>{k, vpsl, std::move(v)};
    }

    std::optional<std::vector<size_t> *> search(const Key &key)
    {
        size_t p = hash(key) & (size - 1);
        size_t vpsl = 0;
        while (table[p].has_value())
        {
            if (table[p]->key == key)
            {
                return table[p]->indices.get();
            }
            if (vpsl > table[p]->psl)
            {
                return std::nullopt;
            }
            p = (p + 1) & (size - 1);
            vpsl++;
        }
        return std::nullopt;
    }
};
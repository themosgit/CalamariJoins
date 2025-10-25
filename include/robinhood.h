#pragma once
#include <vector>
#include <optional>
#include <cstddef>
#include <functional>
#include <llvm/ADT/SmallVector.h>

/*
Robin hood hashing algorithm: the idea behind this hashing algorithm is
to keep the psl(probe sequence lenght) as low as possible.
Psl is the distance from key's ideal position to actual position.
on collision if new key has higher psl than existing key,they swap positions
this keeps the variance low and search fast


*/


template <typename Key>

struct Entry
{
    Key key;
    size_t psl;
    llvm::SmallVector<size_t,1>indices; //pointer in vector
    Entry() : psl(0) {}
    Entry(Key k, size_t p, llvm::SmallVector<size_t,1> idx) //constructor with key,psl +unique_ptr
        : key(k),psl(p), indices(std::move(idx)) {} //then initialize
};

template <typename Key>
class RobinHoodTable
{
private:
    std::vector<std::optional<Entry<Key>>> table; //allows empty slots without wasting memory
    size_t size;
    size_t hash(const Key &key) const
    {
        size_t h =  std::hash<Key>{}(key); //hash mixing with MurmurHash3 finalizer for better distribution
        h ^= h >> 33; //
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
        auto v = llvm::SmallVector<size_t,1>();//heap alloc
        v.push_back(idx);
        // check if the key already exists
        while (table[p].has_value())
        {
            if (table[p]->key == key)
            {
                table[p]->indices.push_back(idx);
                return;
            }
            // if entry has smaller psl swap
            if (vpsl > table[p]->psl)
            {
                std::swap(k, table[p]->key);
                std::swap(v, table[p]->indices);
                std::swap(vpsl, table[p]->psl);
            }
            p = (p + 1) & (size - 1);
            vpsl++;
        }
        table[p] = Entry<Key>{k, vpsl, std::move(v)};
    }

    std::optional<llvm::SmallVector<size_t,1> *> search(const Key &key)
    {
        size_t p = hash(key) & (size - 1);
        size_t vpsl = 0;
        while (table[p].has_value())
        {
            if (table[p]->key == key)
            {
                return &table[p]->indices;
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

#pragma once
#include <vector>
#include <optional>
#include <cstddef>
#include <functional>
#include <llvm/ADT/SmallVector.h>
#include <iostream>
#include <stdexcept>
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
    llvm::SmallVector<size_t,1> indices; //small vector to avoid heap alocation for 1 bucket(keep it in stack instead)
    Entry() : psl(0) {}
    Entry(Key k, size_t p, llvm::SmallVector<size_t,1> idx)
        : key(k),psl(p), indices(std::move(idx)) {} 
};
        

template <typename Key>
class RobinHoodTable
{
private:
    size_t size;
    std::vector<Entry<Key>> table;
    size_t hash(const Key &key) const{
        size_t h =  std::hash<Key>{}(key); //hash mixing with MurmurHash3 finalizer for better distribution
        h ^= h >> 33; //
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;
    return h;
    }

public:
    RobinHoodTable(size_t build_size){
        //keep size table to size of 2's for faster hash function
        build_size = build_size ? build_size : 1;
        size_t power_of_2 = 1;
        while(power_of_2 * 0.60 < build_size) power_of_2 <<= 1;
        size = power_of_2;
        table.resize(size);
    }
    

    const size_t capacity(void) {
        return size;
    }

    inline void prefetch(const Key& key) const noexcept { //prefetch to avoid cache misses
        size_t hash_val = hash(key);
        size_t base_index = hash_val & (size - 1);
        __builtin_prefetch(&table[base_index], 0, 3);
}


    void insert(const Key &key, size_t idx){

        size_t p = hash(key) & (size - 1);
        size_t vpsl = 0;
        Key k = key;
        auto v = llvm::SmallVector<size_t,1>();
        v.push_back(idx);
        // check if the key already exists
        while (!table[p].indices.empty()){
            __builtin_prefetch(&table[(p + 1) & (size - 1)], 0, 3);
            if (table[p].key == key){
                table[p].indices.push_back(idx); //push it to the end of the vec
                return;
            }
            // if entry has smaller psl swap
            if (vpsl > table[p].psl){
                std::swap(k, table[p].key);
                std::swap(v, table[p].indices);
                std::swap(vpsl, table[p].psl);
            }
            p = (p + 1) & (size - 1);
            vpsl++;
        }
        table[p] = Entry<Key>{k, vpsl, std::move(v)}; // if it's empty insert
    }

    llvm::SmallVector<size_t,1>* find(const Key &key){
        size_t p = hash(key) & (size - 1);
        size_t vpsl = 0;
        while (!table[p].indices.empty()){
            __builtin_prefetch(&table[(p + 1) & (size - 1)], 0, 3);
            if (table[p].key == key) return &table[p].indices; // found
            if (vpsl > table[p].psl) break; //if psl's bigger than the psl of the current idx means it doesn't exist so return

            p = (p + 1) & (size - 1);
            vpsl++;
        }
        return nullptr; //not found
    }
};
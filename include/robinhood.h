#pragma once
#include <vector>
#include <optional>
#include <cstddef>
#include <functional>

template<typename Key>

struct Entry{
    Key key;
    size_t psl;
    std::vector<size_t>indices;
};


template<typename Key>
class RobinHoodTable{
private:
    std::vector<std::optional<Entry<Key>>>table;
    size_t size;
    size_t hash(const Key& key) const{
        return std::hash<Key>{}(key);
    }
public:
    RobinHoodTable(size_t s) : size(s), table(s){}

    void insert(const Key& key, size_t idx){
        size_t p = hash(key) % size;
        size_t vpsl = 0;
        Key k = key;
        std::vector<size_t> v{idx};
        //check if the key already exists
        while(table[p].has_value()){
            if(table[p]->key == key){
                table[p]->indices.push_back(idx);
                return;
            }
            //if entry has smaller psl swap
            if(vpsl > table[p]->psl){
                std::swap(k,table[p]->key);
                std::swap(v,table[p]->indices);
                std::swap(vpsl,table[p]->psl);
            }
            p = (p+1) % size;
            vpsl++;
        }
        table[p] = Entry<Key>{k,vpsl,std::move(v)};
    }

    std::optional<std::vector<size_t>*>search(const Key& key){
        size_t p = hash(key) % size;
        size_t vpsl = 0;
        while(table[p].has_value()){
            if(table[p]->key == key){
                return &table[p]->indices;
            }
            if(vpsl > table[p]->psl){
                return std::nullopt;
            }
            p = (p + 1) % size;
            vpsl++;
        }
        return std::nullopt;
    }
};
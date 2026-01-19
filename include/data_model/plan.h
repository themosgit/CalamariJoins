/*
 * Copyright 2025 Matthias Boehm, TU Berlin
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file plan.h
 * @brief Execution plan (Plan, PlanNode) and columnar output (ColumnarTable).
 *
 * Plan: tree of JoinNode/ScanNode with output_attrs mapping columns to sources.
 * ColumnarTable: 8KB page-based columnar format for contest API output.
 *
 * @see intermediate.h for join intermediate format.
 */
#pragma once

#include <data_model/statement.h>
#include <foundation/attribute.h>

#if !defined(TEAMOPT_USE_DUCKDB) || defined(TEAMOPT_BUILD_CACHE)
#include <sys/mman.h>
#endif

/**
 * @brief RAII mmap wrapper with refcount. munmap on last ref release. Move-only.
 */
class MappedMemory {
  public:
    void *addr;
    size_t length;
    size_t refs;
    MappedMemory(void *addr, size_t length)
        : addr(addr), length(length), refs(0) {}

    MappedMemory(const MappedMemory &) = delete;
    MappedMemory &operator=(const MappedMemory &) = delete;

    MappedMemory(MappedMemory &&other) noexcept
        : addr(other.addr), length(other.length), refs(other.refs) {
        other.addr = nullptr;
        other.length = 0;
        other.refs = 0;
    }

    MappedMemory &operator=(MappedMemory &&other) noexcept {
        if (this != &other) {
            addr = other.addr;
            length = other.length;
            refs = other.refs;
            other.addr = nullptr;
            other.length = 0;
            other.refs = 0;
        }
        return *this;
    }

    ~MappedMemory() {
#if !defined(TEAMOPT_USE_DUCKDB) || defined(TEAMOPT_BUILD_CACHE)
        munmap(addr, length);
#endif
    }
};

/** @brief Plan node type discriminator. */
enum class NodeType {
    HashJoin, /**< Inner join node with two children. */
    Scan,     /**< Leaf node referencing a base table. */
};

/**
 * @brief Leaf node referencing base table. base_table_id indexes Plan::inputs.
 * Base tables must outlive execution (VARCHAR refs their pages).
 */
struct ScanNode {
    size_t base_table_id; /**< Index into Plan::inputs. */
};

/**
 * @brief Inner join node. Children indexed in Plan::nodes.
 * Equi-join on left_attr = right_attr. Keys always INT32 (contest invariant).
 */
struct JoinNode {
    /** Optimizer hint: if true, prefer left child as build side.
     *  Execution may override based on actual cardinalities. */
    bool build_left;
    size_t left;       /**< Index of left child in Plan::nodes. */
    size_t right;      /**< Index of right child in Plan::nodes. */
    size_t left_attr;  /**< Join key: index into left child's output_attrs. */
    size_t right_attr; /**< Join key: index into right child's output_attrs. */
};

/**
 * @brief Plan node (Scan or Join) with output_attrs column mapping.
 *
 * output_attrs: ScanNode → base table column index; JoinNode → combined L/R
 * index ([0,left_size) = left, [left_size,...) = right). May be remapped by
 * select_build_probe_side(). @see join_setup.h
 */
struct PlanNode {
    std::variant<ScanNode, JoinNode> data; /**< Node-specific data. */
    /** Output column mapping: (source_index, type) pairs. */
    std::vector<std::tuple<size_t, DataType>> output_attrs;

    PlanNode(std::variant<ScanNode, JoinNode> data,
             std::vector<std::tuple<size_t, DataType>> output_attrs)
        : data(std::move(data)), output_attrs(std::move(output_attrs)) {}
};

/**
 * @brief ColumnarTable page size (8KB). L2-cache sized; smaller than
 * IR_PAGE_SIZE due to null bitmap/offset overhead. @see intermediate.h
 */
constexpr size_t PAGE_SIZE = 8192;

/**
 * @brief 8-byte aligned page (8KB) for columnar data.
 *
 * INT32: [num_rows:u16][num_values:u16][values...][bitmap at end]
 * VARCHAR: [num_rows:u16][num_offsets:u16][offsets:u16...][string bytes][bitmap]
 * Long string markers: 0xFFFF (first), 0xFFFE (continuation).
 * Dense page (no NULLs): num_rows == num_values → fast path.
 */
struct alignas(8) Page {
    std::byte data[PAGE_SIZE];
};

/**
 * @brief Column as page sequence. Owned (new/delete) or shared via MappedMemory
 * refcount. Move-only.
 */
struct Column {
    DataType type;               /**< INT32 or VARCHAR. */
    std::vector<Page *> pages;   /**< Pointers to data pages. */
    MappedMemory *mapped_memory; /**< Shared memory region, or nullptr. */

    Page *new_page() {
        auto ret = new Page;
        pages.push_back(ret);
        return ret;
    }

    void assign_mapped_memory(MappedMemory *mapped_memory) {
        this->mapped_memory = mapped_memory;
        this->mapped_memory->refs++;
    }

    Column(DataType data_type)
        : type(data_type), pages(), mapped_memory(nullptr) {}

    Column(Column &&other) noexcept
        : type(other.type), pages(std::move(other.pages)),
          mapped_memory(other.mapped_memory) {
        other.pages.clear();
        other.mapped_memory = nullptr;
    }

    Column &operator=(Column &&other) noexcept {
        if (this != &other) {
            for (auto *page : pages) {
                delete page;
            }
            type = other.type;
            pages = std::move(other.pages);
            other.pages.clear();
            mapped_memory = other.mapped_memory;
            other.mapped_memory = nullptr;
        }
        return *this;
    }

    Column(const Column &) = delete;
    Column &operator=(const Column &) = delete;

    ~Column() {
        if (mapped_memory != nullptr) {
            if (--mapped_memory->refs == 0)
                delete mapped_memory;
            return;
        }
        for (auto *page : pages) {
            delete page;
        }
    }
};

/**
 * @brief Contest API output format: columnar, page-based.
 * Root join output via materialize(). @see intermediate.h for non-root format.
 */
struct ColumnarTable {
    size_t num_rows{0};          /**< Total row count across all pages. */
    std::vector<Column> columns; /**< One Column per output attribute. */
};

std::tuple<std::vector<std::vector<Data>>, std::vector<DataType>>
from_columnar(const ColumnarTable &table);
ColumnarTable from_table(const std::vector<std::vector<Data>> &table,
                         const std::vector<DataType> &data_types);

/**
 * @brief Multi-way join plan: flat vector of PlanNodes + base tables.
 * root indexes final result node. Base tables must outlive execute().
 */
struct Plan {
    std::vector<PlanNode>
        nodes; /**< All nodes; indices reference each other. */
    std::vector<ColumnarTable> inputs; /**< Pre-loaded base tables. */
    size_t root;                       /**< Index of root node in nodes. */

    /**
     * @brief Create JoinNode. @return node index. Execution may override build_left.
     */
    size_t
    new_join_node(bool build_left, size_t left, size_t right, size_t left_attr,
                  size_t right_attr,
                  std::vector<std::tuple<size_t, DataType>> output_attrs) {
        JoinNode join{
            .build_left = build_left,
            .left = left,
            .right = right,
            .left_attr = left_attr,
            .right_attr = right_attr,
        };
        auto ret = nodes.size();
        nodes.emplace_back(join, std::move(output_attrs));
        return ret;
    }

    /** @brief Create ScanNode (leaf). @return node index. */
    size_t
    new_scan_node(size_t base_table_id,
                  std::vector<std::tuple<size_t, DataType>> output_attrs) {
        ScanNode scan{.base_table_id = base_table_id};
        auto ret = nodes.size();
        nodes.emplace_back(scan, std::move(output_attrs));
        return ret;
    }

    /** @brief Add base table (moved). @return index for ScanNode. */
    size_t new_input(ColumnarTable input) {
        auto ret = inputs.size();
        inputs.emplace_back(std::move(input));
        return ret;
    }
};

/**
 * @brief Incremental column builder: page alloc, insert, bitmap, auto-finalize.
 *
 * INT32: [num_rows:u16][num_values:u16][values...][bitmap at end].
 * Overflow detection finalizes current page and allocates next.
 *
 * @tparam T int32_t or std::string. @see page_builders.h for optimized variant.
 */
template <class T> struct ColumnInserter {
    Column &column;
    size_t last_page_idx = 0;
    uint16_t num_rows = 0;
    size_t data_end = data_begin();
    std::vector<uint8_t> bitmap;

    constexpr static size_t data_begin() {
        if (sizeof(T) < 4) {
            return 4;
        } else {
            return sizeof(T);
        }
    }

    ColumnInserter(Column &column) : column(column) {
        bitmap.resize(PAGE_SIZE);
    }

    /** @brief Get current page, allocating if needed. Does not advance index. */
    std::byte *get_page() {
        if (last_page_idx == column.pages.size()) [[unlikely]] {
            column.new_page();
        }
        auto *page = column.pages[last_page_idx];
        return page->data;
    }

    /** @brief Finalize page (write header, bitmap) and advance to next. */
    void save_page() {
        auto *page = get_page();
        *reinterpret_cast<uint16_t *>(page) = num_rows;
        *reinterpret_cast<uint16_t *>(page + 2) =
            static_cast<uint16_t>((data_end - data_begin()) / sizeof(T));
        size_t bitmap_size = (num_rows + 7) / 8;
        memcpy(page + PAGE_SIZE - bitmap_size, bitmap.data(), bitmap_size);
        ++last_page_idx;
        num_rows = 0;
        data_end = data_begin();
    }

    void set_bitmap(size_t idx) {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        bitmap[byte_idx] |= (0x1 << bit_idx);
    }

    void unset_bitmap(size_t idx) {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        bitmap[byte_idx] &= ~(0x1 << bit_idx);
    }

    /** @brief Insert non-NULL value; auto-finalizes on overflow. */
    void insert(T value) {
        if (data_end + 4 + num_rows / 8 + 1 > PAGE_SIZE) [[unlikely]] {
            save_page();
        }
        auto *page = get_page();
        *reinterpret_cast<T *>(page + data_end) = value;
        data_end += sizeof(T);
        set_bitmap(num_rows);
        ++num_rows;
    }

    /** @brief Insert NULL (bitmap only, no data). */
    void insert_null() {
        if (data_end + num_rows / 8 + 1 > PAGE_SIZE) [[unlikely]] {
            save_page();
        }
        unset_bitmap(num_rows);
        ++num_rows;
    }

    /** @brief Flush partial page. Call after all inserts. */
    void finalize() {
        if (num_rows != 0) {
            save_page();
        }
    }
};

/**
 * @brief VARCHAR ColumnInserter: offset array + backward string writing.
 *
 * Format: [num_rows:u16][num_offsets:u16][offsets:u16...][strings][bitmap].
 * Long strings (>PAGE_SIZE-7): 0xFFFF (first), 0xFFFE (continuation).
 *
 * @see page_builders.h VarcharPageBuilder, intermediate.h value_t encoding.
 */
template <> struct ColumnInserter<std::string> {
    Column &column;
    size_t last_page_idx = 0;
    uint16_t num_rows = 0;
    uint16_t data_size = 0;
    size_t offset_end = 4;
    std::vector<char> data;
    std::vector<uint8_t> bitmap;

    constexpr static size_t offset_begin() { return 4; }

    ColumnInserter(Column &column) : column(column) {
        data.resize(PAGE_SIZE);
        bitmap.resize(PAGE_SIZE);
    }

    /** @brief Get current page, allocating if needed. Does not advance index. */
    std::byte *get_page() {
        if (last_page_idx == column.pages.size()) [[unlikely]] {
            column.new_page();
        }
        auto *page = column.pages[last_page_idx];
        return page->data;
    }

    /** @brief Write long string (>PAGE_SIZE-7) across pages. 0xFFFF/0xFFFE markers. */
    void save_long_string(std::string_view value) {
        size_t offset = 0;
        auto first_page = true;
        while (offset < value.size()) {
            auto *page = get_page();
            if (first_page) {
                *reinterpret_cast<uint16_t *>(page) = 0xffff;
                first_page = false;
            } else {
                *reinterpret_cast<uint16_t *>(page) = 0xfffe;
            }
            auto page_data_len = std::min(value.size() - offset, PAGE_SIZE - 4);
            *reinterpret_cast<uint16_t *>(page + 2) = page_data_len;
            memcpy(page + 4, value.data() + offset, page_data_len);
            offset += page_data_len;
            ++last_page_idx;
        }
    }

    /** @brief Finalize VARCHAR page and advance. */
    void save_page() {
        auto *page = get_page();
        *reinterpret_cast<uint16_t *>(page) = num_rows;
        *reinterpret_cast<uint16_t *>(page + 2) =
            static_cast<uint16_t>((offset_end - offset_begin()) / 2);
        size_t bitmap_size = (num_rows + 7) / 8;
        memcpy(page + offset_end, data.data(), data_size);
        memcpy(page + PAGE_SIZE - bitmap_size, bitmap.data(), bitmap_size);
        ++last_page_idx;
        num_rows = 0;
        data_size = 0;
        offset_end = offset_begin();
    }

    /** @brief Mark row valid in bitmap. */
    void set_bitmap(size_t idx) {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        bitmap[byte_idx] |= (0x1 << bit_idx);
    }

    /** @brief Mark row NULL in bitmap. */
    void unset_bitmap(size_t idx) {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        bitmap[byte_idx] &= ~(0x1 << bit_idx);
    }

    /** @brief Insert string. Long strings use save_long_string(). */
    void insert(std::string_view value) {
        if (value.size() > PAGE_SIZE - 7) {
            if (num_rows > 0) {
                save_page();
            }
            save_long_string(value);
        } else {
            if (offset_end + sizeof(uint16_t) + data_size + value.size() +
                    num_rows / 8 + 1 >
                PAGE_SIZE) {
                save_page();
            }
            memcpy(data.data() + data_size, value.data(), value.size());
            data_size += static_cast<uint16_t>(value.size());
            auto *page = get_page();
            *reinterpret_cast<uint16_t *>(page + offset_end) = data_size;
            offset_end += sizeof(uint16_t);
            set_bitmap(num_rows);
            ++num_rows;
        }
    }

    /** @brief Insert NULL (offset grows for row alignment). */
    void insert_null() {
        if (offset_end + data_size + num_rows / 8 + 1 > PAGE_SIZE)
            [[unlikely]] {
            save_page();
        }
        unset_bitmap(num_rows);
        ++num_rows;
    }

    /** @brief Flush partial page. Call after all inserts. */
    void finalize() {
        if (num_rows != 0) {
            save_page();
        }
    }
};

/**
 * @namespace Contest
 * @brief Contest API: build_context/destroy_context, execute(), TimingStats.
 * @see Plan, intermediate.h
 */
namespace Contest {

/** @brief Execution phase timing breakdown (ms resolution). */
struct TimingStats {
    int64_t hashtable_build_ms = 0;  /**< Hash table construction time. */
    int64_t hash_join_probe_ms = 0;  /**< Parallel probe phase duration. */
    int64_t nested_loop_join_ms = 0; /**< Nested loop join (for tiny tables). */
    int64_t materialize_ms = 0;      /**< Final ColumnarTable construction. */
    int64_t setup_ms = 0;            /**< JoinSetup + build/probe selection. */
    int64_t total_execution_ms = 0;  /**< Wall-clock total for execute(). */
    int64_t intermediate_ms = 0; /**< construct_intermediate for non-root. */
};

/** @brief Allocate execution context (worker pool, shared state). */
void *build_context();

/** @brief Release execution context resources. */
void destroy_context(void *);

/**
 * @brief Execute plan depth-first, return ColumnarTable.
 * @see execute.cpp for implementation.
 */
ColumnarTable execute(const Plan &plan, void *context,
                      TimingStats *stats_out = nullptr,
                      bool show_detailed_timing = false);

} // namespace Contest

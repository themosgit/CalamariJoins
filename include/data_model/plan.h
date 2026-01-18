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
 * @brief Execution plan structures and columnar output format.
 *
 * Defines the Plan structure built by the SQL parser, containing a tree of
 * join/scan nodes. Each node has output_attrs mapping output column indices
 * to source columns with their types.
 *
 * Also defines ColumnarTable, the required output format for the contest API,
 * using page-based columnar storage with null bitmaps.
 *
 * @see intermediate.h for the intermediate result format used between joins.
 */
#pragma once

#include <data_model/statement.h>
#include <foundation/attribute.h>

#if !defined(TEAMOPT_USE_DUCKDB) || defined(TEAMOPT_BUILD_CACHE)
#include <sys/mman.h>
#endif

/**
 * @brief RAII wrapper for mmap'd memory with reference counting.
 *
 * Multiple Column objects can share the same mapped region. The refs counter
 * tracks active users; munmap is called when the last reference is released.
 * Used by both ColumnarTable output and intermediate results from
 * BatchAllocator.
 *
 * @note Move-only type to prevent accidental double-free of mapped regions.
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

/**
 * @brief Discriminator for plan node types.
 */
enum class NodeType {
    HashJoin, /**< Inner join node with two children. */
    Scan,     /**< Leaf node referencing a base table. */
};

/**
 * @brief Leaf node that references a pre-loaded base table.
 *
 * The base_table_id indexes into Plan::inputs. ScanNodes return non-owning
 * pointers to ColumnarTable during execution; the base table must remain
 * valid until final materialization since intermediate VARCHAR values
 * reference its pages.
 */
struct ScanNode {
    size_t base_table_id; /**< Index into Plan::inputs. */
};

/**
 * @brief Inner join node with two children and equi-join condition.
 *
 * Children are referenced by index into Plan::nodes. The join condition
 * is an equality predicate between left_attr and right_attr columns.
 *
 * @note Join keys are always INT32 (contest invariant). VARCHAR columns
 *       are never used as join keys.
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
 * @brief A node in the execution plan tree (either Scan or Join).
 *
 * Each node specifies which columns appear in its output via output_attrs.
 * This mapping tracks column provenance through the join tree.
 *
 * **output_attrs semantics:**
 * - For ScanNode: index refers to column in the base ColumnarTable
 * - For JoinNode: index refers to combined left+right child output
 *   - Indices [0, left_output_size) map to left child's output columns
 *   - Indices [left_output_size, ...) map to right child's output columns
 *
 * During execution, select_build_probe_side() may remap these indices
 * based on which child becomes the build vs probe side.
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
 * @brief Page size for ColumnarTable output format (8KB).
 *
 * Chosen for cache efficiency: fits in L2 cache on most processors.
 * Smaller than intermediate format's 16KB pages since final output
 * requires more metadata (explicit null bitmaps, offset arrays).
 *
 * @see intermediate.h IR_PAGE_SIZE for larger intermediate pages.
 */
constexpr size_t PAGE_SIZE = 8192;

/**
 * @brief Fixed-size memory block for columnar data storage.
 *
 * Pages are 8-byte aligned for efficient access to 64-bit values.
 * Page layout depends on column type:
 *
 * **INT32 page format:**
 * - Bytes 0-1: num_rows (u16) - total rows including NULLs
 * - Bytes 2-3: num_values (u16) - count of non-NULL values
 * - Bytes 4+: packed INT32 values
 * - End-N: validity bitmap (1=valid, 0=NULL)
 *
 * **VARCHAR page format:**
 * - Bytes 0-1: num_rows (u16) or special marker (0xFFFF/0xFFFE for long
 * strings)
 * - Bytes 2-3: num_offsets (u16)
 * - Bytes 4+: cumulative end offsets (u16 each)
 * - After offsets: packed string bytes
 * - End-N: validity bitmap
 *
 * If num_rows == num_values, page is "dense" (no NULLs) enabling fast paths.
 */
struct alignas(8) Page {
    std::byte data[PAGE_SIZE];
};

/**
 * @brief A single column in a ColumnarTable, stored as a sequence of pages.
 *
 * Columns can either own their pages individually (new/delete) or share
 * pages from a MappedMemory region (arena allocation). The mapped_memory
 * pointer determines cleanup behavior: if set, pages are freed via the
 * shared region's reference counting.
 *
 * @note Move-only type. Copy is deleted to prevent accidental page duplication.
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
 * @brief Columnar table format required by the contest API.
 *
 * Stores data column-by-column in pages for cache-efficient access.
 * This is the final output format produced by materialize() for root joins.
 *
 * @see intermediate.h for column_t, the intermediate format used between
 *      non-root joins (uses larger pages and value_t encoding).
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
 * @brief Complete execution plan for a multi-way join query.
 *
 * Contains a tree of PlanNodes (stored flat in a vector) and pre-loaded
 * base tables. The root field identifies which node produces the final result.
 *
 * **Lifetime:** Base tables in inputs must remain valid until execute()
 * completes, since intermediate VARCHAR values encode references to their
 * pages.
 */
struct Plan {
    std::vector<PlanNode>
        nodes; /**< All nodes; indices reference each other. */
    std::vector<ColumnarTable> inputs; /**< Pre-loaded base tables. */
    size_t root;                       /**< Index of root node in nodes. */

    /**
     * @brief Construct a new JoinNode in the plan tree.
     *
     * @param build_left   Optimizer hint: prefer left child as build side.
     * @param left         Index of left child in Plan::nodes.
     * @param right        Index of right child in Plan::nodes.
     * @param left_attr    Join key: index into left child's output_attrs.
     * @param right_attr   Join key: index into right child's output_attrs.
     * @param output_attrs Projection: (combined_index, type) pairs where
     *                     indices [0, left_size) refer to left child output,
     *                     indices [left_size, ...) refer to right child output.
     * @return Index of the newly created node in Plan::nodes.
     *
     * @note Execution may override build_left based on actual cardinalities.
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

    /**
     * @brief Construct a new ScanNode (leaf) in the plan tree.
     *
     * @param base_table_id Index into Plan::inputs identifying the base table.
     * @param output_attrs  Projection: (column_index, type) pairs where
     *                      column_index refers to columns in the base table.
     * @return Index of the newly created node in Plan::nodes.
     *
     * @note ScanNodes return non-owning pointers to ColumnarTable during
     *       execution; base tables must remain valid until final
     * materialization.
     */
    size_t
    new_scan_node(size_t base_table_id,
                  std::vector<std::tuple<size_t, DataType>> output_attrs) {
        ScanNode scan{.base_table_id = base_table_id};
        auto ret = nodes.size();
        nodes.emplace_back(scan, std::move(output_attrs));
        return ret;
    }

    /**
     * @brief Add a pre-loaded base table to the plan.
     *
     * @param input ColumnarTable to add (ownership transferred via move).
     * @return Index of the table in Plan::inputs, used by ScanNode.
     *
     * @note Base tables must remain valid until execute() completes since
     *       intermediate VARCHAR values encode references to their pages.
     */
    size_t new_input(ColumnarTable input) {
        auto ret = inputs.size();
        inputs.emplace_back(std::move(input));
        return ret;
    }
};

/**
 * @brief Helper for building ColumnarTable columns incrementally.
 *
 * Handles page allocation, value insertion, null bitmap management, and
 * automatic page finalization when capacity is reached. Template specialization
 * on value type enables type-specific optimizations.
 *
 * **INT32 page format (base template):**
 * - [0-1]: num_rows (u16) - total rows including NULLs
 * - [2-3]: num_values (u16) - count of non-NULL values
 * - [4+]: packed INT32 values (4 bytes each)
 * - [end-N]: validity bitmap (1=valid, 0=NULL)
 *
 * **Overflow detection:** Checks if data_end + sizeof(value) + bitmap_size
 * exceeds PAGE_SIZE before each insert. When overflow detected, finalizes
 * current page and allocates next.
 *
 * **Bitmap management:** Uses set_bitmap() to mark valid rows, unset_bitmap()
 * for NULLs. Bitmap written to page end during save_page().
 *
 * @tparam T Value type: int32_t for INT32 columns, std::string for VARCHAR.
 *
 * @see VarcharPageBuilder in page_builders.h for optimized materialization.
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

    /**
     * @brief Get pointer to current working page, allocating if needed.
     *
     * @return Pointer to page data buffer.
     *
     * @note Does not advance last_page_idx; call save_page() to commit.
     */
    std::byte *get_page() {
        if (last_page_idx == column.pages.size()) [[unlikely]] {
            column.new_page();
        }
        auto *page = column.pages[last_page_idx];
        return page->data;
    }

    /**
     * @brief Finalize current page and advance to next.
     *
     * Writes page header (num_rows, num_values), copies bitmap to page end,
     * then resets builder state for next page. Appends finalized page to
     * column.
     *
     * @note Bitmap stored at page end to avoid memmove when data grows forward.
     */
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

    /**
     * @brief Insert a non-NULL value into the column.
     *
     * Checks for page overflow before insertion; finalizes current page
     * if insufficient space. Writes value, updates bitmap, increments row
     * count.
     *
     * @param value The value to insert.
     */
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

    /**
     * @brief Insert a NULL value into the column.
     *
     * Checks for page overflow (bitmap space only, no data space needed).
     * Marks row as NULL in bitmap and increments row count.
     */
    void insert_null() {
        if (data_end + num_rows / 8 + 1 > PAGE_SIZE) [[unlikely]] {
            save_page();
        }
        unset_bitmap(num_rows);
        ++num_rows;
    }

    /**
     * @brief Complete column construction, flushing any partial page.
     *
     * Must be called after all inserts to ensure final page is written
     * to the column. No-op if current page is empty.
     */
    void finalize() {
        if (num_rows != 0) {
            save_page();
        }
    }
};

/**
 * @brief Specialization of ColumnInserter for VARCHAR columns.
 *
 * Manages variable-length strings with offset arrays for indexed access.
 * Uses backward-writing strategy to avoid memmove when appending strings.
 *
 * **VARCHAR page format:**
 * - [0-1]: num_rows (u16) - or 0xFFFF/0xFFFE for long strings
 * - [2-3]: num_offsets (u16) - count of offset entries
 * - [4+]: cumulative end offsets (u16 each)
 * - After offsets: packed string bytes (written backward from end)
 * - [end-N]: validity bitmap
 *
 * **Long string handling:** Strings exceeding (PAGE_SIZE - 7) bytes span
 * multiple pages with markers:
 * - First page: num_rows = 0xFFFF (65535)
 * - Continuation pages: num_rows = 0xFFFE (65534)
 * - These markers distinguish long strings from normal multi-row pages.
 *
 * **Backward writing:** Offsets written forward from byte 4, strings written
 * backward from page end (minus bitmap space). Avoids costly memmove operations
 * as more strings are added.
 *
 * @see VarcharPageBuilder in page_builders.h for optimized materialization.
 * @see intermediate.h for value_t encoding used during joins.
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

    /**
     * @brief Get pointer to current working page, allocating if needed.
     *
     * @return Pointer to page data buffer.
     *
     * @note Does not advance last_page_idx; call save_page() to commit.
     */
    std::byte *get_page() {
        if (last_page_idx == column.pages.size()) [[unlikely]] {
            column.new_page();
        }
        auto *page = column.pages[last_page_idx];
        return page->data;
    }

    /**
     * @brief Write a long string (> PAGE_SIZE - 7 bytes) across multiple pages.
     *
     * Splits string into chunks, marking first page with 0xFFFF and
     * continuation pages with 0xFFFE. Each page stores chunk length in
     * num_offsets field and data starting at offset 4.
     *
     * @param value The string to write (must exceed single-page capacity).
     *
     * @note Long strings bypass normal offset array and bitmap; entire page
     *       (except 4-byte header) stores string data.
     */
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

    /**
     * @brief Finalize current VARCHAR page and advance to next.
     *
     * Writes header (num_rows, num_offsets), copies offset array (forward from
     * byte 4), string data (from staging buffer), and bitmap (backward from
     * end). Resets builder state for next page.
     */
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

    /**
     * @brief Mark row as valid (non-NULL) in bitmap.
     *
     * @param idx Row index within current page (0-based).
     */
    void set_bitmap(size_t idx) {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        bitmap[byte_idx] |= (0x1 << bit_idx);
    }

    /**
     * @brief Mark row as NULL in bitmap.
     *
     * @param idx Row index within current page (0-based).
     */
    void unset_bitmap(size_t idx) {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        bitmap[byte_idx] &= ~(0x1 << bit_idx);
    }

    /**
     * @brief Insert a non-NULL string into the column.
     *
     * For long strings (> PAGE_SIZE - 7), flushes current page and delegates
     * to save_long_string(). For normal strings, checks overflow considering
     * offset array, string data, and bitmap space. Appends cumulative offset
     * to page and copies string to staging buffer.
     *
     * @param value The string to insert.
     */
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

    /**
     * @brief Insert a NULL value into the VARCHAR column.
     *
     * Checks for page overflow (no string data, just offset and bitmap space).
     * Marks row as NULL in bitmap and increments row count. Offset array still
     * grows to maintain 1:1 correspondence with row indices.
     */
    void insert_null() {
        if (offset_end + data_size + num_rows / 8 + 1 > PAGE_SIZE)
            [[unlikely]] {
            save_page();
        }
        unset_bitmap(num_rows);
        ++num_rows;
    }

    /**
     * @brief Complete column construction, flushing any partial page.
     *
     * Must be called after all inserts to ensure final page is written
     * to the column. No-op if current page is empty.
     */
    void finalize() {
        if (num_rows != 0) {
            save_page();
        }
    }
};

/**
 * @namespace Contest
 * @brief Contest execution API and timing instrumentation.
 *
 * Provides the main execution interface (execute()), context management
 * (build_context/destroy_context), and performance timing (TimingStats).
 * This is the primary public API for the join execution engine.
 *
 * Key functions:
 * - **build_context()**: Allocate worker pool and shared state.
 * - **execute()**: Run a query plan and return columnar results.
 * - **destroy_context()**: Release resources.
 *
 * @see Plan for query plan structure.
 * @see mema::column_t for intermediate result format.
 */
namespace Contest {

/**
 * @brief Performance timing breakdown for query execution.
 *
 * Captures millisecond-resolution timings for each execution phase.
 * Used for performance analysis and benchmark comparisons.
 */
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
 * @brief Execute a multi-way join query plan.
 *
 * Traverses the plan tree depth-first, executing joins recursively.
 * Returns the final result as a ColumnarTable.
 *
 * @param plan      Query plan with nodes and pre-loaded base tables.
 * @param context   Execution context from build_context().
 * @param stats_out Optional output for timing breakdown.
 * @param show_detailed_timing If true, print timing to stderr.
 * @return Final join result in columnar format.
 *
 * @see execute.cpp for implementation details.
 */
ColumnarTable execute(const Plan &plan, void *context,
                      TimingStats *stats_out = nullptr,
                      bool show_detailed_timing = false);

} // namespace Contest

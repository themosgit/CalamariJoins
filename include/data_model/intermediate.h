/**
 * @file intermediate.h
 * @brief Intermediate join result types and input abstraction.
 *
 * Provides:
 * - mema::value_t: 4-byte value encoding (INT32 direct, VARCHAR as page/offset)
 * - mema::column_t: 16KB-paged column for materialized values
 * - mema::DeferredTable: 16KB-paged 32-bit row ID storage per base table
 * - IntermediateResult: Lightweight result with selective materialization
 * - JoinInput: Unified abstraction over columnar tables and intermediate
 * results
 *
 * Base tables must outlive execution.
 *
 * @see plan.h for ColumnarTable, construct_intermediate.h for building results.
 * @see deferred_plan.h for AnalyzedJoinNode with column decisions.
 */
#pragma once

#include <cstdint>
#include <data_access/table.h>
#include <data_model/deferred_plan.h>
#include <data_model/plan.h>
#include <foundation/common.h>
#include <optional>
#include <platform/arena.h>
#include <variant>
#include <vector>

/**
 * @namespace mema
 * @brief Compact join intermediate: value_t (4B) + column_t (16KB pages) +
 * DeferredTable (32-bit row IDs) + key_row_column_t (8B tuples).
 *
 * value_t: INT32 direct or VARCHAR page/offset ref. column_t: arena-allocated
 * pages with write_at(). DeferredTable: 32-bit row ID storage per base table.
 * key_row_column_t: (key, row_id) tuples for join key propagation.
 *
 * @see Contest::IntermediateResult, plan.h ColumnarTable.
 */
namespace mema {

/**
 * @brief Join key with associated row ID for tuple-based storage.
 *
 * For LEFT_ONLY/RIGHT_ONLY modes: row_id is base table row ID (zero
 * indirection) For BOTH mode: row_id may be IR index (requires deferred table
 * lookup)
 *
 * 8-byte aligned for efficient memory access and potential SIMD operations.
 */
struct alignas(8) KeyRowPair {
    int32_t key;     ///< Join key value
    uint32_t row_id; ///< Row ID (base table or IR index depending on mode)
};

/**
 * @brief Column of (key, row_id) tuples for join key storage.
 *
 * Enables accelerated hashtable build (tuples match internal format) and
 * zero-indirection row ID propagation through join chains. Used instead of
 * separate column_t for join key columns.
 *
 * Memory layout: 16KB pages containing 2048 KeyRowPair entries each.
 */
struct key_row_column_t {
    static constexpr size_t PAGE_SIZE = 1 << 14; // 16KB
    static constexpr size_t PAIRS_PER_PAGE =
        PAGE_SIZE / sizeof(KeyRowPair);       // 2048
    static constexpr size_t ENTRY_SHIFT = 11; // log2(2048)
    static constexpr size_t ENTRY_MASK = PAIRS_PER_PAGE - 1;

    struct alignas(PAGE_SIZE) Page {
        KeyRowPair data[PAIRS_PER_PAGE];
    };

    std::vector<Page *> pages;
    size_t num_values = 0;

    /// Base table ID for row_id component (valid when stores_base_row_ids=true)
    uint8_t base_table_id = 0;

    /// Source column in base table (for VARCHAR provenance)
    uint8_t source_column = 0;

    /// True if row_id contains base table row IDs, false if IR indices
    bool stores_base_row_ids = false;

    key_row_column_t() = default;

    key_row_column_t(key_row_column_t &&other) noexcept
        : pages(std::move(other.pages)), num_values(other.num_values),
          base_table_id(other.base_table_id),
          source_column(other.source_column),
          stores_base_row_ids(other.stores_base_row_ids) {
        other.pages.clear();
        other.num_values = 0;
    }

    key_row_column_t &operator=(key_row_column_t &&other) noexcept {
        if (this != &other) {
            pages = std::move(other.pages);
            num_values = other.num_values;
            base_table_id = other.base_table_id;
            source_column = other.source_column;
            stores_base_row_ids = other.stores_base_row_ids;
            other.pages.clear();
            other.num_values = 0;
        }
        return *this;
    }

    key_row_column_t(const key_row_column_t &) = delete;
    key_row_column_t &operator=(const key_row_column_t &) = delete;

    ~key_row_column_t() = default;

    /// O(1) read: idx >> 11 for page, idx & 0x7FF for offset
    inline KeyRowPair operator[](size_t idx) const {
        return pages[idx >> ENTRY_SHIFT]->data[idx & ENTRY_MASK];
    }

    /// Thread-safe write at idx (requires pages set up first)
    inline void write_at(size_t idx, KeyRowPair pair) {
        pages[idx >> ENTRY_SHIFT]->data[idx & ENTRY_MASK] = pair;
    }

    /// Read only the key at index
    inline int32_t key_at(size_t idx) const {
        return pages[idx >> ENTRY_SHIFT]->data[idx & ENTRY_MASK].key;
    }

    /// Read only the row_id at index
    inline uint32_t row_id_at(size_t idx) const {
        return pages[idx >> ENTRY_SHIFT]->data[idx & ENTRY_MASK].row_id;
    }

    size_t row_count() const { return num_values; }
    void set_row_count(size_t count) { num_values = count; }

    /// Pre-allocate pages from arena
    inline void pre_allocate_from_arena(Contest::platform::ThreadArena &arena,
                                        size_t count) {
        size_t pages_needed = (count + PAIRS_PER_PAGE - 1) / PAIRS_PER_PAGE;
        pages.reserve(pages_needed);
        for (size_t i = 0; i < pages_needed; ++i) {
            void *ptr =
                arena.alloc_chunk<Contest::platform::ChunkType::IR_PAGE>();
            pages.push_back(reinterpret_cast<Page *>(ptr));
        }
        num_values = count;
    }
};

/**
 * @brief 4-byte value: INT32 direct, VARCHAR packed (19-bit page + 13-bit
 * offset).
 *
 * NULL = INT32_MIN, long string offset = 0x1FFF. Refs valid only while
 * source exists.
 */
struct alignas(4) value_t {
    int32_t value;

    /** @brief Encode VARCHAR as page/offset. */
    static inline value_t encode_string(int32_t page_idx, int32_t offset_idx) {
        return {(offset_idx << 19) | (page_idx & 0x7FFFF)};
    }

    /** @brief Decode VARCHAR to page/offset. */
    static inline void decode_string(int32_t encoded, int32_t &page_idx,
                                     int32_t &offset_idx) {

        page_idx = encoded & 0x7FFFF;
        offset_idx = (static_cast<uint32_t>(encoded) >> 19) & 0x1FFF;
    }

    /** @brief Sentinel for long strings. */
    static constexpr int32_t LONG_STRING_OFFSET = 0x1FFF;

    /** @brief NULL sentinel for both types. */
    static constexpr int32_t NULL_VALUE = INT32_MIN;

    /** @brief Check if this value represents NULL. */
    inline bool is_null() const { return value == NULL_VALUE; }
};

/**
 * @brief Page size for intermediate results (16KB, larger than ColumnarTable).
 */
constexpr size_t IR_PAGE_SIZE = 1 << 14;

/** @brief Number of value_t entries per intermediate page. */
constexpr size_t CAP_PER_PAGE = IR_PAGE_SIZE / sizeof(value_t);

/**
 * @brief Intermediate column: 16KB pages of value_t, no header.
 *
 * All pages are arena-allocated (memory freed on arena reset between queries).
 * write_at() is thread-safe after pages are set up; source_table/column track
 * VARCHAR provenance. Move-only.
 *
 * @see construct_intermediate.h, plan.h, materialize.h
 */
struct column_t {
  public:
    /** @brief Intermediate page: fixed array of value_t entries. */
    struct alignas(IR_PAGE_SIZE) Page {
        value_t data[CAP_PER_PAGE];
    };

  private:
    size_t num_values = 0; /**< Total value count across all pages. */

  public:
    std::vector<Page *> pages; /**< Pointers to arena-allocated pages. */

    /** @brief Base table index for VARCHAR dereferencing. */
    uint8_t source_table = 0;

    /** @brief Column index within source table. */
    uint8_t source_column = 0;

  public:
    column_t() = default;

    column_t(column_t &&other) noexcept
        : num_values(other.num_values), pages(std::move(other.pages)),
          source_table(other.source_table), source_column(other.source_column) {
        other.pages.clear();
        other.num_values = 0;
    }

    column_t &operator=(column_t &&other) noexcept {
        if (this != &other) {
            num_values = other.num_values;
            pages = std::move(other.pages);
            source_table = other.source_table;
            source_column = other.source_column;

            other.pages.clear();
            other.num_values = 0;
        }
        return *this;
    }

    column_t(const column_t &) = delete;
    column_t &operator=(const column_t &) = delete;

    ~column_t() = default;

    /**
     * @brief O(1) read: idx>>12 for page, idx&0xFFF for offset.
     * @note No bounds check.
     */
    inline const value_t &operator[](size_t idx) const {
        return pages[idx >> 12]->data[idx & 0xFFF];
    }

    /** @brief Total value count. */
    size_t row_count() const { return num_values; }

    /** @brief Pre-allocate pages from arena. */
    inline void pre_allocate_from_arena(Contest::platform::ThreadArena &arena,
                                        size_t count) {
        static_assert(sizeof(Page) ==
                          Contest::platform::ChunkSize<
                              Contest::platform::ChunkType::IR_PAGE>::value,
                      "Page size mismatch with IR_PAGE chunk size");
        size_t pages_needed = (count + CAP_PER_PAGE - 1) / CAP_PER_PAGE;
        pages.reserve(pages_needed);
        for (size_t i = 0; i < pages_needed; ++i) {
            void *ptr =
                arena.alloc_chunk<Contest::platform::ChunkType::IR_PAGE>();
            pages.push_back(reinterpret_cast<Page *>(ptr));
        }
        num_values = count;
    }

    /** @brief Set row count without allocation (for assembly pattern). */
    inline void set_row_count(size_t count) { num_values = count; }

    /** @brief Thread-safe write at idx (requires pages to be set up first). */
    inline void write_at(size_t idx, const value_t &val) {
        pages[idx >> 12]->data[idx & 0xFFF] = val;
    }
};

/** @brief Alias for a collection of intermediate columns. */
using Columnar = std::vector<column_t>;

/**
 * @brief Per-base-table deferred row ID storage with multi-column tracking.
 *
 * Stores 32-bit row IDs for a single base table. All columns from this
 * base table share the same row ID lookup, reducing memory from 8 bytes
 * per column to 4 bytes per table.
 *
 * Uses 16KB pages (reuses IR_PAGE arena chunk) with 4096 uint32_t entries.
 */
struct DeferredTable {
    static constexpr size_t PAGE_SIZE = 1 << 14; // 16KB
    static constexpr size_t ENTRIES_PER_PAGE =
        PAGE_SIZE / sizeof(uint32_t);         // 4096
    static constexpr size_t ENTRY_SHIFT = 12; // log2(4096)
    static constexpr size_t ENTRY_MASK = ENTRIES_PER_PAGE - 1;

    struct alignas(PAGE_SIZE) Page {
        uint32_t data[ENTRIES_PER_PAGE];
    };

    std::vector<Page *> pages;
    size_t num_values = 0;

    /// Base table ID this deferred table references
    uint8_t base_table_id = 0;

    /// True if this deferred table comes from build side (vs probe)
    bool from_build = false;

    /// Column indices from this base table that need deferred resolution
    std::vector<uint8_t> column_indices;

    DeferredTable() = default;

    DeferredTable(DeferredTable &&other) noexcept
        : pages(std::move(other.pages)), num_values(other.num_values),
          base_table_id(other.base_table_id), from_build(other.from_build),
          column_indices(std::move(other.column_indices)) {
        other.pages.clear();
        other.num_values = 0;
    }

    DeferredTable &operator=(DeferredTable &&other) noexcept {
        if (this != &other) {
            pages = std::move(other.pages);
            num_values = other.num_values;
            base_table_id = other.base_table_id;
            from_build = other.from_build;
            column_indices = std::move(other.column_indices);
            other.pages.clear();
            other.num_values = 0;
        }
        return *this;
    }

    DeferredTable(const DeferredTable &) = delete;
    DeferredTable &operator=(const DeferredTable &) = delete;

    ~DeferredTable() = default;

    /// O(1) read: idx >> 12 for page, idx & 0xFFF for offset
    inline uint32_t operator[](size_t idx) const {
        return pages[idx >> ENTRY_SHIFT]->data[idx & ENTRY_MASK];
    }

    /// Thread-safe write at idx (requires pages set up first)
    inline void write_at(size_t idx, uint32_t row_id) {
        pages[idx >> ENTRY_SHIFT]->data[idx & ENTRY_MASK] = row_id;
    }

    size_t row_count() const { return num_values; }
    void set_row_count(size_t count) { num_values = count; }

    /// Check if this table tracks a specific base column
    bool has_column(uint8_t col_idx) const {
        for (uint8_t c : column_indices) {
            if (c == col_idx)
                return true;
        }
        return false;
    }
};

} // namespace mema

namespace Contest {

/**
 * @brief Reference from a column to its deferred table.
 */
struct DeferredColumnRef {
    uint8_t table_idx; ///< Index into IntermediateResult::deferred_tables
    uint8_t base_col;  ///< Base column index in Plan::inputs[base_table_id]
};

/**
 * @brief Lightweight intermediate result with selective materialization.
 *
 * Stores join key as (value, row_id) tuples for accelerated hashtable build
 * and zero-indirection row ID propagation. Other columns use per-table 32-bit
 * row ID storage for deferred resolution.
 *
 * For LEFT_ONLY/RIGHT_ONLY modes: join_key_tuples stores base table row IDs
 * For BOTH mode: join_key_tuples may store IR indices + DeferredTable for other
 * side
 *
 * @see AnalyzedColumnInfo for materialization decisions.
 * @see key_row_column_t for tuple storage.
 * @see DeferredTable for 32-bit row ID storage.
 */
struct IntermediateResult {
    /// Join key stored as (value, row_id) tuples for accelerated propagation.
    /// Replaces materialized column for join key when present.
    std::optional<mema::key_row_column_t> join_key_tuples;

    /// Index of join key column in output (nullopt if root or no tuples).
    std::optional<size_t> join_key_idx;

    /// Other materialized columns (non-join-key columns marked MATERIALIZE).
    std::vector<mema::column_t> materialized;

    /// Map: original column index -> index in materialized (nullopt if
    /// deferred or is join key).
    std::vector<std::optional<size_t>> materialized_map;

    /// Per-base-table deferred row ID storage. One DeferredTable per unique
    /// (from_build, base_table_id) pair. All columns from same base table share
    /// the same row ID lookup. Used for BOTH mode's non-tracked side.
    std::vector<mema::DeferredTable> deferred_tables;

    /// Map: original column index -> DeferredColumnRef (nullopt if
    /// materialized). The ref contains table_idx (into deferred_tables) and
    /// base_col for resolution.
    std::vector<std::optional<DeferredColumnRef>> deferred_map;

    /// Reference to node info for column provenance resolution.
    const AnalyzedJoinNode *node_info = nullptr;

    /// Total row count.
    size_t num_rows = 0;

    IntermediateResult() = default;
    IntermediateResult(IntermediateResult &&) = default;
    IntermediateResult &operator=(IntermediateResult &&) = default;
    IntermediateResult(const IntermediateResult &) = delete;
    IntermediateResult &operator=(const IntermediateResult &) = delete;

    /** @brief Total row count. */
    size_t row_count() const { return num_rows; }

    /** @brief Check if join key is stored as tuples. */
    bool has_join_key_tuples() const { return join_key_tuples.has_value(); }

    /** @brief Check if join key tuples contain base row IDs (vs IR indices). */
    bool join_key_has_base_rows() const {
        return join_key_tuples && join_key_tuples->stores_base_row_ids;
    }

    /** @brief Get join key tuple at index. */
    mema::KeyRowPair get_join_key_tuple(size_t idx) const {
        return join_key_tuples ? (*join_key_tuples)[idx]
                               : mema::KeyRowPair{0, 0};
    }

    /** @brief Check if column was materialized (not deferred). */
    bool is_materialized(size_t orig_idx) const {
        return orig_idx < materialized_map.size() &&
               materialized_map[orig_idx].has_value();
    }

    /** @brief Check if column is the join key (stored as tuples). */
    bool is_join_key(size_t orig_idx) const {
        return join_key_idx.has_value() && *join_key_idx == orig_idx;
    }

    /** @brief Check if column is deferred. */
    bool is_deferred(size_t orig_idx) const {
        return orig_idx < deferred_map.size() &&
               deferred_map[orig_idx].has_value();
    }

    /** @brief Get materialized column, or nullptr if deferred/join key. */
    const mema::column_t *get_materialized(size_t orig_idx) const {
        if (!is_materialized(orig_idx))
            return nullptr;
        return &materialized[*materialized_map[orig_idx]];
    }

    /** @brief Get deferred table for a column, or nullptr if materialized. */
    const mema::DeferredTable *get_deferred_table(size_t orig_idx) const {
        if (!is_deferred(orig_idx))
            return nullptr;
        return &deferred_tables[deferred_map[orig_idx]->table_idx];
    }

    /** @brief Get mutable deferred table for a column, or nullptr. */
    mema::DeferredTable *get_deferred_table_mut(size_t orig_idx) {
        if (!is_deferred(orig_idx))
            return nullptr;
        return &deferred_tables[deferred_map[orig_idx]->table_idx];
    }

    /** @brief Get base column index for deferred column. */
    uint8_t get_deferred_base_col(size_t orig_idx) const {
        if (!is_deferred(orig_idx))
            return 0;
        return deferred_map[orig_idx]->base_col;
    }

    /** @brief Get full DeferredColumnRef for a column, or nullptr. */
    const DeferredColumnRef *get_deferred_ref(size_t orig_idx) const {
        if (!is_deferred(orig_idx))
            return nullptr;
        return &(*deferred_map[orig_idx]);
    }

    /** @brief Number of deferred tables (unique base tables). */
    size_t num_deferred_tables() const { return deferred_tables.size(); }
};

/**
 * @brief Unified abstraction over columnar tables and intermediate results.
 *
 * Stores ColumnarTable* (base scans) or IntermediateResult (child joins).
 * Provides uniform interface for columnar (base table) and intermediate
 * data sources.
 *
 * @see IntermediateResult for intermediate join results.
 * @see ColumnarTable for base table storage.
 */
struct JoinInput {
    /// Either base table pointer or owned IntermediateResult.
    std::variant<const ColumnarTable *, IntermediateResult> data;

    /// Original plan node for output_attrs mapping.
    const PlanNode *node = nullptr;

    /// Analyzed plan node for materialization decisions.
    const AnalyzedNode *analyzed_node = nullptr;

    /// Base table ID (for columnar inputs).
    uint8_t table_id = 0;

    /** @brief True if data is columnar (base table). */
    bool is_columnar() const {
        return std::holds_alternative<const ColumnarTable *>(data);
    }

    /** @brief Row count for join key column. */
    size_t row_count(size_t col_idx) const {
        if (is_columnar()) {
            const auto *table = std::get<const ColumnarTable *>(data);
            return table->num_rows;
        }
        return std::get<IntermediateResult>(data).row_count();
    }

    /** @brief Total row count. */
    size_t row_count() const {
        if (is_columnar()) {
            const auto *table = std::get<const ColumnarTable *>(data);
            return table->num_rows;
        }
        return std::get<IntermediateResult>(data).row_count();
    }

    /** @brief Number of output columns. */
    size_t output_size() const {
        if (node)
            return node->output_attrs.size();
        return 0;
    }

    /**
     * @brief Get deferred table for a column index.
     *
     * For columnar inputs, returns nullptr (caller must encode fresh).
     * For IntermediateResult inputs, returns existing deferred table.
     */
    const mema::DeferredTable *get_deferred_table(size_t col_idx) const {
        if (is_columnar())
            return nullptr;
        return std::get<IntermediateResult>(data).get_deferred_table(col_idx);
    }

    /**
     * @brief Get base column index for a deferred column.
     *
     * For columnar inputs, returns 0 (caller must use column metadata).
     * For IntermediateResult inputs, returns stored base column index.
     */
    uint8_t get_deferred_base_col(size_t col_idx) const {
        if (is_columnar())
            return 0;
        return std::get<IntermediateResult>(data).get_deferred_base_col(
            col_idx);
    }

    /**
     * @brief Get full DeferredColumnRef for a column index.
     *
     * For columnar inputs, returns nullptr (caller must encode fresh).
     * For IntermediateResult inputs, returns pointer to DeferredColumnRef.
     */
    const DeferredColumnRef *get_deferred_ref(size_t col_idx) const {
        if (is_columnar())
            return nullptr;
        return std::get<IntermediateResult>(data).get_deferred_ref(col_idx);
    }
};

} // namespace Contest

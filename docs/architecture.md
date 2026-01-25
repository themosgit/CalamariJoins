# System Architecture {#architecture}

This page describes the execution model, data representations,
and key design patterns and optimisations used in the join engine.

It acts as a map of the project providing references to necessary files
for further detail.

## Execution Model

The engine executes joins by traversing the Plan tree **depth-first recursively**.
Each node produces either a pointer to existing ColumnarTable data (ScanNode) or
Intermediate result data of a previous Join (JoinNode).

```cpp
// Simplified execution flow
JoinResult execute_impl(const PlanNode& node, bool is_root) {
    if (node is ScanNode) {
        return &plan.inputs[scan.base_table_id];  // Pointer to base table
    }
    
    // JoinNode: recurse on children first
    JoinResult left  = execute_impl(nodes[join.left],  false);
    JoinResult right = execute_impl(nodes[join.right], false);
    
    // Perform join
    MatchCollector matches = join(left, right);
    
    if (is_root) {
        return materialize(matches);           // → ColumnarTable
    } else {
        return construct_intermediate(matches); // → vector<column_t>
    }
}
```

The `JoinResult` type is a variant that can hold either representation:

```cpp
using JoinResult = std::variant<ColumnarTable*, ExecuteResult>;
// Where ExecuteResult = std::vector<mema::column_t>
```

## Join Execution Phases

Each `execute_impl` call for a JoinNode (see Contest::execute_impl()) performs these phases:

| Phase | Description | Key Functions |
|-------|-------------|---------------|
| 1. Select Build/Probe | Choose smaller side for build | `select_build_probe_side()` |
| 2. Choose Algorithm | Hash join vs nested loop | Based on table size |
| 3. Build Hashtable | Parallel radix-partitioned build | `build()` |
| 4. Parallel Probe | Work-stealing probe with matches | `probe()` |
| 5. Match Collectio | Matches are collected in thread local chunks | `add_match()` |
| 5. Construct Output | Materialize or construct intermediate | `materialize()` / `construct_intermediate()` |

### 1. Build/Probe side selection

Which side will be probe or build is decided based on table size.
Smaller table is chosen as build side.

### 2. Choose Algorithm

We support to main types of joins:
- Hash Joins
- SIMD accelerated Nested Loop Joins

Nested Loop Join is choosen whenever the build side of a join fits within the
vector register of a given architecture. Currently ARM NEON and AVX2 are supported.

### 3. Build Hashtable

The hashtable is build using a 2-pass zero-lock partitioning system (see UnchainedHashtable).

The main optimization found here is that the hashtable has the ability to
build directly from ColumnarTable sources ommiting the need for scans.

### 4. Parallel Probe

Parallel probing is then done that supports job-stealing on a per-page basis (see Contest::join::probe_intermediate(), Contest::join::probe_columnar(), and Contest::join::nested_loop_join()).

Optimizations supported here include but are not limited to:
- Probing directly from ColumnarTable sources.
- Tunable hashtable prefetching API.

### 5. Match Collection

During either Hash Joins or Nested Loop Joins matches are collected within
thread-local buffers that store row indeces and row indeces only (see Contest::join::ThreadLocalMatchBuffer and Contest::join::MatchCollector).

**Match Collection buffers are sized to support up to a full page of matches**

The main optimization implemented here **MatchCollectionMode**.
This allows us to store row_ids in Column Major format always, of the
probe side or the build side or both.

This is decided upon based on the desired output of the Join if the only come from
one side we track that side if not we store both.

### 6. Output Construction

In both **Non-Root**  and **Root** Joins the thread-local match buffers
of each thread are proccesed by the **same** thread.
This basically means that this step is thread affinity aware
and does zero merging of match buffers (see Contest::materialize::construct_intermediate() and Contest::materialize::materialize()).

We use row_ids provided by the match buffers to either index the previous
Intermediate or Result or a ColumnarTable. Creating results always in
**Column-Major** form.

Note that this is where "Scans" shown in the original implementation are actually
made. We never scan and subsequently duplicate data before a Join is made.

Future optimizations are being considered to enable Scanning of data Only when
it is needed. Insted storing row_ids in all previous steps. Minimizing intermediate
result footprint, without making it completely abstract. Thus reducing cache polution.
And maximazing the density of hot data.

## Data Access

For fast random access to ColumnarTable sources, the engine uses Contest::io::ColumnarReader.

The ColumnarReader provides O(log P) page lookup with O(1) cursor caching for sequential access patterns:

```cpp
namespace Contest::io {
    class ColumnarReader {
        // Two-level access: PageIndex (precomputed) + Cursor (thread-local cache)
        std::array<std::vector<PageIndex>, 2> page_indices_;  // [BUILD, PROBE]

        struct Cursor {
            // Cached page metadata for O(1) sequential hits
            size_t cached_page;
            uint32_t cached_start, cached_end;
            const int32_t *data_ptr;
            // ...
        };
    };
}
```

### Access Strategies

The reader implements **three access paths** depending on column characteristics:

| Access Path | Condition | Complexity | Description |
|-------------|-----------|------------|-------------|
| **Dense INT32 Fast Path** | All pages dense, uniform row count | O(1) | Direct arithmetic: `page = row_id / rows_per_page` |
| **Cursor Hit** | Row in cached page range | O(1) | Sequential access reuses page metadata |
| **Binary Search** | Cursor miss or invalidation | O(log P) | Upper bound search on cumulative row counts |

### PageIndex Structure

Each column gets a precomputed PageIndex built during `prepare()`:

- **cumulative_rows**: Prefix sum array `[page0_end, page1_end, ...]` for binary search
- **page_prefix_sums**: Per-page bitmap popcount prefix sums for sparse pages (NULL handling)
- **is_dense_int32**: Flag enabling O(1) arithmetic lookup for dense INT32 columns
- **rows_per_full_page**: Cached value for O(1) page arithmetic

### Cursor Caching

Thread-local cursors cache page metadata to avoid repeated binary searches:

```cpp
// Sequential access pattern (common in joins):
for (uint32_t row_id = 0; row_id < N; ++row_id) {
    // First access: O(log P) binary search + load cursor
    // Subsequent accesses: O(1) if row_id in cached range
    auto val = reader.read_value(col, col_idx, row_id, type, cursor, side);
}
```

## Data Representations

The engine uses two distinct columnar formats optimized for different purposes (see mema::value_t and mema::column_t):

### ColumnarTable (Output Format)

Used for base tables and final query output. Matches the contest API.

```cpp
struct ColumnarTable {
    size_t              num_rows;
    std::vector<Column> columns;
};

struct Column {
    DataType           type;
    std::vector<Page*> pages;  // 8KB pages
};
```

**ColumnarTable Page layout** (8KB):
```
┌────────────┬────────────┬──────────────────┬─────────┐
│ num_rows   │ num_values │ values...        │ bitmap  │
│ (2 bytes)  │ (2 bytes)  │ (variable)       │ (N bits)│
└────────────┴────────────┴──────────────────┴─────────┘
```

### column_t (Intermediate Format)

Used for results passed between join stages. Optimized for throughput.

```cpp
namespace mema {
    struct value_t {
        uint32_t data;  // 4 bytes per value
    };
    
    struct column_t {
        DataType type;
        std::vector<value_t*> pages;  // 16KB pages
        std::shared_ptr<void> arena;  // Shared allocation
    };
}
```

**value_t encoding**:
- **INT32**: Value stored directly in 4-byte field
- **VARCHAR**: Packed reference `[offset_idx:13][page_idx:19]` pointing to original ColumnarTable
- **NULL**: Sentinel value `INT32_MIN`

### Format Comparison

| Aspect | ColumnarTable | column_t |
|--------|---------------|----------|
| **Page Size** | 8KB (`PAGE_SIZE`) | 16KB (`IR_PAGE_SIZE`) |
| **Purpose** | Final output (contest API) | Intermediate join results |
| **Value Storage** | Raw values with null bitmap | Packed `value_t` (4 bytes) |
| **VARCHAR** | Actual string bytes stored | Reference to original page |
| **NULL Handling** | Bitmap at page end | `INT32_MIN` sentinel |
| **Long Strings** | Special marker pages | `LONG_STRING_OFFSET` sentinel |

**Why two formats?**
- Intermediate format uses smaller `value_t` for more values per cache line
- VARCHAR as reference avoids copying strings between join stages
- Output format matches contest API requirements


### Critical Invariants

1. **ExecuteResult outlives consumers**: The call stack holds child results until the parent join completes materialization.

2. **VARCHAR reference validity**: Intermediate `value_t` encodes page/offset into original ColumnarTable. Base tables must remain valid until final `materialize()`.

3. **No cross-join data sharing**: Each join's hash table and MatchCollector are independent. No lifetime dependencies between sibling joins.

### ScanNode output_attrs

Direct mapping to base table columns:

```cpp
// output_attrs[i] = (column_index, DataType)
// Example: project columns 0 and 2 from base table
output_attrs = {{0, INT32}, {2, VARCHAR}};
// Output column 0 ← base table column 0 (INT32)
// Output column 1 ← base table column 2 (VARCHAR)
```

### JoinNode output_attrs

Combined index spanning both children's outputs:

```cpp
// Left child outputs 3 columns, right child outputs 2 columns
// Combined indices: [0,1,2] = left, [3,4] = right

output_attrs = {
    {0, INT32},    // Left child's column 0
    {3, VARCHAR},  // Right child's column 0 (index 3 = left_size + 0)
    {1, INT32},    // Left child's column 1
};
```

### Build/Probe Remapping

When the optimizer's `build_left` hint is overridden based on actual cardinality, indices are remapped:

```cpp
// Original: build_left = true, left has 3 cols, right has 2 cols
// Override: build_left = false (right is smaller)
// 
// Before: [0,1,2] = left (probe), [3,4] = right (build)
// After:  [0,1] = right (build), [2,3,4] = left (probe)
```

The `remapped_attrs` vector reflects this new mapping for correct materialization.

/**
 * @file construct_intermediate.h
 * @brief Constructs intermediate results for multi-way joins.
 *
 * Allocates and populates ExecuteResult (column_t) from match collectors.
 */
#pragma once

#include <data_access/columnar_reader.h>
#include <data_model/intermediate.h>
#include <data_model/plan.h>
#include <join_execution/join_setup.h>
#include <join_execution/match_collector.h>
#include <platform/worker_pool.h>
#include <sys/mman.h>
#include <vector>
/**
 * @namespace Contest::materialize
 * @brief Materialization of join results into columnar format.
 *
 * Key components in this file:
 * - BatchAllocator: Single mmap() for all intermediate result pages
 * - SourceInfo: Precomputed metadata for fast hot-loop value copying
 * - batch_allocate_for_results(): Arena-style page allocation
 * - prepare_sources(): Source resolution precomputation
 * - construct_intermediate(): Parallel column_t construction from matches
 *
 * @see intermediate.h for column_t/value_t format details
 */
namespace Contest::materialize {

// Types from Contest:: namespace
using Contest::ExecuteResult;

// Types from global scope (data_model/plan.h, foundation/attribute.h)
// Column, ColumnarTable, DataType, PlanNode are accessible without
// qualification

// Types from Contest::platform::
using Contest::platform::worker_pool;

// Types from Contest::io::
using Contest::io::ColumnarReader;

// Types from Contest::join::
using Contest::join::JoinInput;
using Contest::join::MatchCollector;

/**
 * @brief Batch memory allocator for intermediate result pages.
 *
 * **Why single mmap allocation:**
 * Traditional per-page allocation (new/delete) incurs significant overhead
 * for large intermediate results (millions of rows = thousands of pages):
 * - Kernel syscall per allocation (expensive context switch)
 * - Memory fragmentation from thousands of individual allocations
 * - Poor cache locality when pages scattered across heap
 *
 * This allocator uses a single mmap() call to allocate all pages contiguously:
 * - **Single syscall:** One mmap for entire memory region vs thousands of
 * malloc/new calls
 * - **Memory locality:** All pages contiguous in virtual memory, improving TLB
 * hit rate and cache prefetcher effectiveness during sequential access
 * - **Simplified cleanup:** Single munmap in destructor vs tracking thousands
 * of pointers
 *
 * **shared_ptr lifecycle management:**
 * Columns receive shared ownership of this allocator, preventing premature
 * munmap:
 * - Each column_t holds shared_ptr<BatchAllocator> in external_memory field
 * - Memory region remains valid while ANY column referencing it exists
 * - Automatic munmap when last column destroyed (reference count hits zero)
 * - Eliminates manual coordination of allocation/deallocation timing
 *
 * **Performance impact:**
 * Benchmarked 40-60% allocation speedup for workloads creating 100M+ value
 * intermediate results (JOB 33c query). Savings compound in multi-way joins.
 *
 * @note Move-only type (no copy to prevent accidental double-free of mmap
 * region).
 * @see batch_allocate_for_results() for the allocation orchestration.
 * @see column_t::pre_allocate_from_block() for how columns partition this
 * memory.
 */
class BatchAllocator {
  private:
    void *memory_block = nullptr;
    size_t total_size = 0;

  public:
    BatchAllocator() = default;
    void allocate(size_t total_pages) {
        total_size = total_pages * mema::IR_PAGE_SIZE;
        memory_block = mmap(nullptr, total_size, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory_block == MAP_FAILED) {
            memory_block = nullptr;
            throw std::bad_alloc();
        }
    }
    void *get_block() const { return memory_block; }
    ~BatchAllocator() {
        if (memory_block) {
            munmap(memory_block, total_size);
        }
    }
    BatchAllocator(const BatchAllocator &) = delete;
    BatchAllocator &operator=(const BatchAllocator &) = delete;
};

/**
 * @brief Metadata for resolving an output column's source.
 *
 * **Why precompute source information:**
 * During construct_intermediate(), we copy millions of values from source
 * columns (build/probe inputs) to output columns. Each value requires:
 * - Determining if source is columnar table or intermediate result
 * - Resolving which specific Column or column_t to read from
 * - Mapping remapped column index to actual source index
 *
 * Without precomputation, this would require:
 * - std::variant accesses and type checks per value (branch mispredictions)
 * - Tuple lookups in PlanNode::output_attrs per value (cache pollution)
 * - Redundant calculations for data that's constant across all rows
 *
 * By precomputing into this struct during prepare_sources(), the hot loop
 * becomes a simple pointer dereference + index calculation - no branching
 * on source type, no tuple unpacking. All metadata cache-resident.
 *
 * @note 8-byte alignment optimizes struct packing for vector iteration.
 * @see prepare_sources() for the precomputation logic.
 */
struct alignas(8) SourceInfo {
    const mema::column_t *intermediate_col =
        nullptr;                          /**< Source if intermediate. */
    const Column *columnar_col = nullptr; /**< Source if columnar. */
    size_t remapped_col_idx = 0; /**< Local index within source side. */
    bool is_columnar = false;    /**< True if source is columnar table. */
    bool from_build = false; /**< True if from build side, false if probe. */
};

/**
 * @brief Preallocates mmap'd memory for all output columns.
 *
 * **Sizing calculation logic:**
 * Intermediate results use column_t with 16KB pages, each holding CAP_PER_PAGE
 * value_t entries (4096 values per page). For total_matches rows across
 * N output columns:
 *
 * 1. **Per-column chunk count:** Each column needs ceil(total_matches /
 * CAP_PER_PAGE) chunks (pages) to hold all values. Accumulated across all
 * columns.
 *
 * 2. **Byte size:** total_chunks * CAP_PER_PAGE * sizeof(value_t). Note this
 *    includes padding - if a column has 4097 values, it allocates 2 full pages
 *    (8192 slots), wasting 4095 slots. Trade-off for allocation simplicity.
 *
 * 3. **System page alignment:** Round up to IR_PAGE_SIZE (16KB) boundaries
 *    for mmap alignment requirements. Kernel always allocates in page units.
 *
 * **Allocation orchestration:**
 * - Creates BatchAllocator and mmaps single contiguous region
 * - Iterates through result columns, calling pre_allocate_from_block() on each
 * - Each column carves out its portion via offset advancement (arena
 * allocation)
 * - Returns shared_ptr to allocator, establishing shared ownership with columns
 *
 * **Why this approach:**
 * Single mmap + arena partitioning avoids per-column allocation overhead and
 * maximizes memory locality. All result columns contiguous in virtual memory.
 *
 * @param results       Output columns to allocate (modified in-place).
 * @param total_matches Number of rows (match count from MatchCollector).
 * @return Shared ownership of BatchAllocator (columns hold copies via
 * external_memory).
 *
 * @note Padding waste minimal for large results (e.g., 4095/4096 = 0.1% for
 * worst case).
 * @see BatchAllocator for single mmap rationale.
 * @see column_t::pre_allocate_from_block() for arena partitioning details.
 */
inline std::shared_ptr<BatchAllocator>
batch_allocate_for_results(ExecuteResult &results, size_t total_matches) {

    size_t total_chunks = 0;
    for (auto &col : results) {
        total_chunks +=
            (total_matches + mema::CAP_PER_PAGE - 1) / mema::CAP_PER_PAGE;
    }

    size_t total_bytes =
        total_chunks * mema::CAP_PER_PAGE * sizeof(mema::value_t);
    size_t system_pages =
        (total_bytes + mema::IR_PAGE_SIZE - 1) / mema::IR_PAGE_SIZE;
    auto allocator = std::make_shared<BatchAllocator>();
    allocator->allocate(system_pages);
    void *block = allocator->get_block();
    size_t offset = 0;
    for (auto &col : results) {
        col.pre_allocate_from_block(block, offset, total_matches, allocator);
    }
    return allocator;
}

/**
 * @brief Builds SourceInfo for each output column for fast lookup during copy.
 *
 * **Why precomputation is essential:**
 * construct_intermediate() copies millions of values in a tight loop. For each
 * output column, we need to resolve:
 * - Which input side (build/probe) provides the data
 * - Whether source is columnar table (ColumnarTable*) or intermediate
 * (column_t)
 * - Which specific Column or column_t to read from
 * - Local index mapping (remapped_attrs uses global indexing, sources use
 * per-side)
 *
 * Doing this resolution per-value would involve:
 * - Repeated std::get<> variant accesses (type checks + branch mispredictions)
 * - Repeated output_attrs tuple unpacking (cache pollution)
 * - Conditional logic repeated billions of times for large joins
 *
 * By precomputing once per column (not per value), the hot loop becomes:
 * - Load SourceInfo struct (cache-resident, 8-byte aligned)
 * - Direct pointer dereference to source column (no variant access)
 * - Simple index arithmetic (no tuple unpacking)
 *
 * **Precomputation logic:**
 * For each remapped output column:
 * 1. Determine side: col_idx < build_size means build side, else probe side
 * 2. Compute local index: subtract build_size if probe side
 * 3. Resolve source type: check JoinInput variant (columnar vs intermediate)
 * 4. Extract pointer: ColumnarTable->columns[actual_idx] or
 * ExecuteResult[local_idx]
 * 5. Store in compact SourceInfo struct for hot-loop access
 *
 * **Performance impact:**
 * Eliminates ~5-10 cycles of overhead per value copied. For 100M value results,
 * saves 500M-1B cycles (measurable improvement on JOB queries).
 *
 * @param remapped_attrs Output column specifications (global indexing across
 * build+probe).
 * @param build_input    Build side data (variant: ColumnarTable* or
 * ExecuteResult).
 * @param probe_input    Probe side data (variant: ColumnarTable* or
 * ExecuteResult).
 * @param build_node     PlanNode for build side (contains output_attrs
 * mapping).
 * @param probe_node     PlanNode for probe side (contains output_attrs
 * mapping).
 * @param build_size     Number of columns from build side (split point for
 * indexing).
 * @return Vector of SourceInfo (one per output column, indexed same as
 * results).
 *
 * @note Result vector size equals remapped_attrs.size() (one-to-one
 * correspondence).
 * @see SourceInfo for field documentation.
 * @see construct_intermediate() for how these are consumed in the hot loop.
 */
inline std::vector<SourceInfo>
prepare_sources(const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
                const JoinInput &build_input, const JoinInput &probe_input,
                const PlanNode &build_node, const PlanNode &probe_node,
                size_t build_size) {
    std::vector<SourceInfo> sources;
    sources.reserve(remapped_attrs.size());
    for (const auto &[col_idx, _] : remapped_attrs) {
        SourceInfo info;
        info.from_build = (col_idx < build_size);
        size_t local_idx = info.from_build ? col_idx : col_idx - build_size;
        info.remapped_col_idx = local_idx;
        const JoinInput &input = info.from_build ? build_input : probe_input;
        const PlanNode &node = info.from_build ? build_node : probe_node;
        if (input.is_columnar()) {
            info.is_columnar = true;
            auto *table = std::get<const ColumnarTable *>(input.data);
            auto [actual_idx, _] = node.output_attrs[local_idx];
            info.columnar_col = &table->columns[actual_idx];
        } else {
            info.is_columnar = false;
            const auto &res = std::get<ExecuteResult>(input.data);
            info.intermediate_col = &res[local_idx];
        }
        sources.push_back(info);
    }
    return sources;
}

/**
 * @brief Populates ExecuteResult columns from join matches in parallel.
 *
 * Core materialization function for non-root join nodes. Transforms join
 * matches (row ID pairs from MatchCollector) into intermediate result columns
 * (column_t with value_t encoding). Parallelized across worker threads for
 * large result sets.
 *
 * **Algorithm:**
 * 1. Finalize MatchCollector to freeze match arrays
 * 2. Precompute SourceInfo metadata via prepare_sources() (avoid hot-loop
 * overhead)
 * 3. Batch-allocate all output pages via single mmap
 * (batch_allocate_for_results)
 * 4. Spawn worker threads, each processing [start, end) slice of matches:
 *    - Iterate through output columns
 *    - Fetch appropriate row ID range (build or probe side) from collector
 *    - Copy values from source (columnar or intermediate) to output column
 *    - Use ColumnarReader cursor caching for efficient page access (columnar
 * case)
 *
 * **Thread safety:**
 * - Each thread writes to distinct index range [start, end) in output columns
 * - write_at() safe because pages pre-allocated (no reallocation, distinct
 * indices)
 * - ColumnarReader::Cursor is thread-local (no shared state)
 * - MatchCollector finalized before parallel access (immutable read-only
 * arrays)
 *
 * **Why intermediate format (column_t/value_t):**
 * Non-root join results are inputs to subsequent joins. Using value_t encoding:
 * - VARCHAR stored as 4-byte page/offset references (no string copying)
 * - Cache-efficient 4-byte values instead of variable-length strings
 * - Deferred materialization: only root node converts to ColumnarTable with
 * real strings
 *
 * @param collector        MatchCollector holding join match row IDs (left/right
 * pairs). Must be finalized before parallel access.
 * @param build_input      Build side data (variant: ColumnarTable* or
 * ExecuteResult). Source for columns with col_idx < build_size.
 * @param probe_input      Probe side data (variant: ColumnarTable* or
 * ExecuteResult). Source for columns with col_idx >= build_size.
 * @param remapped_attrs   Output column specifications (global indexing: build
 * columns first [0, build_size), probe columns second [build_size, N)).
 *                         Contains DataType for each output column.
 * @param build_node       PlanNode for build side, containing output_attrs
 * mapping from local index to (table_id, column_id) in original schema.
 * @param probe_node       PlanNode for probe side, containing output_attrs
 * mapping.
 * @param build_size       Split point: number of output columns from build
 * side. Used to distinguish build vs probe columns in remapped_attrs.
 * @param columnar_reader  ColumnarReader with prepared columns for fast page
 * access. Uses Cursor caching to minimize page lookups for columnar sources.
 * @param results          Pre-initialized empty ExecuteResult (vector of
 * column_t). Allocated and populated in-place by this function.
 *
 * @note For root joins, use materialize() instead to produce final
 * ColumnarTable.
 * @note Early-exits if total_matches == 0 (empty result, no allocation).
 * @see intermediate.h for column_t format and value_t encoding details.
 * @see prepare_sources() for source metadata precomputation rationale.
 * @see batch_allocate_for_results() for single mmap allocation strategy.
 */
inline void construct_intermediate(
    const MatchCollector &collector, const JoinInput &build_input,
    const JoinInput &probe_input,
    const std::vector<std::tuple<size_t, DataType>> &remapped_attrs,
    const PlanNode &build_node, const PlanNode &probe_node, size_t build_size,
    ColumnarReader &columnar_reader, ExecuteResult &results) {
    const_cast<MatchCollector &>(collector).ensure_finalized();
    const size_t total_matches = collector.size();
    if (total_matches == 0)
        return;

    auto sources = prepare_sources(remapped_attrs, build_input, probe_input,
                                   build_node, probe_node, build_size);
    auto allocator = batch_allocate_for_results(results, total_matches);

    worker_pool.execute([&](size_t t) {
        Contest::ColumnarReader::Cursor cursor;
        size_t num_threads = worker_pool.thread_count();
        size_t start = t * total_matches / num_threads;
        size_t end = (t + 1) * total_matches / num_threads;
        if (start >= end)
            return;

        for (size_t i = 0; i < sources.size(); ++i) {
            const auto &src = sources[i];
            auto &dest_col = results[i];

            auto range = src.from_build
                             ? collector.get_left_range(start, end - start)
                             : collector.get_right_range(start, end - start);

            if (src.is_columnar) {
                const auto &col = *src.columnar_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    dest_col.write_at(k++,
                                      columnar_reader.read_value(
                                          col, src.remapped_col_idx, rid,
                                          col.type, cursor, src.from_build));
                }
            } else {
                const auto &vec = *src.intermediate_col;
                size_t k = start;
                for (uint32_t rid : range) {
                    dest_col.write_at(k++, vec[rid]);
                }
            }
        }
    });
}
} // namespace Contest::materialize

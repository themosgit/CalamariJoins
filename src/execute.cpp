/**
 * @file execute.cpp
 * @brief Recursive join tree execution engine.
 *
 * Executes multi-way joins by traversing the plan tree depth-first.
 * Each join node: resolve inputs -> select build/probe sides -> choose
 * algorithm (hash vs nested loop) -> collect matches -> materialize or
 * construct intermediate.
 *
 * **Execution flow:**
 * 1. execute() is called with the Plan and root node index
 * 2. execute_impl() recursively processes join nodes
 * 3. resolve_join_input() handles ScanNode (returns ColumnarTable*) or
 *    JoinNode (recurses, returns ExecuteResult)
 * 4. Root joins produce ColumnarTable via materialize()
 * 5. Non-root joins produce ExecuteResult via construct_intermediate()
 *
 * **Data lifetimes:**
 * - Base tables (ColumnarTable*) live for entire query execution
 * - ExecuteResult from child joins is held on call stack until parent completes
 * - VARCHAR value_t references remain valid because base tables outlive them
 *
 * Row ordering is non-deterministic due to work-stealing parallelism.
 * This is semantically correct as SQL joins don't guarantee order.
 *
 * @see plan.h for Plan structure and output_attrs mapping.
 * @see match_collector.h for parallel match accumulation.
 * @see materialize.h and construct_intermediate.h for output construction.
 */
#include <foundation/attribute.h>
#if defined(__APPLE__) && defined(__aarch64__)
#include <platform/hardware_darwin.h>
#elif defined(SPC__USE_BENCHMARKVM_HARDWARE)
#include <platform/hardware_benchmarkvm.h>
#else
#include <platform/hardware.h>
#endif

#include <chrono>
#include <data_access/columnar_reader.h>
#include <data_model/intermediate.h>
#include <iostream>
#include <join_execution/hash_join.h>
#include <join_execution/hashtable.h>
#include <join_execution/join_setup.h>
#include <join_execution/nested_loop.h>
#include <materialization/construct_intermediate.h>
#include <materialization/materialize.h>
#include <variant>

namespace Contest {

// Import join execution types into Contest namespace
using namespace join;

// Import specific materialization functions (namespace qualified for
// materialize::materialize to avoid collision with namespace name)
using materialize::construct_intermediate;
using materialize::create_empty_result;

/**
 * @brief Result of execute_impl(): either intermediate or final output.
 *
 * - ExecuteResult (vector<column_t>): Intermediate format for non-root joins.
 *   Values stored as value_t with VARCHAR encoded as page/offset references.
 * - ColumnarTable: Final output format for root join, per contest API.
 */
using JoinResult = std::variant<ExecuteResult, ColumnarTable>;

/**
 * @brief Recursive join execution with timing instrumentation.
 *
 * @param plan      Complete query plan with nodes and base tables.
 * @param node_idx  Index of current node in plan.nodes.
 * @param is_root   If true, produce ColumnarTable; otherwise ExecuteResult.
 * @param stats     Accumulates timing for each execution phase.
 * @return JoinResult containing either intermediate or final result.
 */
JoinResult execute_impl(const Plan &plan, size_t node_idx, bool is_root,
                        TimingStats &stats);

/**
 * @brief Resolve a plan node to its join input data.
 *
 * For ScanNode: Returns non-owning pointer to pre-loaded ColumnarTable.
 * For JoinNode: Recursively executes the child join, returns owned
 * ExecuteResult.
 *
 * This function embodies the depth-first traversal: leaf nodes (scans) return
 * immediately, while join nodes trigger recursive execution.
 *
 * @param plan      Complete query plan.
 * @param node_idx  Index of node to resolve.
 * @param stats     Timing accumulator for recursive calls.
 * @return JoinInput with data variant and metadata.
 */
JoinInput resolve_join_input(const Plan &plan, size_t node_idx,
                             TimingStats &stats) {
    JoinInput input;
    const auto &node = plan.nodes[node_idx];
    input.node = &node;

    if (const auto *scan = std::get_if<ScanNode>(&node.data)) {
        input.data = &plan.inputs[scan->base_table_id];
        input.table_id = scan->base_table_id;
    } else {
        auto result = execute_impl(plan, node_idx, false, stats);
        input.data = std::get<ExecuteResult>(std::move(result));
        input.table_id = 0;
    }
    return input;
}

/**
 * @brief Core recursive join execution.
 *
 * **Execution phases per join:**
 * 1. Resolve left and right child inputs (may recurse)
 * 2. Select build/probe sides (smaller input becomes build side)
 * 3. Choose algorithm: hash join for larger tables, nested loop for tiny
 * 4. Build hash table (or use stack-allocated keys for nested loop)
 * 5. Probe in parallel, collecting matches to MatchCollector
 * 6. Output construction based on is_root flag
 *
 * **Algorithm selection:**
 * - Nested loop used when build side has fewer than HASH_TABLE_THRESHOLD rows
 * - Hash join with radix-partitioned parallel build otherwise
 *
 * **Memory management:**
 * - Hash table and MatchCollector are local to this call, freed on return
 * - Child ExecuteResults held on stack until materialization completes
 * - setup.results pre-allocated for non-root output
 */
JoinResult execute_impl(const Plan &plan, size_t node_idx, bool is_root,
                        TimingStats &stats) {
    auto &node = plan.nodes[node_idx];

    if (!std::holds_alternative<JoinNode>(node.data)) {
        return ExecuteResult{};
    }

    const auto &join = std::get<JoinNode>(node.data);
    const auto &output_attrs = node.output_attrs;
    const auto &left_node = plan.nodes[join.left];
    const auto &right_node = plan.nodes[join.right];

    /* Resolve child inputs: ScanNodes return ColumnarTable*, JoinNodes recurse
     */
    JoinInput left_input = resolve_join_input(plan, join.left, stats);
    JoinInput right_input = resolve_join_input(plan, join.right, stats);

    /**
     * Select build/probe sides: smaller input becomes build side to minimize
     * hash table size. Remaps output_attrs accordingly if sides are swapped.
     * @see join_setup.h::select_build_probe_side() for the heuristic.
     */
    auto setup_start = std::chrono::high_resolution_clock::now();
    auto config =
        select_build_probe_side(join, left_input, right_input, output_attrs);
    const JoinInput &build_input = config.build_left ? left_input : right_input;
    const JoinInput &probe_input = config.build_left ? right_input : left_input;
    const auto &build_node = config.build_left ? left_node : right_node;
    const auto &probe_node = config.build_left ? right_node : left_node;

    bool build_is_columnar = build_input.is_columnar();
    bool probe_is_columnar = probe_input.is_columnar();

    /**
     * Algorithm selection: nested loop for tiny tables (< 8 rows) to avoid
     * hash table overhead. At this size, linear scan is faster due to:
     * - No hash computation or partitioning cost
     * - Entire build side fits in L1 cache
     * - Simple loop has better branch prediction
     * @see nested_loop.h for the nested loop implementation.
     */
    const size_t HASH_TABLE_THRESHOLD = 8;
    size_t build_rows = build_input.row_count(config.build_attr);
    bool use_nested_loop = (build_rows < HASH_TABLE_THRESHOLD);

    /**
     * Setup pre-allocates ExecuteResult for non-root joins and prepares
     * ColumnarReader infrastructure (PageIndex built lazily on demand).
     */
    JoinSetup setup = setup_join(build_input, probe_input, build_node,
                                 probe_node, left_node, right_node, left_input,
                                 right_input, output_attrs, build_rows);
    auto setup_end = std::chrono::high_resolution_clock::now();
    auto setup_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        setup_end - setup_start);
    stats.setup_ms += setup_elapsed.count();

    /**
     * Optimize match collection: if output only needs columns from one side,
     * skip collecting row IDs from the other side (50% memory savings).
     * @see join_setup.h::determine_collection_mode() for the optimization.
     */
    MatchCollectionMode collection_mode = determine_collection_mode(
        config.remapped_attrs, config.build_left ? left_input.output_size()
                                                 : right_input.output_size());

    MatchCollector collector(collection_mode);

    /* Nested loop join: used for tiny tables where hash overhead exceeds
     * benefit */
    if (use_nested_loop) {

        auto nested_loop_start = std::chrono::high_resolution_clock::now();

        nested_loop_join(build_input, probe_input, config.build_attr,
                         config.probe_attr, collector, collection_mode);
        auto nested_loop_end = std::chrono::high_resolution_clock::now();
        auto nested_loop_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                nested_loop_end - nested_loop_start);
        stats.nested_loop_join_ms += nested_loop_elapsed.count();
    } else {
        /**
         * Hash join: radix-partitioned parallel build, then parallel probe.
         * Build function dispatches on data type (ColumnarTable vs
         * ExecuteResult).
         * @see hashtable.h for the unchained hash table structure.
         * @see hash_join.h for build/probe implementations.
         */
        auto build_start = std::chrono::high_resolution_clock::now();
        UnchainedHashtable hash_table =
            build_is_columnar
                ? build_from_columnar(build_input, config.build_attr)
                : build_from_intermediate(build_input, config.build_attr);
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(build_end -
                                                                  build_start);
        stats.hashtable_build_ms += build_elapsed.count();

        /**
         * Parallel probe with work-stealing: threads grab probe pages
         * atomically, accumulating matches to thread-local buffers for
         * lock-free collection.
         */
        auto probe_start = std::chrono::high_resolution_clock::now();
        if (probe_is_columnar) {
            probe_columnar(hash_table, probe_input, config.probe_attr,
                           collector, collection_mode);
        } else {
            const auto &probe_result =
                std::get<ExecuteResult>(probe_input.data);
            probe_intermediate(hash_table, probe_result[config.probe_attr],
                               collector, collection_mode);
        }
        auto probe_end = std::chrono::high_resolution_clock::now();
        auto probe_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(probe_end -
                                                                  probe_start);
        stats.hash_join_probe_ms += probe_elapsed.count();
    }
    JoinResult final_result;
    if (is_root) {
        auto mat_start = std::chrono::high_resolution_clock::now();
        /**
         * Empty result fast path: avoid PageIndex build and materialization
         * overhead when no matches exist. Produces schema-correct empty table.
         */
        if (collector.size() == 0) {
            final_result = create_empty_result(config.remapped_attrs);
        } else {
            /**
             * Lazy PageIndex construction: only build indices for columns
             * actually needed in output (projection pushdown optimization).
             * Built here after probing completes, before materialization needs
             * it.
             */
            prepare_output_columns(
                setup.columnar_reader, build_input, probe_input, build_node,
                probe_node, config.remapped_attrs, build_input.output_size());

            final_result = materialize::materialize(
                collector, build_input, probe_input, config.remapped_attrs,
                build_node, probe_node, build_input.output_size(),
                setup.columnar_reader, plan);
        }
        auto mat_end = std::chrono::high_resolution_clock::now();
        stats.materialize_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(mat_end -
                                                                  mat_start)
                .count();

    } else {
        auto inter_start = std::chrono::high_resolution_clock::now();
        if (collector.size() > 0) {
            /**
             * Lazy PageIndex construction for non-root joins.
             * Same projection optimization as root case.
             */
            prepare_output_columns(
                setup.columnar_reader, build_input, probe_input, build_node,
                probe_node, config.remapped_attrs, build_input.output_size());

            construct_intermediate(collector, build_input, probe_input,
                                   config.remapped_attrs, build_node,
                                   probe_node, build_input.output_size(),
                                   setup.columnar_reader, setup.results);
        }
        /**
         * Transfer ownership of setup.results (pre-allocated ExecuteResult)
         * to caller. Held on stack until parent join completes materialization.
         */
        final_result = std::move(setup.results);
        auto inter_end = std::chrono::high_resolution_clock::now();
        stats.intermediate_ms +=
            std::chrono::duration_cast<std::chrono::milliseconds>(inter_end -
                                                                  inter_start)
                .count();
    }

    return final_result;
}

/**
 * @brief Public entry point for query execution.
 *
 * Executes the complete join plan starting from the root node. The root
 * join always produces a ColumnarTable as required by the contest API.
 *
 * Timing breakdown is captured in stats and optionally printed to stdout.
 *
 * @param plan      Query plan with nodes and pre-loaded base tables.
 * @param context   Execution context (currently unused, reserved for future).
 * @param stats_out Optional output for timing breakdown.
 * @param show_detailed_timing If true, print timing to stdout.
 * @return Final join result as ColumnarTable.
 */
ColumnarTable execute(const Plan &plan, void *context, TimingStats *stats_out,
                      bool show_detailed_timing) {
    auto total_start = std::chrono::high_resolution_clock::now();

    TimingStats stats;
    auto result = execute_impl(plan, plan.root, true, stats);

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        total_end - total_start);
    stats.total_execution_ms = total_elapsed.count();

    if (show_detailed_timing) {
        int64_t accounted =
            stats.hashtable_build_ms + stats.hash_join_probe_ms +
            stats.nested_loop_join_ms + stats.materialize_ms + stats.setup_ms;
        int64_t other = stats.total_execution_ms - accounted;

        std::cout << "Hashtable Build Time: " << stats.hashtable_build_ms
                  << " ms\n";
        std::cout << "Hash Join Probe Time: " << stats.hash_join_probe_ms
                  << " ms\n";
        std::cout << "Nested Loop Join Time: " << stats.nested_loop_join_ms
                  << " ms\n";
        std::cout << "Materialization Time: " << stats.materialize_ms
                  << " ms\n";
        std::cout << "Intermediate Time: " << stats.intermediate_ms << " ms\n";
        std::cout << "Setup Time: " << stats.setup_ms << " ms\n";
        std::cout << "Other Overhead: " << other << " ms\n";
        std::cout << "Total Execution Time: " << stats.total_execution_ms
                  << " ms\n";
    }

    if (stats_out) {
        *stats_out = stats;
    }

    return std::move(std::get<ColumnarTable>(result));
}

/** @brief Allocate execution context. Currently a no-op; reserved for future
 * use. */
void *build_context() { return nullptr; }

/** @brief Release execution context. Currently a no-op. */
void destroy_context(void *context) { (void)context; }

} // namespace Contest

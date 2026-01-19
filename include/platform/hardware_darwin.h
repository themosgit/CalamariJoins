/**
 * @file hardware_darwin.h
 * @brief Hardware configuration for Apple Silicon (M1/M2/M3).
 *
 * 128-byte cache lines, 4MB L2/cluster, no L3, NEON SIMD, LSE atomics.
 * Detected via `__APPLE__ && __aarch64__`.
 *
 * @note E-cores have different cache sizes but are not used for joins.
 * @see hardware.h for generic x86-64 fallback.
 * @see hardware_benchmarkvm.h for contest VM.
 */
#define SPC__AARCH64
#define SPC__CPU_NAME "Apple M1"
#define SPC__CORE_COUNT 8
#define SPC__THREAD_COUNT 8
#define SPC__NUMA_NODE_COUNT 1
#define SPC__NUMA_NODES_ACTIVE_IN_BENCHMARK 1
#define SPC__NUMA_NODE_DRAM_MB 16384
#define SPC__OS "macOS 26.0"
#define SPC__KERNEL "Darwin 25.0.0 arm64"
#define SPC__SUPPORTS_NEON

#define SPC__LEVEL1_ICACHE_SIZE 196608
#define SPC__LEVEL1_ICACHE_ASSOC 8
#define SPC__LEVEL1_ICACHE_LINESIZE 128
#define SPC__LEVEL1_DCACHE_SIZE 131072 // 64 KB
#define SPC__LEVEL1_DCACHE_ASSOC 8
#define SPC__LEVEL1_DCACHE_LINESIZE 128
#define SPC__LEVEL1_DCACHE_SETS 64      // 64 KB / (8-way × 128B line)
#define SPC__LEVEL2_CACHE_SIZE 12582912 // 4 MB
#define SPC__LEVEL2_CACHE_ASSOC 16
#define SPC__LEVEL2_CACHE_LINESIZE 128
#define SPC__LEVEL3_CACHE_SIZE 0
#define SPC__LEVEL3_CACHE_ASSOC 0
#define SPC__LEVEL3_CACHE_LINESIZE 0
#define SPC__LEVEL4_CACHE_SIZE 0
#define SPC__LEVEL4_CACHE_ASSOC 0
#define SPC__LEVEL4_CACHE_LINESIZE 0

#define SPC__PAGE_SIZE 16384
#define SPC__PREFETCH_DISTANCE 6
#define SPC__MEMORY_LATENCY_NS 100
#define SPC__SIMD_WIDTH 16
#define SPC__HAS_LSE

/**
 * @file hardware_darwin.h
 * @brief Hardware configuration for Apple Silicon (M1/M2/M3).
 *
 * Platform-specific constants for Apple Silicon Macs. Detected via
 * `__APPLE__ && __aarch64__` preprocessor guards. Values reflect M1
 * performance cores (P-cores) which are used for compute-intensive work.
 *
 * ### Apple Silicon characteristics:
 * - 128-byte cache lines (vs 64-byte on x86)
 * - Large L2 cache per P-core (4MB shared cluster)
 * - No L3 cache (unified memory architecture)
 * - NEON SIMD support (128-bit vectors)
 * - LSE atomics for efficient lock-free operations
 *
 * @note E-cores (efficiency cores) have different cache sizes but are
 *       typically not used for join processing due to lower performance.
 *
 * @see hardware.h for generic x86-64 fallback
 * @see hardware_benchmarkvm.h for contest benchmark VM configuration
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

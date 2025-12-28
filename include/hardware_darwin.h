// Hardware information for Apple M1 Mac
// Darwin 25.0.0 arm64
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

// l1 and l2 caches from performance cores
#define SPC__LEVEL1_ICACHE_SIZE 196608 // 128 KB
#define SPC__LEVEL1_ICACHE_ASSOC 8
#define SPC__LEVEL1_ICACHE_LINESIZE 128
#define SPC__LEVEL1_DCACHE_SIZE 131072 // 64 KB
#define SPC__LEVEL1_DCACHE_ASSOC 8
#define SPC__LEVEL1_DCACHE_LINESIZE 128
#define SPC__LEVEL1_DCACHE_SETS 64 // 64 KB / (8-way × 128B line)
#define SPC__LEVEL2_CACHE_SIZE 12582912 // 4 MB
#define SPC__LEVEL2_CACHE_ASSOC 16
#define SPC__LEVEL2_CACHE_LINESIZE 128
#define SPC__LEVEL3_CACHE_SIZE 0
#define SPC__LEVEL3_CACHE_ASSOC 0
#define SPC__LEVEL3_CACHE_LINESIZE 0
#define SPC__LEVEL4_CACHE_SIZE 0
#define SPC__LEVEL4_CACHE_ASSOC 0
#define SPC__LEVEL4_CACHE_LINESIZE 0

// Memory and performance characteristics
#define SPC__PAGE_SIZE 16384 // 16 KB (queried via getconf PAGE_SIZE)
#define SPC__PREFETCH_DISTANCE 6 // Prefetch 6 cache lines ahead (768 bytes)
#define SPC__MEMORY_LATENCY_NS 100 // ~100ns main memory latency (typical for M1)
#define SPC__SIMD_WIDTH 16 // 16 bytes (128-bit NEON registers)
#define SPC__HAS_LSE // ARM Large System Extensions (hw.optional.armv8_1_atomics: 1)

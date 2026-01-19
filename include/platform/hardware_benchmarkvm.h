/**
 * @file hardware_benchmarkvm.h
 * @brief Hardware configuration for SIGMOD contest benchmark VM.
 *
 * Intel Xeon E5-2680 v3: 24 threads, 256KB L2/core, 30MB shared L3, AVX2.
 * Activated via SPC__USE_BENCHMARKVM_HARDWARE.
 *
 * @note Critical for cache-aware partitioning in
 *       UnchainedHashtable::compute_num_partitions().
 * @see hardware.h for generic fallback.
 * @see hardware_darwin.h for Apple Silicon.
 */
#define SPC__X86_64
#define SPC__CPU_NAME "Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz"
#define SPC__CORE_COUNT 24
#define SPC__THREAD_COUNT 24
#define SPC__OS "Ubuntu 24.04.3 LTS"
#define SPC__KERNEL "Linux 6.8.0-90-generic x86_64"
#define SPC__NUMA_NODE_DRAM_MB 63488
#define SPC__SUPPORTS_AVX2

#define SPC__LEVEL1_ICACHE_SIZE 32768
#define SPC__LEVEL1_ICACHE_ASSOC 8
#define SPC__LEVEL1_ICACHE_LINESIZE 64

#define SPC__LEVEL1_DCACHE_SIZE 32768
#define SPC__LEVEL1_DCACHE_ASSOC 8
#define SPC__LEVEL1_DCACHE_LINESIZE 64

#define SPC__LEVEL2_CACHE_SIZE 262144
#define SPC__LEVEL2_CACHE_ASSOC 8
#define SPC__LEVEL2_CACHE_LINESIZE 64

#define SPC__LEVEL3_CACHE_SIZE 31457280
#define SPC__LEVEL3_CACHE_ASSOC 20
#define SPC__LEVEL3_CACHE_LINESIZE 64

#define SPC__LEVEL4_CACHE_SIZE 0
#define SPC__LEVEL4_CACHE_ASSOC
#define SPC__LEVEL4_CACHE_LINESIZE

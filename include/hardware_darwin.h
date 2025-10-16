// Hardware information for Apple M1 Mac
// Architecture from `uname -srm`: Darwin 25.0.0 arm64
#define SPC__AARCH64
// CPU information
#define SPC__CPU_NAME "Apple M1"
// M1 has 8 cores total (4 performance + 4 efficiency cores)
// All cores show as physical cores (no hyperthreading)
#define SPC__CORE_COUNT 8
#define SPC__THREAD_COUNT 8
#define SPC__NUMA_NODE_COUNT 1
#define SPC__NUMA_NODES_ACTIVE_IN_BENCHMARK 1
// Main memory per NUMA node (MB): 17179869184 bytes = 16384 MB
#define SPC__NUMA_NODE_DRAM_MB 16384
// OS version from `sw_vers`
#define SPC__OS "macOS 26.0"
// Kernel from `uname -srm`
#define SPC__KERNEL "Darwin 25.0.0 arm64"
// ARM: M1 supports NEON
#define SPC__SUPPORTS_NEON
// Cache information from `sysctl -a | grep cache`
// Note: M1 has asymmetric cores with different cache sizes:
// - Performance cores (perflevel0): L1I=192KB, L1D=128KB, L2=12MB
// - Efficiency cores (perflevel1): L1I=128KB, L1D=64KB, L2=4MB
// Using efficiency core values as baseline (hw.l1icachesize, hw.l1dcachesize, hw.l2cachesize)
#define SPC__LEVEL1_ICACHE_SIZE                 196608  // 128 KB
#define SPC__LEVEL1_ICACHE_ASSOC                8
#define SPC__LEVEL1_ICACHE_LINESIZE             128
#define SPC__LEVEL1_DCACHE_SIZE                 131072   // 64 KB
#define SPC__LEVEL1_DCACHE_ASSOC                8
#define SPC__LEVEL1_DCACHE_LINESIZE             128
#define SPC__LEVEL2_CACHE_SIZE                  12582912 // 4 MB
#define SPC__LEVEL2_CACHE_ASSOC                 16
#define SPC__LEVEL2_CACHE_LINESIZE              128
#define SPC__LEVEL3_CACHE_SIZE                  0
#define SPC__LEVEL3_CACHE_ASSOC                 0
#define SPC__LEVEL3_CACHE_LINESIZE              0
#define SPC__LEVEL4_CACHE_SIZE                  0
#define SPC__LEVEL4_CACHE_ASSOC                 0
#define SPC__LEVEL4_CACHE_LINESIZE              0

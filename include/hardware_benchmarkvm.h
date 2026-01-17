// Hardware information for benchmark VM.
// Architecture from `uname -srm`.
#define SPC__X86_64

// CPU from `/proc/cpuinfo`.
#define SPC__CPU_NAME "Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz"


// Actual hardware capability is 24 threads (12 cores x 2 HT).
#define SPC__CORE_COUNT 24 
#define SPC__THREAD_COUNT 24 

// Obtained from `lsb_release -a`.
#define SPC__OS "Ubuntu 24.04.3 LTS"

// Obtained from: `uname -srm`.
#define SPC__KERNEL "Linux 6.8.0-90-generic x86_64"

// Main memory (MB).
#define SPC__NUMA_NODE_DRAM_MB 63488 

// Intel: possible options are AVX, AVX2, and AVX512.
#define SPC__SUPPORTS_AVX2

// Cache information
// CORRECTED VALUES FOR HASWELL-EP (E5-2680 v3)

// L1 Instruction: 32KB, 8-way associative
#define SPC__LEVEL1_ICACHE_SIZE                 32768
#define SPC__LEVEL1_ICACHE_ASSOC                8
#define SPC__LEVEL1_ICACHE_LINESIZE             64

// L1 Data: 32KB, 8-way associative
#define SPC__LEVEL1_DCACHE_SIZE                 32768
#define SPC__LEVEL1_DCACHE_ASSOC                8
#define SPC__LEVEL1_DCACHE_LINESIZE             64

// L2 Cache: 256KB per core, 8-way associative
#define SPC__LEVEL2_CACHE_SIZE                  262144
#define SPC__LEVEL2_CACHE_ASSOC                 8
#define SPC__LEVEL2_CACHE_LINESIZE              64

// L3 Cache: 30MB shared, 20-way associative
#define SPC__LEVEL3_CACHE_SIZE                  31457280
#define SPC__LEVEL3_CACHE_ASSOC                 20
#define SPC__LEVEL3_CACHE_LINESIZE              64

#define SPC__LEVEL4_CACHE_SIZE                  0
#define SPC__LEVEL4_CACHE_ASSOC
#define SPC__LEVEL4_CACHE_LINESIZE

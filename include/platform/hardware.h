/**
 *
 * @file hardware.h
 *
 * @see hardware_darwin.h for Apple Silicon.
 * @brief Hardware configuration for AMD Ryzen 9 7950X3D.
 *
 **/
#pragma once

#define SPC__CORE_COUNT 16
#define SPC__THREAD_COUNT 32
#define SPC__LEVEL1_DCACHE_SIZE 524288
#define SPC__LEVEL2_CACHE_SIZE 16777216
#define SPC__LEVEL3_CACHE_SIZE 134217728
#define SPC__LEVEL1_DCACHE_LINESIZE 64
#ifndef SPC__NUMA_NODE_DRAM_MB
    #define SPC__NUMA_NODE_DRAM_MB 14336
#endif

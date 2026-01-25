/**
 *
 * @file hardware.h
 *
 * @see hardware_darwin.h for Apple Silicon.
 *
 **/
#pragma once

#define SPC__CORE_COUNT 6
#define SPC__THREAD_COUNT 6
#define SPC__LEVEL1_DCACHE_SIZE 32768
#define SPC__LEVEL2_CACHE_SIZE 1048576
#define SPC__LEVEL3_CACHE_SIZE 33554432
#define SPC__LEVEL1_DCACHE_LINESIZE 64
#ifndef SPC__NUMA_NODE_DRAM_MB
    #define SPC__NUMA_NODE_DRAM_MB 16384
#endif

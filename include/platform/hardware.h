/**
 * @file hardware.h
 * @brief Default hardware configuration constants.
 *
 * Fallback for generic x86-64 Linux when neither Darwin nor benchmarkvm
 * defined.
 *
 * @see hardware_darwin.h for Apple Silicon.
 * @see hardware_benchmarkvm.h for contest VM.
 */
#pragma once

#define SPC__CORE_COUNT 8
#define SPC__THREAD_COUNT 16
#define SPC__LEVEL1_DCACHE_SIZE 32768
#define SPC__LEVEL2_CACHE_SIZE 1048576
#define SPC__LEVEL3_CACHE_SIZE 33554432
#define SPC__LEVEL1_DCACHE_LINESIZE 64
#define SPC__NUMA_NODE_DRAM_MB 16384

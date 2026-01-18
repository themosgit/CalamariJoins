/**
 * @file hardware.h
 * @brief Default hardware configuration constants.
 *
 * Fallback platform configuration used when neither Darwin (Apple Silicon)
 * nor SPC__USE_BENCHMARKVM_HARDWARE is defined. Provides conservative
 * defaults for generic x86-64 Linux systems.
 *
 * ### Constants defined:
 * - SPC__CORE_COUNT: Physical core count
 * - SPC__THREAD_COUNT: Logical thread count (including SMT)
 * - SPC__LEVEL1_DCACHE_SIZE: L1 data cache size in bytes
 * - SPC__LEVEL2_CACHE_SIZE: L2 cache size in bytes
 * - SPC__LEVEL3_CACHE_SIZE: L3 cache size in bytes (0 if none)
 * - SPC__LEVEL1_DCACHE_LINESIZE: Cache line size in bytes
 *
 * @see hardware_darwin.h for Apple Silicon configuration
 * @see hardware_benchmarkvm.h for contest benchmark VM configuration
 */
#pragma once

#define SPC__CORE_COUNT 8
#define SPC__THREAD_COUNT 16
#define SPC__LEVEL1_DCACHE_SIZE 32768
#define SPC__LEVEL2_CACHE_SIZE 1048576
#define SPC__LEVEL3_CACHE_SIZE 33554432
#define SPC__LEVEL1_DCACHE_LINESIZE 64

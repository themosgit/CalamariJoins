/**
 * @file simd_compare.h
 * @brief SIMD comparison helpers for nested loop join operations.
 *
 * Platform-specific (AVX2/NEON) vectorized comparison primitives.
 * All functions are inline for zero overhead.
 *
 * @see nested_loop.h
 */
#pragma once

#include <cstdint>
#include <data_model/intermediate.h>
#include <join_execution/match_collector.h>

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace Contest::join::simd {

/**
 * @brief Compare single probe value against 8 register-resident build values.
 *
 * Used by nested loop join where build side fits in SIMD registers.
 * Build values must be 32-byte aligned for AVX2 aligned load.
 *
 * @param probe_id    Row ID of probe tuple
 * @param probe_val   Probe key value
 * @param build_vals  Pointer to 8 aligned build values (alignas(32))
 * @param build_ids   Pointer to build row IDs
 * @param build_count Actual number of valid build values (<=8)
 * @param buffer      Thread-local match buffer
 */
inline void eq_scan_build(uint32_t probe_id, int32_t probe_val,
                          const int32_t *build_vals, const uint32_t *build_ids,
                          size_t build_count, ThreadLocalMatchBuffer &buffer) {
#if defined(__x86_64__) && defined(__AVX2__)
    __m256i build_reg =
        _mm256_load_si256(reinterpret_cast<const __m256i *>(build_vals));
    const int valid_mask = (1 << build_count) - 1;

    __m256i probe_reg = _mm256_set1_epi32(probe_val);
    __m256i cmp = _mm256_cmpeq_epi32(probe_reg, build_reg);
    int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) & valid_mask;

    while (mask) {
        int idx = __builtin_ctz(mask);
        buffer.add_match(build_ids[idx], probe_id);
        mask &= mask - 1;
    }

#elif defined(__aarch64__)
    int32x4_t build_lo = vld1q_s32(build_vals);
    int32x4_t build_hi = vld1q_s32(build_vals + 4);
    const uint32_t valid_mask = (1u << build_count) - 1;

    static constexpr uint32_t mask_bits_lo[4] = {1, 2, 4, 8};
    static constexpr uint32_t mask_bits_hi[4] = {16, 32, 64, 128};
    const uint32x4_t bit_mask_lo = vld1q_u32(mask_bits_lo);
    const uint32x4_t bit_mask_hi = vld1q_u32(mask_bits_hi);

    int32x4_t probe_reg = vdupq_n_s32(probe_val);
    uint32x4_t cmp_lo = vceqq_s32(probe_reg, build_lo);
    uint32x4_t cmp_hi = vceqq_s32(probe_reg, build_hi);
    uint32_t mask = vaddvq_u32(vandq_u32(cmp_lo, bit_mask_lo)) |
                    vaddvq_u32(vandq_u32(cmp_hi, bit_mask_hi));
    mask &= valid_mask;

    while (mask) {
        int idx = __builtin_ctz(mask);
        buffer.add_match(build_ids[idx], probe_id);
        mask &= mask - 1;
    }

#else
    for (size_t k = 0; k < build_count; ++k) {
        if (build_vals[k] == probe_val) {
            buffer.add_match(build_ids[k], probe_id);
        }
    }
#endif
}

/**
 * @brief Batch compare columnar probe values against all build values.
 *
 * Processes 8 (AVX2) or 4 (NEON) probe values per iteration.
 * Returns count of rows processed; caller handles remainder with scalar.
 *
 * @param data        Pointer to columnar int32 data
 * @param num_rows    Total rows in page
 * @param base_row_id Starting row ID
 * @param build_vals  Pointer to 8 aligned build values (alignas(32))
 * @param build_ids   Pointer to build row IDs
 * @param build_count Actual build count (<=8)
 * @param buffer      Thread-local match buffer
 * @return Number of rows processed with SIMD
 */
inline uint16_t eq_batch_columnar(const int32_t *data, uint16_t num_rows,
                                  uint32_t base_row_id,
                                  const int32_t *build_vals,
                                  const uint32_t *build_ids, size_t build_count,
                                  ThreadLocalMatchBuffer &buffer) {
    uint16_t j = 0;

#if defined(__x86_64__) && defined(__AVX2__)
    for (; j + 8 <= num_rows; j += 8) {
        __m256i probe_batch =
            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(data + j));

        for (size_t k = 0; k < build_count; ++k) {
            __m256i build_val = _mm256_set1_epi32(build_vals[k]);
            __m256i cmp = _mm256_cmpeq_epi32(probe_batch, build_val);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp));

            while (mask) {
                int idx = __builtin_ctz(mask);
                buffer.add_match(build_ids[k], base_row_id + j + idx);
                mask &= mask - 1;
            }
        }
    }

#elif defined(__aarch64__)
    static constexpr uint32_t mask_bits[4] = {1, 2, 4, 8};
    const uint32x4_t bit_mask = vld1q_u32(mask_bits);

    for (; j + 4 <= num_rows; j += 4) {
        int32x4_t probe_batch = vld1q_s32(data + j);

        for (size_t k = 0; k < build_count; ++k) {
            int32x4_t build_val = vdupq_n_s32(build_vals[k]);
            uint32x4_t cmp = vceqq_s32(probe_batch, build_val);
            uint32_t mask = vaddvq_u32(vandq_u32(cmp, bit_mask));

            while (mask) {
                int idx = __builtin_ctz(mask);
                buffer.add_match(build_ids[k], base_row_id + j + idx);
                mask &= mask - 1;
            }
        }
    }
#endif

    return j;
}

/// SIMD batch size for intermediate processing
#if defined(__x86_64__) && defined(__AVX2__)
inline constexpr size_t INTERMEDIATE_BATCH_SIZE = 8;
#elif defined(__aarch64__)
inline constexpr size_t INTERMEDIATE_BATCH_SIZE = 4;
#else
inline constexpr size_t INTERMEDIATE_BATCH_SIZE = 0; // No SIMD batching
#endif

/**
 * @brief Batch compare intermediate values against all build values.
 *
 * Handles NULL values (mema::value_t::NULL_VALUE). Caller ensures all
 * values are on same page. Processes 8 (AVX2) or 4 (NEON) values.
 *
 * @param vals        Pointer to intermediate int32 values (same page)
 * @param base_idx    Starting index in intermediate column
 * @param build_vals  Pointer to 8 aligned build values (alignas(32))
 * @param build_ids   Pointer to build row IDs
 * @param build_count Actual build count (<=8)
 * @param buffer      Thread-local match buffer
 */
inline void eq_batch_intermediate(const int32_t *vals, size_t base_idx,
                                  const int32_t *build_vals,
                                  const uint32_t *build_ids, size_t build_count,
                                  ThreadLocalMatchBuffer &buffer) {
#if defined(__x86_64__) && defined(__AVX2__)
    __m256i probe_batch =
        _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vals));

    // Check for NULLs
    __m256i null_val = _mm256_set1_epi32(mema::value_t::NULL_VALUE);
    __m256i null_mask = _mm256_cmpeq_epi32(probe_batch, null_val);
    int nulls = _mm256_movemask_ps(_mm256_castsi256_ps(null_mask));
    int valid = ~nulls & 0xFF;

    for (size_t k = 0; k < build_count; ++k) {
        __m256i build_val = _mm256_set1_epi32(build_vals[k]);
        __m256i cmp = _mm256_cmpeq_epi32(probe_batch, build_val);
        int mask = _mm256_movemask_ps(_mm256_castsi256_ps(cmp)) & valid;

        while (mask) {
            int idx = __builtin_ctz(mask);
            buffer.add_match(build_ids[k],
                             static_cast<uint32_t>(base_idx + idx));
            mask &= mask - 1;
        }
    }

#elif defined(__aarch64__)
    int32x4_t probe_batch = vld1q_s32(vals);
    static constexpr uint32_t mask_bits[4] = {1, 2, 4, 8};
    const uint32x4_t bit_mask = vld1q_u32(mask_bits);

    // Check for NULLs
    int32x4_t null_val = vdupq_n_s32(mema::value_t::NULL_VALUE);
    uint32x4_t null_cmp = vceqq_s32(probe_batch, null_val);
    uint32_t nulls = vaddvq_u32(vandq_u32(null_cmp, bit_mask));
    uint32_t valid = ~nulls & 0xF;

    for (size_t k = 0; k < build_count; ++k) {
        int32x4_t build_val = vdupq_n_s32(build_vals[k]);
        uint32x4_t cmp = vceqq_s32(probe_batch, build_val);
        uint32_t mask = vaddvq_u32(vandq_u32(cmp, bit_mask)) & valid;

        while (mask) {
            int idx = __builtin_ctz(mask);
            buffer.add_match(build_ids[k],
                             static_cast<uint32_t>(base_idx + idx));
            mask &= mask - 1;
        }
    }
#else
    // Scalar fallback - should not be called when INTERMEDIATE_BATCH_SIZE == 0
    (void)vals;
    (void)base_idx;
    (void)build_vals;
    (void)build_ids;
    (void)build_count;
    (void)buffer;
#endif
}

} // namespace Contest::join::simd

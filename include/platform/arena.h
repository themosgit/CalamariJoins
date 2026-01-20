/**
 * @file arena.h
 * @brief Global arena allocator with per-chunk-type regions.
 *
 * Provides lock-free per-thread allocation via bump pointers.
 * Single mmap at startup, divided into regions with different page policies.
 * Memory freed only on reset_all() between queries.
 *
 * @see worker_pool.h for SPC__THREAD_COUNT.
 * @see execute.cpp for reset_all() call pattern.
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <new>
#include <sys/mman.h>

#if defined(__APPLE__) && defined(__aarch64__)
#include <platform/hardware_darwin.h>
#elif defined(SPC__USE_BENCHMARKVM_HARDWARE)
#include <platform/hardware_benchmarkvm.h>
#else
#include <platform/hardware.h>
#endif

namespace Contest::platform {

// ============================================================================
// Constants
// ============================================================================

static constexpr size_t PAGE_4KB = 4096;
static constexpr size_t PAGE_2MB = 2 * 1024 * 1024;

// ============================================================================
// Chunk Types (4 types, OUTPUT_PAGE removed - Column pages use existing path)
// ============================================================================

/**
 * @brief Chunk type enumeration for arena regions.
 */
enum class ChunkType : uint8_t {
    HASH_CHUNK = 0,  ///< 4KB  - hash table partition chunks
    IR_PAGE = 1,     ///< 16KB - intermediate result pages
    INDEX_CHUNK = 2, ///< 32KB - match collector index chunks
    GENERAL = 3,     ///< Variable - misc allocations

    NUM_TYPES = 4
};

// ============================================================================
// Compile-time Chunk Sizes
// ============================================================================

/**
 * @brief Compile-time chunk size lookup.
 * @tparam CT ChunkType to get size for.
 */
template <ChunkType CT> struct ChunkSize;
template <> struct ChunkSize<ChunkType::HASH_CHUNK> {
    static constexpr size_t value = 4096;
};
template <> struct ChunkSize<ChunkType::IR_PAGE> {
    static constexpr size_t value = 16384;
};
template <> struct ChunkSize<ChunkType::INDEX_CHUNK> {
    static constexpr size_t value = 32768;
};
template <> struct ChunkSize<ChunkType::GENERAL> {
    static constexpr size_t value = 0;
};

/// Runtime chunk size array indexed by ChunkType.
inline constexpr size_t CHUNK_SIZES[] = {4096, 16384, 32768, 0};

// ============================================================================
// Page Policies
// ============================================================================

/**
 * @brief MADVISE page policy per region.
 */
enum class PagePolicy : uint8_t {
    SMALL_PAGES, ///< MADV_NOHUGEPAGE (4KB base pages)
    HUGE_PAGES,  ///< MADV_HUGEPAGE (2MB THP)
};

/// Page policy per region. HASH_CHUNK uses small pages (4KB chunks fit poorly
/// in huge pages); others use huge pages for TLB efficiency.
inline constexpr PagePolicy REGION_PAGE_POLICY[] = {
    PagePolicy::SMALL_PAGES, // HASH_CHUNK
    PagePolicy::HUGE_PAGES,  // IR_PAGE
    PagePolicy::HUGE_PAGES,  // INDEX_CHUNK
    PagePolicy::HUGE_PAGES,  // GENERAL
};

// ============================================================================
// Region Configuration
// ============================================================================

/**
 * @brief Region size configuration based on available DRAM.
 *
 * Uses 75% of SPC__NUMA_NODE_DRAM_MB, divided equally (25%) among 4 regions.
 */
struct RegionConfig {
    size_t total_arena_bytes;

    RegionConfig() {
        // 75% of available DRAM
        total_arena_bytes = static_cast<size_t>(SPC__NUMA_NODE_DRAM_MB) *
                            1024ULL * 1024ULL * 3ULL / 4ULL;
    }

    /// Get total size for a region (25% each).
    size_t get(ChunkType /*ct*/) const { return total_arena_bytes / 4; }

    /// Get total arena size.
    size_t total() const { return total_arena_bytes; }

    /// Get per-thread slice size for a region.
    size_t per_thread(ChunkType ct) const {
        return get(ct) / SPC__THREAD_COUNT;
    }
};

/// Default region configuration.
inline const RegionConfig DEFAULT_CONFIG{};

// ============================================================================
// Region Slice (per-thread bump allocator)
// ============================================================================

/**
 * @brief Per-thread slice within a region.
 *
 * Lock-free bump allocation: offset++ is plain integer increment.
 * No synchronization needed (single-writer per slice).
 */
struct RegionSlice {
    char *base = nullptr;  ///< Start of this thread's slice.
    size_t capacity = 0;   ///< Total bytes in slice.
    size_t offset = 0;     ///< Current allocation offset.
    size_t chunk_size = 0; ///< Fixed chunk size (0 for GENERAL).

    /// Initialize slice with base pointer, capacity, and chunk size.
    void init(char *b, size_t cap, size_t cs) {
        base = b;
        capacity = cap;
        offset = 0;
        chunk_size = cs;
    }

    /**
     * @brief Allocate single chunk.
     * @return Pointer to chunk, or nullptr if exhausted.
     */
    void *allocate() noexcept {
        if (chunk_size == 0)
            return nullptr; // Use alloc_bytes for GENERAL
        if (offset + chunk_size > capacity)
            return nullptr;
        void *ptr = base + offset;
        offset += chunk_size;
        return ptr;
    }

    /**
     * @brief Allocate N contiguous chunks.
     * @param count Number of chunks.
     * @return Pointer to first chunk, or nullptr if exhausted.
     */
    void *allocate_n(size_t count) noexcept {
        size_t size = chunk_size * count;
        if (offset + size > capacity)
            return nullptr;
        void *ptr = base + offset;
        offset += size;
        return ptr;
    }

    /**
     * @brief Allocate arbitrary bytes with alignment.
     * @param size Bytes to allocate.
     * @param align Alignment (must be power of 2).
     * @return Aligned pointer, or nullptr if exhausted.
     */
    void *alloc_bytes(size_t size, size_t align = 16) noexcept {
        // align must be power of 2
        size_t aligned = (offset + align - 1) & ~(align - 1);
        if (aligned + size > capacity)
            return nullptr;
        void *ptr = base + aligned;
        offset = aligned + size;
        return ptr;
    }

    /// Reset slice to beginning (O(1)).
    void reset() noexcept { offset = 0; }

    /// Bytes currently used.
    size_t used() const noexcept { return offset; }

    /// Bytes remaining.
    size_t remaining() const noexcept { return capacity - offset; }
};

// ============================================================================
// Thread Arena
// ============================================================================

/**
 * @brief Thread-local arena providing access to all region slices.
 *
 * Each thread gets one ThreadArena with isolated RegionSlice objects.
 * No synchronization needed for allocation.
 */
class ThreadArena {
    RegionSlice slices_[static_cast<size_t>(ChunkType::NUM_TYPES)];

  public:
    /// Initialize a slice for a chunk type.
    void set_slice(ChunkType ct, char *base, size_t cap, size_t chunk_size) {
        slices_[static_cast<size_t>(ct)].init(base, cap, chunk_size);
    }

    // ========== Typed Chunk Allocation ==========

    /**
     * @brief Allocate single chunk of compile-time known type.
     * @tparam CT ChunkType to allocate from.
     * @return Pointer to chunk.
     * @throws std::bad_alloc if exhausted.
     */
    template <ChunkType CT> void *alloc_chunk() {
        void *ptr = slices_[static_cast<size_t>(CT)].allocate();
        if (!ptr) [[unlikely]]
            throw std::bad_alloc();
        return ptr;
    }

    /**
     * @brief Allocate N contiguous chunks of compile-time known type.
     * @tparam CT ChunkType to allocate from.
     * @param count Number of chunks.
     * @return Pointer to first chunk.
     * @throws std::bad_alloc if exhausted.
     */
    template <ChunkType CT> void *alloc_chunks(size_t count) {
        void *ptr = slices_[static_cast<size_t>(CT)].allocate_n(count);
        if (!ptr) [[unlikely]]
            throw std::bad_alloc();
        return ptr;
    }

    /**
     * @brief Allocate single chunk with runtime type selection.
     * @param ct ChunkType to allocate from.
     * @return Pointer to chunk.
     * @throws std::bad_alloc if exhausted.
     */
    void *alloc_chunk(ChunkType ct) {
        void *ptr = slices_[static_cast<size_t>(ct)].allocate();
        if (!ptr) [[unlikely]]
            throw std::bad_alloc();
        return ptr;
    }

    // ========== General Allocation ==========

    /**
     * @brief Allocate arbitrary bytes from GENERAL region.
     * @param size Bytes to allocate.
     * @param align Alignment (must be power of 2, default 16).
     * @return Aligned pointer.
     * @throws std::bad_alloc if exhausted.
     */
    void *alloc(size_t size, size_t align = 16) {
        void *ptr =
            slices_[static_cast<size_t>(ChunkType::GENERAL)].alloc_bytes(size,
                                                                         align);
        if (!ptr) [[unlikely]]
            throw std::bad_alloc();
        return ptr;
    }

    /**
     * @brief Allocate array of T from GENERAL region.
     * @tparam T Element type.
     * @param count Number of elements.
     * @return Pointer to array.
     */
    template <typename T> T *alloc_array(size_t count) {
        return static_cast<T *>(alloc(sizeof(T) * count, alignof(T)));
    }

    /**
     * @brief Allocate and construct object in GENERAL region.
     * @tparam T Object type.
     * @tparam Args Constructor argument types.
     * @param args Constructor arguments.
     * @return Pointer to constructed object.
     */
    template <typename T, typename... Args> T *alloc_object(Args &&...args) {
        void *ptr = alloc(sizeof(T), alignof(T));
        return new (ptr) T(std::forward<Args>(args)...);
    }

    // ========== Lifecycle ==========

    /// Direct access to a region slice.
    RegionSlice &get_slice(ChunkType ct) {
        return slices_[static_cast<size_t>(ct)];
    }

    /// Reset all slices to beginning.
    void reset() noexcept {
        for (auto &s : slices_)
            s.reset();
    }

    /// Total bytes used across all slices.
    size_t total_used() const noexcept {
        size_t sum = 0;
        for (const auto &s : slices_)
            sum += s.used();
        return sum;
    }
};

// ============================================================================
// Region Allocator (typed wrapper)
// ============================================================================

/**
 * @brief Typed region allocator for a specific chunk type.
 * @tparam CT ChunkType to allocate from.
 * @tparam T Type to allocate (must fit in chunk).
 */
template <ChunkType CT, typename T> class RegionAllocator {
    static_assert(CT == ChunkType::GENERAL || sizeof(T) <= ChunkSize<CT>::value,
                  "Type too large for chunk type");
    ThreadArena *arena_;

  public:
    explicit RegionAllocator(ThreadArena &arena) : arena_(&arena) {}

    /// Allocate single T.
    T *alloc() { return static_cast<T *>(arena_->alloc_chunk<CT>()); }

    /// Allocate N contiguous T.
    T *alloc_n(size_t n) {
        return static_cast<T *>(arena_->alloc_chunks<CT>(n));
    }
};

// ============================================================================
// Arena Manager (global)
// ============================================================================

/**
 * @brief Global arena manager.
 *
 * Allocates single large mmap at construction, divides into regions,
 * applies per-region madvise policies, divides each region among threads.
 */
class ArenaManager {
    void *memory_base_ = nullptr;
    size_t total_size_ = 0;
    ThreadArena thread_arenas_[SPC__THREAD_COUNT];
    void *region_bases_[static_cast<size_t>(ChunkType::NUM_TYPES)] = {};

  public:
    explicit ArenaManager(const RegionConfig &config = DEFAULT_CONFIG) {
        // Calculate total with 2MB alignment per region
        total_size_ = 0;
        for (size_t i = 0; i < static_cast<size_t>(ChunkType::NUM_TYPES); ++i) {
            size_t region_size = config.get(static_cast<ChunkType>(i));
            size_t aligned = (region_size + PAGE_2MB - 1) & ~(PAGE_2MB - 1);
            total_size_ += aligned;
        }

        // Single large mmap
        memory_base_ = mmap(nullptr, total_size_, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory_base_ == MAP_FAILED)
            throw std::bad_alloc();

        // Initialize regions
        char *ptr = static_cast<char *>(memory_base_);

        for (size_t r = 0; r < static_cast<size_t>(ChunkType::NUM_TYPES); ++r) {
            ChunkType ct = static_cast<ChunkType>(r);
            size_t region_size =
                (config.get(ct) + PAGE_2MB - 1) & ~(PAGE_2MB - 1);
            size_t per_thread = region_size / SPC__THREAD_COUNT;
            size_t chunk_size = CHUNK_SIZES[r];

            region_bases_[r] = ptr;

// Apply madvise on Linux only
#if defined(__linux__)
            if (REGION_PAGE_POLICY[r] == PagePolicy::HUGE_PAGES) {
                madvise(ptr, region_size, MADV_HUGEPAGE);
            } else {
                madvise(ptr, region_size, MADV_NOHUGEPAGE);
            }
#endif

            // Divide region among threads
            for (int t = 0; t < SPC__THREAD_COUNT; ++t) {
                thread_arenas_[t].set_slice(ct, ptr + t * per_thread,
                                            per_thread, chunk_size);
            }

            ptr += region_size;
        }
    }

    ~ArenaManager() {
        if (memory_base_ && memory_base_ != MAP_FAILED)
            munmap(memory_base_, total_size_);
    }

    ArenaManager(const ArenaManager &) = delete;
    ArenaManager &operator=(const ArenaManager &) = delete;

    /// Get thread arena by thread ID.
    ThreadArena &get(size_t thread_id) noexcept {
        return thread_arenas_[thread_id];
    }

    /// Reset all thread arenas (call between queries).
    void reset_all() noexcept {
        for (int t = 0; t < SPC__THREAD_COUNT; ++t)
            thread_arenas_[t].reset();
    }

    /// Get base pointer for a region (for debugging).
    void *region_base(ChunkType ct) const noexcept {
        return region_bases_[static_cast<size_t>(ct)];
    }

    /// Get total arena size.
    size_t total_size() const noexcept { return total_size_; }
};

// ============================================================================
// Global Instance and Helper
// ============================================================================

/// Global arena manager instance (initialized at program startup).
inline ArenaManager arena_manager;

/// Get thread arena by thread ID.
inline ThreadArena &get_arena(size_t thread_id) {
    return arena_manager.get(thread_id);
}

} // namespace Contest::platform

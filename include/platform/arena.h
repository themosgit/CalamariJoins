/**
 * @file arena.h
 * @brief Global arena allocator with unified per-thread allocation.
 *
 * Provides lock-free per-thread allocation via bump pointers.
 * Single mmap at startup, divided equally among threads.
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
 // Arena Configuration
 // ============================================================================
 
 /**
  * @brief Arena size configuration based on available DRAM.
  *
  * Uses 75% of SPC__NUMA_NODE_DRAM_MB for the arena.
  */
 struct ArenaConfig {
     size_t total_arena_bytes;
 
     ArenaConfig() {
         // 75% of available DRAM
         total_arena_bytes = static_cast<size_t>(SPC__NUMA_NODE_DRAM_MB) *
                             1024ULL * 1024ULL * 3ULL / 4ULL;
     }
 
     /// Get total arena size.
     size_t total() const { return total_arena_bytes; }
 
     /// Get per-thread slice size.
     size_t per_thread() const {
         return total_arena_bytes / SPC__THREAD_COUNT;
     }
 };
 
 /// Default arena configuration.
 inline const ArenaConfig DEFAULT_CONFIG{};
 
 // ============================================================================
 // Arena Slice (per-thread bump allocator)
 // ============================================================================
 
 /**
  * @brief Per-thread slice of the arena.
  *
  * Lock-free bump allocation: offset++ is plain integer increment.
  * No synchronization needed (single-writer per slice).
  */
 struct ArenaSlice {
     char *base = nullptr;  ///< Start of this thread's slice.
     size_t capacity = 0;   ///< Total bytes in slice.
     size_t offset = 0;     ///< Current allocation offset.
 
     /// Initialize slice with base pointer and capacity.
     void init(char *b, size_t cap) {
         base = b;
         capacity = cap;
         offset = 0;
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
  * @brief Thread-local arena with unified allocation.
  *
  * Each thread gets one ThreadArena with a single ArenaSlice.
  * No synchronization needed for allocation.
  */
 class ThreadArena {
     ArenaSlice slice_;
 
   public:
     /// Initialize the arena slice.
     void init(char *base, size_t capacity) {
         slice_.init(base, capacity);
     }
 
     // ========== Typed Chunk Allocation ==========
 
     /**
      * @brief Allocate single chunk of compile-time known type.
      * @tparam CT ChunkType to allocate from.
      * @return Pointer to chunk.
      * @throws std::bad_alloc if exhausted.
      */
     template <ChunkType CT> void *alloc_chunk() {
         static constexpr size_t sz = ChunkSize<CT>::value;
         static constexpr size_t align = (sz >= 4096) ? 4096 : 64;
         void *ptr = slice_.alloc_bytes(sz, align);
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
         static constexpr size_t sz = ChunkSize<CT>::value;
         static constexpr size_t align = (sz >= 4096) ? 4096 : 64;
         void *ptr = slice_.alloc_bytes(sz * count, align);
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
         size_t sz = CHUNK_SIZES[static_cast<size_t>(ct)];
         size_t align = (sz >= 4096) ? 4096 : 64;
         void *ptr = slice_.alloc_bytes(sz, align);
         if (!ptr) [[unlikely]]
             throw std::bad_alloc();
         return ptr;
     }
 
     // ========== General Allocation ==========
 
     /**
      * @brief Allocate arbitrary bytes from the arena.
      * @param size Bytes to allocate.
      * @param align Alignment (must be power of 2, default 16).
      * @return Aligned pointer.
      * @throws std::bad_alloc if exhausted.
      */
     void *alloc(size_t size, size_t align = 16) {
         void *ptr = slice_.alloc_bytes(size, align);
         if (!ptr) [[unlikely]]
             throw std::bad_alloc();
         return ptr;
     }
 
     /**
      * @brief Allocate array of T from the arena.
      * @tparam T Element type.
      * @param count Number of elements.
      * @return Pointer to array.
      */
     template <typename T> T *alloc_array(size_t count) {
         return static_cast<T *>(alloc(sizeof(T) * count, alignof(T)));
     }
 
     /**
      * @brief Allocate and construct object in the arena.
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
 
     /// Direct access to the arena slice.
     ArenaSlice &get_slice() { return slice_; }
 
     /// Reset slice to beginning.
     void reset() noexcept { slice_.reset(); }
 
     /// Total bytes used.
     size_t total_used() const noexcept { return slice_.used(); }
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
  * Allocates single large mmap at construction, divides equally among threads.
  * Uses huge pages for TLB efficiency.
  */
 class ArenaManager {
     void *memory_base_ = nullptr;
     size_t total_size_ = 0;
     ThreadArena thread_arenas_[SPC__THREAD_COUNT];
 
   public:
     explicit ArenaManager(const ArenaConfig &config = DEFAULT_CONFIG) {
         // Align total to 2MB boundary
         total_size_ = (config.total() + PAGE_2MB - 1) & ~(PAGE_2MB - 1);
 
         // Single large mmap
         memory_base_ = mmap(nullptr, total_size_, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
         if (memory_base_ == MAP_FAILED)
             throw std::bad_alloc();
 
 // Apply madvise on Linux only - use huge pages for entire arena
 #if defined(__linux__)
         madvise(memory_base_, total_size_, MADV_HUGEPAGE);
 #endif
 
         // Divide equally among threads
         size_t per_thread = total_size_ / SPC__THREAD_COUNT;
         char *ptr = static_cast<char *>(memory_base_);
         for (int t = 0; t < SPC__THREAD_COUNT; ++t) {
             thread_arenas_[t].init(ptr + t * per_thread, per_thread);
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
 
     /// Get total arena size.
     size_t total_size() const noexcept { return total_size_; }
 };
 
 // ============================================================================
 // Global Instance and Helper
 // ============================================================================
 
 /// Global arena manager instance (inline global, constructed at program startup).
 inline ArenaManager g_arena_manager{};
 
 /// Get thread arena by thread ID.
 inline ThreadArena &get_arena(size_t thread_id) {
     return g_arena_manager.get(thread_id);
 }
 
 } // namespace Contest::platform
 
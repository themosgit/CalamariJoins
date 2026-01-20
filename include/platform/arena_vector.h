/**
 * @file arena_vector.h
 * @brief Arena-backed vector for trivially copyable types.
 *
 * Does not free memory on clear/destruction (arena handles lifecycle).
 * Uses 1.5x growth factor for efficient reallocation.
 *
 * @see arena.h for ThreadArena.
 */
#pragma once

#include <algorithm>
#include <cstring>
#include <platform/arena.h>
#include <type_traits>

namespace Contest::platform {

/**
 * @brief Arena-backed vector for trivially copyable types.
 *
 * Memory is allocated from ThreadArena's GENERAL region.
 * Memory is never freed individually - arena reset handles cleanup.
 * Growth uses 1.5x factor for balance between memory usage and reallocation
 * count.
 *
 * @tparam T Element type (must be trivially copyable).
 */
template <typename T> class ArenaVector {
    static_assert(std::is_trivially_copyable_v<T>,
                  "ArenaVector requires trivially copyable type");

    T *data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
    ThreadArena *arena_ = nullptr;

    /// Calculate next capacity using 1.5x growth, minimum 16 elements.
    static size_t next_capacity(size_t required) {
        size_t cap = 16; // Minimum capacity
        while (cap < required) {
            cap = cap + cap / 2; // 1.5x growth
        }
        return cap;
    }

  public:
    ArenaVector() = default;

    /// Construct with arena reference.
    explicit ArenaVector(ThreadArena &arena) : arena_(&arena) {}

    /// Move constructor - transfers ownership, leaves source empty.
    ArenaVector(ArenaVector &&other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_),
          arena_(other.arena_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    /// Move assignment - old data not freed (arena handles it).
    ArenaVector &operator=(ArenaVector &&other) noexcept {
        if (this != &other) {
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            arena_ = other.arena_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    /// Deleted copy constructor - arena memory shouldn't be shared.
    ArenaVector(const ArenaVector &) = delete;

    /// Deleted copy assignment.
    ArenaVector &operator=(const ArenaVector &) = delete;

    /// Set arena after default construction.
    void set_arena(ThreadArena &arena) { arena_ = &arena; }

    /**
     * @brief Reserve capacity for at least n elements.
     *
     * If current capacity is sufficient, does nothing.
     * Otherwise allocates new storage and copies existing elements.
     * Old storage is not freed (arena handles lifecycle).
     *
     * @param n Minimum capacity.
     */
    void reserve(size_t n) {
        if (n <= capacity_)
            return;
        size_t new_cap = next_capacity(n);
        T *new_data =
            static_cast<T *>(arena_->alloc(new_cap * sizeof(T), alignof(T)));
        if (data_ && size_ > 0) {
            std::memcpy(new_data, data_, size_ * sizeof(T));
        }
        data_ = new_data;
        capacity_ = new_cap;
    }

    /**
     * @brief Resize to n elements.
     *
     * If n > capacity, reserves new storage.
     * New elements are not initialized (trivially copyable).
     *
     * @param n New size.
     */
    void resize(size_t n) {
        if (n > capacity_)
            reserve(n);
        size_ = n;
    }

    /**
     * @brief Append element to end.
     *
     * Reserves more capacity if needed.
     *
     * @param v Value to append.
     */
    void push_back(const T &v) {
        if (size_ >= capacity_)
            reserve(size_ + 1);
        data_[size_++] = v;
    }

    /// Clear contents (capacity retained, memory not freed).
    void clear() { size_ = 0; }

    // ========== Element Access ==========

    T &operator[](size_t i) { return data_[i]; }
    const T &operator[](size_t i) const { return data_[i]; }

    T *data() { return data_; }
    const T *data() const { return data_; }

    // ========== Capacity ==========

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }

    // ========== Iterators ==========

    T *begin() { return data_; }
    T *end() { return data_ + size_; }
    const T *begin() const { return data_; }
    const T *end() const { return data_ + size_; }
};

} // namespace Contest::platform

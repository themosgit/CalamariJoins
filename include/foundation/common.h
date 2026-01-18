/**
 * @file common.h
 * @brief Common utilities: hashing, file I/O, and data structures.
 *
 * Provides foundational utilities used throughout the codebase:
 * - **hash_combine**: Combine hash values for composite keys
 * - **File**: RAII wrapper for FILE* handles
 * - **DSU**: Disjoint Set Union (Union-Find) data structure
 * - **unreachable()**: Mark unreachable code paths for optimization
 */

#pragma once

#include <filesystem>
#include <numeric>
#include <vector>

#include <cstdint>
#include <cstdlib>

/**
 * @namespace detail
 * @brief Implementation details for hash_combine.
 *
 * Contains platform-specific hash combining functions based on MurmurHash
 * mixing constants. Not intended for direct use.
 */
namespace detail {
/**
 * @brief Rotate left for 32-bit integers.
 * @param x Value to rotate.
 * @param bits Number of bits to rotate.
 * @return Rotated value.
 */
inline uint32_t rotl32(uint32_t x, uint8_t bits) {
    return (x << bits) | (x >> (32 - bits));
}

/**
 * @brief Combine a 32-bit hash value with a new key (MurmurHash3 style).
 * @param h1 Accumulator hash (modified in place).
 * @param k1 New key to mix in.
 */
inline void hash_combine_impl(uint32_t &h1, uint32_t k1) {
    constexpr uint32_t c1 = 0xcc9e2d51u;
    constexpr uint32_t c2 = 0x1b873593u;

    k1 *= c1;
    k1 = rotl32(k1, 15);
    k1 *= c2;

    h1 ^= k1;
    h1 = rotl32(h1, 13);
    h1 = h1 * 5u + 0xe6546b64u;
}

/**
 * @brief Combine a 64-bit hash value with a new key (MurmurHash2 style).
 * @param h Accumulator hash (modified in place).
 * @param k New key to mix in.
 */
inline void hash_combine_impl(uint64_t &h, uint64_t k) {
    constexpr uint64_t m = UINT64_C(0xc6a4a7935bd1e995);
    constexpr int r = 47;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
    h += 0xe6546b64;
}
} // namespace detail

/**
 * @brief Combine a hash seed with a new hash value.
 *
 * Uses MurmurHash-style mixing to combine hash values for composite keys.
 * Automatically selects 32-bit or 64-bit implementation based on platform.
 *
 * @param seed Accumulator hash value (modified in place).
 * @param k New hash value to combine.
 *
 * ### Example
 * @code
 * size_t hash = 0;
 * hash_combine(hash, std::hash<std::string>{}(name));
 * hash_combine(hash, std::hash<int>{}(id));
 * @endcode
 */
inline void hash_combine(std::size_t &seed, std::size_t k) {
    if constexpr (sizeof(std::size_t) == 4) {
        uint32_t h = static_cast<uint32_t>(seed);
        detail::hash_combine_impl(h, static_cast<uint32_t>(k));
        seed = h;
    } else if constexpr (sizeof(std::size_t) == 8) {
        uint64_t h = static_cast<uint64_t>(seed);
        detail::hash_combine_impl(h, static_cast<uint64_t>(k));
        seed = h;
    } else {
        static_assert(sizeof(std::size_t) == 4 || sizeof(std::size_t) == 8,
                      "Unsupported size_t size for hash_combine");
    }
}

/**
 * @class File
 * @brief RAII wrapper for C-style FILE* handles.
 *
 * Automatically closes the file on destruction. Supports move semantics
 * but not copy. Throws on failed open.
 *
 * ### Example
 * @code
 * File f("data.bin", "rb");
 * fread(buffer, 1, size, f);  // Implicit conversion to FILE*
 * // File automatically closed when f goes out of scope
 * @endcode
 */
class File {
  public:
    /**
     * @brief Open a file with the given mode.
     * @param path File path to open.
     * @param mode C-style mode string ("r", "rb", "w", etc.).
     * @throws std::runtime_error If the file cannot be opened.
     */
    File(const std::filesystem::path &path, const char *mode)
        : handle(std::fopen(path.string().c_str(), mode)) {
        if (!handle) {
            throw std::runtime_error("Failed to open file: " + path.string());
        }
    }

    /// Implicit conversion to FILE* for use with C file APIs.
    operator FILE *() const noexcept { return handle; }

    File(File &&other) noexcept : handle(other.handle) {
        other.handle = nullptr;
    }

    File &operator=(File &&other) noexcept {
        if (this != &other) {
            close();
            handle = other.handle;
            other.handle = nullptr;
        }
        return *this;
    }

    File(const File &) = delete;
    File &operator=(const File &) = delete;

    ~File() { close(); }

  private:
    FILE *handle = nullptr;

    void close() noexcept {
        if (handle) {
            std::fclose(handle);
            handle = nullptr;
        }
    }
};

/**
 * @brief Read an entire file into a string.
 * @param path Path to the file.
 * @return File contents as a string.
 * @throws std::runtime_error If the file cannot be opened.
 */
inline std::string read_file(const std::filesystem::path &path) {
    File f(path, "rb");
    ::fseek(f, 0, SEEK_END);
    auto size = ::ftell(f);
    ::fseek(f, 0, SEEK_SET);
    std::string result;
    result.resize(size);
    std::ignore = ::fread(result.data(), 1, size, f);
    return result;
}

/**
 * @struct DSU
 * @brief Disjoint Set Union (Union-Find) data structure.
 *
 * Tracks equivalence classes with path compression for efficient find
 * operations. Used for grouping columns in join ordering.
 *
 * ### Example
 * @code
 * DSU dsu(10);
 * dsu.unite(1, 2);
 * dsu.unite(2, 3);
 * assert(dsu.find(1) == dsu.find(3));  // Same set
 * @endcode
 */
struct DSU {
    std::vector<size_t> pa; ///< Parent pointers (pa[x] == x means x is root).

    /**
     * @brief Create a DSU with n elements, each in its own set.
     * @param size Number of elements.
     */
    explicit DSU(size_t size) : pa(size) { std::iota(pa.begin(), pa.end(), 0); }

    /**
     * @brief Find the representative of x's set (with path compression).
     * @param x Element to find.
     * @return The root/representative of x's set.
     */
    size_t find(size_t x) { return pa[x] == x ? x : pa[x] = find(pa[x]); }

    /**
     * @brief Unite the sets containing x and y.
     * @param x First element.
     * @param y Second element.
     */
    void unite(size_t x, size_t y) { pa[find(x)] = find(y); }
};

/**
 * @brief Mark a code path as unreachable for compiler optimization.
 *
 * If execution reaches this point, behavior is undefined. Allows the
 * compiler to optimize away impossible branches.
 */
[[noreturn]] inline void unreachable() {
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#if defined(_MSC_VER) && !defined(__clang__) // MSVC
    __assume(false);
#else // GCC, Clang
    __builtin_unreachable();
#endif
}
/**
 * @file common.h
 * @brief Foundational utilities: hashing, file I/O, DSU.
 *
 * - hash_combine: MurmurHash-style composite key hashing
 * - File: RAII FILE* wrapper
 * - DSU: Union-Find with path compression
 * - unreachable(): Compiler optimization hint
 */

#pragma once

#include <filesystem>
#include <numeric>
#include <vector>

#include <cstdint>
#include <cstdlib>

/**
 * @namespace detail
 * @brief Internal: MurmurHash mixing functions. Not for direct use.
 */
namespace detail {
/** @brief 32-bit rotate left. */
inline uint32_t rotl32(uint32_t x, uint8_t bits) {
    return (x << bits) | (x >> (32 - bits));
}

/** @brief MurmurHash3-style 32-bit hash combine. Modifies h1 in place. */
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

/** @brief MurmurHash2-style 64-bit hash combine. Modifies h in place. */
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
 * @brief Combine hash seed with new value (MurmurHash mixing).
 *
 * Selects 32/64-bit impl based on sizeof(size_t). Modifies seed in place.
 *
 * @param seed Accumulator (modified).
 * @param k Value to mix in.
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
 * @brief RAII FILE* wrapper. Move-only, throws on failed open.
 */
class File {
  public:
    /** @brief Open file. @throws std::runtime_error on failure. */
    File(const std::filesystem::path &path, const char *mode)
        : handle(std::fopen(path.string().c_str(), mode)) {
        if (!handle) {
            throw std::runtime_error("Failed to open file: " + path.string());
        }
    }

    /** @brief Implicit conversion to FILE*. */
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

/** @brief Read entire file into string. @throws std::runtime_error on failure. */
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
 * @brief Union-Find with path compression. Used for join column grouping.
 */
struct DSU {
    std::vector<size_t> pa; ///< Parent pointers (pa[x]==x → root).

    /** @brief Initialize n singleton sets. */
    explicit DSU(size_t size) : pa(size) { std::iota(pa.begin(), pa.end(), 0); }

    /** @brief Find root of x's set with path compression. */
    size_t find(size_t x) { return pa[x] == x ? x : pa[x] = find(pa[x]); }

    /** @brief Merge sets containing x and y. */
    void unite(size_t x, size_t y) { pa[find(x)] = find(y); }
};

/** @brief Mark unreachable code path for compiler optimization (UB if reached). */
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
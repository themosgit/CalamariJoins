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

/** @brief Read entire file into string. @throws std::runtime_error on failure.
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

/** @brief Mark unreachable code path for compiler optimization (UB if reached).
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

namespace Contest {

/**
 * @brief Encoded global row ID: 5-bit table_id + 27-bit row_id.
 *
 * Supports up to 32 tables and 134M rows per table.
 * Used to track original scan table rows through recursive joins.
 *
 * Encoding: [table_id (5 bits)][row_id (27 bits)]
 *   - table_id: bits 27-31
 *   - row_id: bits 0-26
 */
struct GlobalRowId {
    static constexpr uint32_t TABLE_BITS = 5;
    static constexpr uint32_t ROW_BITS = 27;
    static constexpr uint32_t TABLE_SHIFT = ROW_BITS;
    static constexpr uint32_t ROW_MASK = (1u << ROW_BITS) - 1;
    static constexpr uint32_t MAX_TABLES = 1u << TABLE_BITS; // 32
    static constexpr uint32_t MAX_ROWS = 1u << ROW_BITS;     // 134,217,728

    /** @brief Encode table_id and row_id into a single uint32_t. */
    static inline uint32_t encode(uint8_t table_id, uint32_t row_id) {
        return (static_cast<uint32_t>(table_id) << TABLE_SHIFT) |
               (row_id & ROW_MASK);
    }

    /** @brief Extract table_id from encoded global row ID. */
    static inline uint8_t table(uint32_t encoded) {
        return static_cast<uint8_t>(encoded >> TABLE_SHIFT);
    }

    /** @brief Extract row_id from encoded global row ID. */
    static inline uint32_t row(uint32_t encoded) { return encoded & ROW_MASK; }
};

/**
 * @brief 64-bit encoding for deferred column provenance.
 *
 * Encodes table_id, column_idx, and row_id into a single 64-bit value
 * for efficient storage and resolution of deferred columns.
 *
 * Encoding: [table_id (8 bits)][column_idx (8 bits)][row_id (48 bits)]
 *   - table_id: bits 56-63
 *   - column_idx: bits 48-55
 *   - row_id: bits 0-47
 *
 * Supports up to 256 tables, 256 columns per table, and 281 trillion rows.
 */
struct DeferredProvenance {
    static constexpr uint64_t ROW_BITS = 48;
    static constexpr uint64_t COLUMN_BITS = 8;
    static constexpr uint64_t TABLE_BITS = 8;

    static constexpr uint64_t ROW_MASK = (1ULL << ROW_BITS) - 1;
    static constexpr uint64_t COLUMN_MASK = (1ULL << COLUMN_BITS) - 1;
    static constexpr uint64_t COLUMN_SHIFT = ROW_BITS;
    static constexpr uint64_t TABLE_SHIFT = ROW_BITS + COLUMN_BITS;

    static constexpr uint64_t MAX_TABLES = 1ULL << TABLE_BITS;   // 256
    static constexpr uint64_t MAX_COLUMNS = 1ULL << COLUMN_BITS; // 256
    static constexpr uint64_t MAX_ROWS = 1ULL << ROW_BITS;       // 281 trillion

    /** @brief Encode table_id, column_idx, row_id into single uint64_t. */
    static inline uint64_t encode(uint8_t table_id, uint8_t column_idx,
                                  uint64_t row_id) {
        return (static_cast<uint64_t>(table_id) << TABLE_SHIFT) |
               (static_cast<uint64_t>(column_idx) << COLUMN_SHIFT) |
               (row_id & ROW_MASK);
    }

    /** @brief Extract table_id from encoded provenance. */
    static inline uint8_t table(uint64_t encoded) {
        return static_cast<uint8_t>(encoded >> TABLE_SHIFT);
    }

    /** @brief Extract column_idx from encoded provenance. */
    static inline uint8_t column(uint64_t encoded) {
        return static_cast<uint8_t>((encoded >> COLUMN_SHIFT) & COLUMN_MASK);
    }

    /** @brief Extract row_id from encoded provenance. */
    static inline uint64_t row(uint64_t encoded) { return encoded & ROW_MASK; }
};

} // namespace Contest
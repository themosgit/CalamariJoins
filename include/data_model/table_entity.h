/**
 * @file table_entity.h
 * @brief TableEntity: composite key (table + column index) for query plans.
 *
 * Hashable, comparable, fmt-formattable.
 */

#pragma once

#include <fmt/core.h>
#include <string>

#include <foundation/common.h>

/**
 * @struct TableEntity
 * @brief Column reference: (table name, column index).
 *
 * Supports comparison (ordered containers), hashing (unordered containers).
 */
struct TableEntity {
    std::string table; ///< Table name/alias.
    int id;            ///< 0-based column index.

    friend bool operator==(const TableEntity &left, const TableEntity &right);
    friend bool operator!=(const TableEntity &left, const TableEntity &right);
    friend bool operator<(const TableEntity &left, const TableEntity &right);
};

inline bool operator==(const TableEntity &left, const TableEntity &right) {
    return left.table == right.table && left.id == right.id;
}

inline bool operator!=(const TableEntity &left, const TableEntity &right) {
    return !(left == right);
}

inline bool operator<(const TableEntity &left, const TableEntity &right) {
    if (left.table < right.table) {
        return true;
    } else if (left.table > right.table) {
        return false;
    } else {
        return left.id < right.id;
    }
}

/** @brief std::hash for TableEntity (uses hash_combine). */
namespace std {
template <> struct hash<TableEntity> {
    size_t operator()(const TableEntity &te) const noexcept {
        size_t seed = 0;
        hash_combine(seed, hash<string>{}(te.table));
        hash_combine(seed, hash<int>{}(te.id));
        return seed;
    }
};

} // namespace std

/** @brief fmt formatter: outputs "(table, id)". */
template <> struct fmt::formatter<TableEntity> {
    template <class ParseContext> constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template <class FormatContext>
    auto format(const TableEntity &te, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "({}, {})", te.table, te.id);
    }
};

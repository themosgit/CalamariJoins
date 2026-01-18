/**
 * @file table_entity.h
 * @brief TableEntity identifier for referencing columns in query plans.
 *
 * Provides a composite key type (table name + column index) used throughout
 * the query execution engine to identify specific columns. Includes hashing
 * support for use in hash maps and fmt formatting for debugging.
 */

#pragma once

#include <fmt/core.h>
#include <string>

#include <foundation/common.h>

/**
 * @struct TableEntity
 * @brief Identifies a column by table name and column index.
 *
 * Used in query plans to reference specific columns across joined tables.
 * Supports comparison operators for use in ordered containers and hashing
 * for unordered containers.
 *
 * ### Example
 * @code
 * TableEntity col1{"movies", 0};  // First column of movies table
 * TableEntity col2{"actors", 2};  // Third column of actors table
 * @endcode
 */
struct TableEntity {
    std::string table; ///< Table name (or alias in query).
    int id;            ///< Zero-based column index within the table.

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

/**
 * @brief std::hash specialization for TableEntity.
 *
 * Enables use of TableEntity as a key in std::unordered_map and
 * std::unordered_set. Combines table name and id hashes using
 * hash_combine().
 */
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

/**
 * @brief fmt formatter for TableEntity.
 *
 * Formats as "(table, id)" for debug output.
 */
template <> struct fmt::formatter<TableEntity> {
    template <class ParseContext> constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template <class FormatContext>
    auto format(const TableEntity &te, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "({}, {})", te.table, te.id);
    }
};

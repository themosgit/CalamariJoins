/**
 * @file attribute.h
 * @brief Column data types and schema attribute definitions.
 *
 * Defines the DataType enum for column types and the Attribute struct for
 * schema definitions. Also provides DISPATCH_DATA_TYPE macro for type-safe
 * dispatch based on runtime DataType values.
 */

#pragma once

#include <array>
#include <string>

#include <fmt/core.h>

/**
 * @enum DataType
 * @brief Supported column data types.
 *
 * Each type corresponds to a specific C++ type and storage format:
 * - INT32: 4-byte signed integer (int32_t)
 * - INT64: 8-byte signed integer (int64_t)
 * - FP64: 8-byte IEEE 754 double (double)
 * - VARCHAR: Variable-length UTF-8 string (std::string)
 */
enum class DataType {
    INT32,   ///< 4-byte integer.
    INT64,   ///< 8-byte integer.
    FP64,    ///< 8-byte floating point.
    VARCHAR, ///< String of arbitrary length.
};

/**
 * @brief fmt formatter for DataType enum.
 *
 * Formats as the enum name (e.g., "INT32", "VARCHAR").
 */
template <> struct fmt::formatter<DataType> {
    template <class ParseContext> constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template <class FormatContext>
    auto format(DataType value, FormatContext &ctx) const {
        static std::array<std::string_view, 4> names{
            "INT32",
            "INT64",
            "FP64",
            "VARCHAR",
        };
        return fmt::format_to(ctx.out(), "{}", names[int(value)]);
    }
};

/**
 * @def DISPATCH_DATA_TYPE
 * @brief Dispatch code based on runtime DataType value.
 *
 * Expands code with a type alias `TYPE` set to the appropriate C++ type
 * for the given DataType. Useful for avoiding repetitive switch statements
 * when working with type-generic columnar operations.
 *
 * @param type The DataType value to dispatch on.
 * @param TYPE The name of the type alias to define.
 * @param ... Code to execute with TYPE defined.
 *
 * ### Example
 * @code
 * DataType dt = DataType::INT32;
 * DISPATCH_DATA_TYPE(dt, T, {
 *     std::vector<T> data;  // T is int32_t
 *     data.push_back(42);
 * });
 * @endcode
 */
#define DISPATCH_DATA_TYPE(type, TYPE, ...)                                    \
    do {                                                                       \
        switch (type) {                                                        \
        case DataType::INT32: {                                                \
            using TYPE = int32_t;                                              \
            __VA_ARGS__                                                        \
            break;                                                             \
        }                                                                      \
        case DataType::INT64: {                                                \
            using TYPE = int64_t;                                              \
            __VA_ARGS__                                                        \
            break;                                                             \
        }                                                                      \
        case DataType::FP64: {                                                 \
            using TYPE = double;                                               \
            __VA_ARGS__                                                        \
            break;                                                             \
        }                                                                      \
        case DataType::VARCHAR: {                                              \
            using TYPE = std::string;                                          \
            __VA_ARGS__                                                        \
            break;                                                             \
        }                                                                      \
        }                                                                      \
    } while (0)

/**
 * @struct Attribute
 * @brief Schema definition for a table column.
 *
 * Pairs a column name with its data type. Used when loading tables from
 * CSV or defining query result schemas.
 */
struct Attribute {
    DataType type;    ///< The column's data type.
    std::string name; ///< The column's name.
};
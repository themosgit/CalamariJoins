/**
 * @file statement.h
 * @brief Filter predicate AST for SQL WHERE clause evaluation.
 *
 * Defines the abstract syntax tree (AST) for representing filter predicates
 * from SQL WHERE clauses. Supports comparison operations (=, <, >, LIKE, etc.)
 * and logical operators (AND, OR, NOT).
 *
 * ### Design Overview
 * - **Statement**: Abstract base class for all predicate nodes
 * - **Comparison**: Leaf nodes comparing a column to a literal value
 * - **LogicalOperation**: Internal nodes combining predicates with AND/OR/NOT
 *
 * ### Evaluation Modes
 * Predicates can be evaluated in two modes:
 * 1. **Row-at-a-time**: `eval(vector<Data>&)` for single-row evaluation
 * 2. **Columnar batch**: `eval(vector<InnerColumnBase*>&)` returns a bitmap
 *    of matching rows, executed in parallel via FilterThreadPool
 *
 * @see InnerColumn for parallel columnar filter operations
 * @see CSVParser for where these predicates are applied during loading
 */

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <fmt/core.h>
#include <re2/re2.h>

/**
 * @typedef Data
 * @brief Variant type for database field values.
 *
 * Represents a single field value that can hold:
 * - int32_t: 32-bit integer (INT32)
 * - int64_t: 64-bit integer (INT64)
 * - double: 64-bit floating point (FP64)
 * - std::string: Variable-length string (VARCHAR)
 * - std::monostate: NULL value
 */
using Data =
    std::variant<int32_t, int64_t, double, std::string, std::monostate>;

/**
 * @typedef Literal
 * @brief Variant type for SQL literal values in predicates.
 *
 * Represents compile-time constant values in comparisons:
 * - int64_t: Integer literals (auto-widened from int32_t)
 * - double: Floating-point literals
 * - std::string: String literals
 * - std::monostate: NULL literal
 */
using Literal = std::variant<int64_t, double, std::string, std::monostate>;

/**
 * @brief fmt formatter specialization for Data variant.
 *
 * Enables printing Data values with fmt::format(). NULL values print as
 * "NULL", other types use their natural representation.
 */
template <> struct fmt::formatter<Data> {
    template <class ParseContext> constexpr auto parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template <class FormatContext>
    auto format(const Data &value, FormatContext &ctx) const {
        return std::visit(
            [&ctx](const auto &value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, std::monostate>) {
                    return fmt::format_to(ctx.out(), "NULL");
                } else {
                    return fmt::format_to(ctx.out(), "{}", value);
                }
            },
            value);
    }
};

struct Attribute;
struct Statement;
struct Comparison;
struct LogicalOperation;
struct InnerColumnBase;

/**
 * @struct Statement
 * @brief Abstract base class for filter predicate AST nodes.
 *
 * Represents a boolean expression that can be evaluated against row data.
 * Concrete implementations are Comparison (leaf) and LogicalOperation (node).
 */
// AST Node
struct Statement {
    virtual ~Statement() = default;

    /**
     * @brief Pretty-print the AST for debugging.
     * @param indent Indentation level for nested printing.
     * @return Human-readable string representation.
     */
    virtual std::string pretty_print(int indent = 0) const = 0;

    /**
     * @brief Evaluate the predicate against a single row.
     * @param record Row data indexed by column position.
     * @return True if the row satisfies the predicate.
     */
    virtual bool eval(const std::vector<Data> &record) const = 0;

    /**
     * @brief Evaluate the predicate against columnar data in parallel.
     *
     * Returns a bitmap where bit i = 1 means row i satisfies the predicate.
     * Leverages FilterThreadPool for parallel evaluation.
     *
     * @param table Column pointers indexed by column position.
     * @return Packed bitmap of matching rows.
     */
    virtual std::vector<uint8_t>
    eval(const std::vector<const InnerColumnBase *> &table) const = 0;
};

/**
 * @struct Comparison
 * @brief Leaf AST node comparing a column value to a literal.
 *
 * Represents predicates like `column = 42`, `name LIKE '%foo%'`, or
 * `value IS NULL`. Supports numeric comparison with type coercion and
 * string comparison with LIKE pattern matching.
 *
 * ### Supported Operators
 * - EQ, NEQ: Equality comparison (works for all types)
 * - LT, GT, LEQ, GEQ: Ordering comparison (numeric and string)
 * - LIKE, NOT_LIKE: SQL pattern matching (strings only)
 * - IS_NULL, IS_NOT_NULL: NULL checks
 *
 * @see like_match() for LIKE pattern matching implementation
 */
struct Comparison : Statement {
    size_t column; ///< Column index to compare.

    /**
     * @enum Op
     * @brief Comparison operators.
     */
    enum Op { EQ, NEQ, LT, GT, LEQ, GEQ, LIKE, NOT_LIKE, IS_NULL, IS_NOT_NULL };

    Op op;         ///< The comparison operator.
    Literal value; ///< The literal value to compare against.

    /**
     * @brief Construct a comparison predicate.
     * @param col Column index.
     * @param o Comparison operator.
     * @param val Literal value (ignored for IS_NULL/IS_NOT_NULL).
     */
    Comparison(size_t col, Op o, Literal val)
        : column(col), op(o), value(std::move(val)) {}

    std::string pretty_print(int indent) const override {
        return fmt::format("{:{}}{} {} {}", "", indent, column, opToString(),
                           valueToString());
    }

    bool eval(const std::vector<Data> &record) const override;
    std::vector<uint8_t>
    eval(const std::vector<const InnerColumnBase *> &table) const override;

    /** @brief Convert operator to string representation. */
    std::string opToString() const {
        switch (op) {
        case EQ:
            return "=";
        case NEQ:
            return "!=";
        case LT:
            return "<";
        case GT:
            return ">";
        case LEQ:
            return "<=";
        case GEQ:
            return ">=";
        case LIKE:
            return "LIKE";
        case NOT_LIKE:
            return "NOT LIKE";
        case IS_NULL:
            return "IS NULL";
        case IS_NOT_NULL:
            return "IS NOT NULL";
        default:
            return "??";
        }
    }

    /** @brief Convert literal value to string representation. */
    std::string valueToString() const {
        if (op == IS_NULL || op == IS_NOT_NULL) {
            return "";
        }
        return visit(
            [](auto &&arg) -> std::string {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, std::string>) {
                    return fmt::format("'{}'", arg);
                } else if constexpr (std::is_same_v<T, std::monostate>) {
                    return "";
                } else {
                    return fmt::format("{}", arg);
                }
            },
            value);
    }

    /**
     * @brief Match a string against a SQL LIKE pattern.
     *
     * Converts LIKE pattern to regex (% -> .*, _ -> .) and uses RE2
     * for matching. Compiled regexes are cached per-thread for efficiency.
     *
     * @param str The string to match.
     * @param pattern The LIKE pattern (e.g., "%foo%", "bar_baz").
     * @return True if the string matches the pattern.
     */
    static bool like_match(std::string_view str, const std::string &pattern) {
        // static cache and mutex
        thread_local auto regex_cache =
            std::unordered_map<std::string, std::unique_ptr<RE2>>{};

        const RE2 *re = nullptr;
        auto it = regex_cache.find(pattern);
        if (it != regex_cache.end()) {
            re = it->second.get();
        }

        // cache miss and compile
        if (!re) {
            // conver to regex
            std::string regex_str;
            for (char c : pattern) {
                if (c == '%') {
                    regex_str += ".*";
                } else if (c == '_') {
                    regex_str += '.';
                } else {
                    // escape sepcical characters
                    if (c == '\\' || c == '.' || c == '^' || c == '$' ||
                        c == '|' || c == '?' || c == '*' || c == '+' ||
                        c == '(' || c == ')' || c == '[' || c == ']' ||
                        c == '{' || c == '}') {
                        regex_str += '\\';
                    }
                    regex_str += c;
                }
            }

            RE2::Options options;

            auto new_re = std::make_unique<RE2>(regex_str, options);
            if (!new_re->ok()) {
                return false; // invalid regex
            }

            re = new_re.get();
            regex_cache.emplace(pattern, std::move(new_re));
        }

        // execute full match
        return RE2::FullMatch(str, *re);
    }

    /**
     * @brief Extract numeric value from a Data variant for comparison.
     * @param data The data value to convert.
     * @return The numeric value as double, or nullopt if not numeric.
     */
    static std::optional<double> get_numeric_value(const Data &data) {
        if (auto *i32 = std::get_if<int32_t>(&data)) {
            return *i32;
        } else if (auto *i64 = std::get_if<int64_t>(&data)) {
            return static_cast<double>(*i64);
        } else if (auto *d = std::get_if<double>(&data)) {
            return *d;
        } else {
            return std::nullopt;
        }
    }

    /**
     * @brief Extract numeric value from a Literal variant for comparison.
     * @param value The literal value to convert.
     * @return The numeric value as double, or nullopt if not numeric.
     */
    static std::optional<double> get_numeric_value(const Literal &value) {
        if (auto *i = std::get_if<int64_t>(&value)) {
            return *i;
        } else if (auto *d = std::get_if<double>(&value)) {
            return *d;
        } else {
            return std::nullopt;
        }
    }
};

/**
 * @struct LogicalOperation
 * @brief Internal AST node combining predicates with logical operators.
 *
 * Represents AND, OR, and NOT operations on child Statement nodes.
 * AND/OR have two children; NOT has one child.
 *
 * ### Evaluation Semantics
 * - AND: Short-circuits on first false (row-at-a-time) or combines bitmaps
 * - OR: Short-circuits on first true (row-at-a-time) or combines bitmaps
 * - NOT: Inverts the child's result
 *
 * For columnar evaluation, logical operations combine child bitmaps using
 * bitwise operations.
 */
struct LogicalOperation : Statement {
    /**
     * @enum Type
     * @brief Logical operation types.
     */
    enum Type { AND, OR, NOT };

    Type op_type; ///< The logical operation type.
    std::vector<std::unique_ptr<Statement>> children; ///< Child predicates.

    /**
     * @brief Create an AND node combining two predicates.
     * @param l Left child predicate.
     * @param r Right child predicate.
     * @return AND node owning both children.
     */
    static std::unique_ptr<LogicalOperation>
    makeAnd(std::unique_ptr<Statement> l, std::unique_ptr<Statement> r) {
        auto node = std::make_unique<LogicalOperation>();
        node->op_type = AND;
        node->children.push_back(std::move(l));
        node->children.push_back(std::move(r));
        return node;
    }

    /**
     * @brief Create an OR node combining two predicates.
     * @param l Left child predicate.
     * @param r Right child predicate.
     * @return OR node owning both children.
     */
    static std::unique_ptr<LogicalOperation>
    makeOr(std::unique_ptr<Statement> l, std::unique_ptr<Statement> r) {
        auto node = std::make_unique<LogicalOperation>();
        node->op_type = OR;
        node->children.push_back(std::move(l));
        node->children.push_back(std::move(r));
        return node;
    }

    /**
     * @brief Create a NOT node inverting a predicate.
     * @param child The predicate to negate.
     * @return NOT node owning the child.
     */
    static std::unique_ptr<LogicalOperation>
    makeNot(std::unique_ptr<Statement> child) {
        auto node = std::make_unique<LogicalOperation>();
        node->op_type = NOT;
        node->children.push_back(std::move(child));
        return node;
    }

    std::string pretty_print(int indent) const override {
        std::string op_str = [this] {
            switch (op_type) {
            case AND:
                return "AND";
            case OR:
                return "OR";
            case NOT:
                return "NOT";
            default:
                return "UNKNOWN";
            }
        }();

        std::string result = fmt::format("{:{}}[{}]\n", "", indent, op_str);

        for (auto &child : children) {
            result += child->pretty_print(indent + 2) + "\n";
        }

        if (!children.empty()) {
            result.pop_back();
        }
        return result;
    }

    bool eval(const std::vector<Data> &record) const override;
    std::vector<uint8_t>
    eval(const std::vector<const InnerColumnBase *> &table) const override;
};

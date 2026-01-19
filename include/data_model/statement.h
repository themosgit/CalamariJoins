/**
 * @file statement.h
 * @brief Filter predicate AST for SQL WHERE evaluation.
 *
 * Statement (base) → Comparison (leaf: column op literal) or LogicalOperation
 * (AND/OR/NOT). Two eval modes: row-at-a-time or columnar batch (bitmap, parallel).
 *
 * @see InnerColumn (parallel filter), CSVParser (predicate application)
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

/** @brief Field variant: int32_t, int64_t, double, string, or monostate (NULL). */
using Data =
    std::variant<int32_t, int64_t, double, std::string, std::monostate>;

/** @brief Literal variant: int64_t, double, string, or monostate (NULL). */
using Literal = std::variant<int64_t, double, std::string, std::monostate>;

/** @brief fmt formatter for Data. NULL prints as "NULL". */
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
 * @brief Abstract predicate AST base. Subclasses: Comparison, LogicalOperation.
 */
struct Statement {
    virtual ~Statement() = default;

    /** @brief Pretty-print AST for debugging. */
    virtual std::string pretty_print(int indent = 0) const = 0;

    /** @brief Evaluate against single row. @return true if satisfied. */
    virtual bool eval(const std::vector<Data> &record) const = 0;

    /** @brief Columnar eval via FilterThreadPool. @return bitmap (bit i=1 → row i matches). */
    virtual std::vector<uint8_t>
    eval(const std::vector<const InnerColumnBase *> &table) const = 0;
};

/**
 * @struct Comparison
 * @brief Leaf: column op literal. Ops: EQ/NEQ/LT/GT/LEQ/GEQ/LIKE/IS_NULL etc.
 * @see like_match() for LIKE implementation.
 */
struct Comparison : Statement {
    size_t column; ///< Column index to compare.

    /** @brief Comparison operators. */
    enum Op { EQ, NEQ, LT, GT, LEQ, GEQ, LIKE, NOT_LIKE, IS_NULL, IS_NOT_NULL };

    Op op;         ///< The comparison operator.
    Literal value; ///< The literal value to compare against.

    /** @brief Construct comparison. val ignored for IS_NULL/IS_NOT_NULL. */
    Comparison(size_t col, Op o, Literal val)
        : column(col), op(o), value(std::move(val)) {}

    std::string pretty_print(int indent) const override {
        return fmt::format("{:{}}{} {} {}", "", indent, column, opToString(),
                           valueToString());
    }

    bool eval(const std::vector<Data> &record) const override;
    std::vector<uint8_t>
    eval(const std::vector<const InnerColumnBase *> &table) const override;

    /** @brief Op to string. */
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

    /** @brief Literal to string. */
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

    /** @brief LIKE pattern match (% → .*, _ → .). Thread-local regex cache. */
    static bool like_match(std::string_view str, const std::string &pattern) {
        thread_local auto regex_cache =
            std::unordered_map<std::string, std::unique_ptr<RE2>>{};

        const RE2 *re = nullptr;
        auto it = regex_cache.find(pattern);
        if (it != regex_cache.end()) {
            re = it->second.get();
        }

        if (!re) {
            std::string regex_str;
            for (char c : pattern) {
                if (c == '%') {
                    regex_str += ".*";
                } else if (c == '_') {
                    regex_str += '.';
                } else {
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
                return false;
            }

            re = new_re.get();
            regex_cache.emplace(pattern, std::move(new_re));
        }

        return RE2::FullMatch(str, *re);
    }

    /** @brief Extract numeric from Data as double, or nullopt. */
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

    /** @brief Extract numeric from Literal as double, or nullopt. */
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
 * @brief AND/OR/NOT combining child Statements. Row: short-circuit; columnar: bitwise.
 */
struct LogicalOperation : Statement {
    /** @brief Logical operation type. */
    enum Type { AND, OR, NOT };

    Type op_type; ///< The logical operation type.
    std::vector<std::unique_ptr<Statement>> children; ///< Child predicates.

    /** @brief Create AND node. */
    static std::unique_ptr<LogicalOperation>
    makeAnd(std::unique_ptr<Statement> l, std::unique_ptr<Statement> r) {
        auto node = std::make_unique<LogicalOperation>();
        node->op_type = AND;
        node->children.push_back(std::move(l));
        node->children.push_back(std::move(r));
        return node;
    }

    /** @brief Create OR node. */
    static std::unique_ptr<LogicalOperation>
    makeOr(std::unique_ptr<Statement> l, std::unique_ptr<Statement> r) {
        auto node = std::make_unique<LogicalOperation>();
        node->op_type = OR;
        node->children.push_back(std::move(l));
        node->children.push_back(std::move(r));
        return node;
    }

    /** @brief Create NOT node. */
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

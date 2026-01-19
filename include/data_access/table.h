/**
 * @file table.h
 * @brief Row-oriented Table and format conversions.
 *
 * Table: row-major vector<vector<Data>>. DumpTable: binary cache (TableMeta +
 * pages). @see ColumnarTable, CSVParser
 */

#pragma once

#include <filesystem>
#include <fmt/core.h>
#include <range/v3/all.hpp>

#include <data_model/plan.h>
#include <data_model/statement.h>
#include <foundation/attribute.h>

#ifdef TEAMOPT_USE_DUCKDB
#include <duckdb.hpp>
#endif

/** @namespace FNVHash @brief FNV-1a hash: XOR-then-multiply. @see Contest::FNVHash in hashtable.h for typed overloads. */
namespace FNVHash {
/// FNV-1a prime multiplier for 64-bit hashes.
constexpr uint64_t FNV_prime = 1099511628211u;
/// FNV-1a initial hash value (offset basis).
constexpr uint64_t offset_basis = 14695981039346656037u;

/** @brief Compute FNV-1a hash of byte sequence. */
inline uint64_t hash(const void *key, size_t len) {
    uint64_t h = offset_basis;
    const unsigned char *p = static_cast<const unsigned char *>(key);
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= FNV_prime;
    }
    return h;
}
}; // namespace FNVHash

/** @struct TableMeta @brief Binary cache header (padded to PAGE_SIZE). Max 16 columns. */
struct TableMeta {
    uint64_t num_rows;      ///< Total number of rows in the table.
    uint64_t num_cols;      ///< Number of columns (max 16).
    DataType types[16];     ///< Data type for each column.
    uint64_t num_pages[16]; ///< Number of pages per column.
};

/// Padding size to align TableMeta to PAGE_SIZE boundary.
#define FILLER_SIZE (PAGE_SIZE - sizeof(struct TableMeta))

/**
 * @class Table
 * @brief Row-oriented table: vector<vector<Data>>. For result output and format conversion.
 * @see ColumnarTable, DumpTable
 */
struct Table {
  public:
    Table() = default;

    /** @brief Construct from row-major data and column types. */
    Table(std::vector<std::vector<Data>> data, std::vector<DataType> types)
        : types_(types), data_(data) {}

    /** @brief Load ColumnarTable from binary cache. @see DumpTable::dump() */
    static ColumnarTable from_cache(const std::filesystem::path &path);

    /** @brief Parse CSV to ColumnarTable with optional filter. @see CSVParser */
    static ColumnarTable from_csv(const std::vector<Attribute> &attributes,
                                  const std::filesystem::path &path,
                                  Statement *filter, bool header = false);

    /** @brief Convert ColumnarTable to row-oriented Table. */
    static Table from_columnar(const ColumnarTable &input);

    /** @brief Extract selected columns as row data (projection during scan). */
    static std::vector<std::vector<Data>>
    copy_scan(const ColumnarTable &table,
              const std::vector<std::tuple<size_t, DataType>> &output_attrs);

    /**
     * @brief Convert this row-oriented table to columnar format.
     * @return ColumnarTable with the same data.
     */
    ColumnarTable to_columnar() const;

    /// @name Accessors
    /// @{

    /// Get read-only access to row data.
    const std::vector<std::vector<Data>> &table() const { return data_; }

    /// Get mutable access to row data.
    std::vector<std::vector<Data>> &table() { return data_; }

    /// Get column types.
    const std::vector<DataType> &types() const { return types_; }

    /// Get number of rows.
    size_t number_rows() const { return this->data_.size(); }

    /// Get number of columns.
    size_t number_cols() const { return this->types_.size(); }

    /// @}

    /** @brief Print pipe-delimited rows. NULL → "NULL", strings quoted, escapes handled. */
    static void print(const std::vector<std::vector<Data>> &data) {
        namespace views = ranges::views;

        auto escape_string = [](const std::string &s) {
            std::string escaped;
            for (char c : s) {
                switch (c) {
                case '"':
                    escaped += "\\\"";
                    break;
                case '\\':
                    escaped += "\\\\";
                    break;
                case '\n':
                    escaped += "\\n";
                    break;
                case '\r':
                    escaped += "\\r";
                    break;
                case '\t':
                    escaped += "\\t";
                    break;
                default:
                    escaped += c;
                    break;
                }
            }
            return escaped;
        };

        for (auto &record : data) {
            auto line =
                record |
                views::transform([&escape_string](
                                     const Data &field) -> std::string {
                    return std::visit(
                        [&escape_string](const auto &arg) {
                            using T = std::decay_t<decltype(arg)>;
                            using namespace std::string_literals;
                            if constexpr (std::is_same_v<T, std::monostate>) {
                                return "NULL"s;
                            } else if constexpr (std::is_same_v<T, int32_t> ||
                                                 std::is_same_v<T, int64_t> ||
                                                 std::is_same_v<T, double>) {
                                return fmt::format("{}", arg);
                            } else if constexpr (std::is_same_v<T,
                                                                std::string>) {
                                return fmt::format("\"{}\"",
                                                   escape_string(arg));
                            }
                        },
                        field);
                }) |
                views::join('|') | ranges::to<std::string>();
            fmt::println("{}", line);
        }
    }

  private:
    std::vector<DataType> types_;         ///< Column types in order.
    std::vector<std::vector<Data>> data_; ///< Row-major table data.

    /** @brief Set column types from attribute list. */
    void set_attributes(const std::vector<Attribute> &attributes) {
        this->types_.clear();
        for (auto &attr : attributes) {
            this->types_.push_back(attr.type);
        }
    }
};

/**
 * @class DumpTable
 * @brief Serialize tables to binary cache. Format: TableMeta + padding + column pages.
 * @see Table::from_cache()
 */
class DumpTable {
  private:
    TableMeta tablemeta = {0}; ///< Metadata to write at file start.
    ColumnarTable *table;      ///< Table to serialize (owned externally).

  public:
    /** @brief Construct from ColumnarTable. Table must remain valid until dump(). */
    DumpTable(ColumnarTable *table) : table(table) {
        tablemeta.num_rows = table->num_rows;
        tablemeta.num_cols = table->columns.size();
        for (size_t i = 0; i < tablemeta.num_cols; ++i) {
            tablemeta.types[i] = table->columns[i].type;
            tablemeta.num_pages[i] = table->columns[i].pages.size();
        }
    }
#ifdef TEAMOPT_USE_DUCKDB
    /** @brief Construct from DuckDB results. Sorts rows. INT32/VARCHAR only. Requires TEAMOPT_USE_DUCKDB. */
    DumpTable(duckdb::MaterializedQueryResult &duckdb_results) {
        std::vector<std::vector<Data>> data;
        std::vector<DataType> types;

        tablemeta.num_rows = duckdb_results.RowCount();
        tablemeta.num_cols = duckdb_results.ColumnCount();
        for (size_t i = 0; i < tablemeta.num_cols; ++i) {
            switch (duckdb_results.types[i].id()) {
            case duckdb::LogicalTypeId::INTEGER:
                tablemeta.types[i] = DataType::INT32;
                types.push_back(DataType::INT32);
                break;
            case duckdb::LogicalTypeId::VARCHAR:
                tablemeta.types[i] = DataType::VARCHAR;
                types.push_back(DataType::VARCHAR);
                break;
            default:
                throw std::runtime_error(
                    fmt::format("{} in DuckDB is not supported",
                                duckdb_results.types[i].ToString()));
            }
        }

        auto &coll = duckdb_results.Collection();
        for (auto &row : coll.Rows()) {
            std::vector<Data> record;
            for (size_t col_idx = 0; col_idx < tablemeta.num_cols; col_idx++) {
                auto val = row.GetValue(col_idx);
                if (val.IsNull()) {
                    record.emplace_back(std::monostate{});
                } else {
                    switch (types[col_idx]) {
                    case DataType::INT32: {
                        record.emplace_back(duckdb::IntegerValue::Get(val));
                        break;
                    }
                    case DataType::VARCHAR: {
                        record.emplace_back(duckdb::StringValue::Get(val));
                        break;
                    }
                    default:
                        throw std::runtime_error("DataType not supported");
                    }
                }
            }
            data.emplace_back(std::move(record));
        }
        std::sort(data.begin(), data.end());
        table = new ColumnarTable();
        *table = Table(data, types).to_columnar();

        for (size_t i = 0; i < tablemeta.num_cols; ++i) {
            tablemeta.num_pages[i] = table->columns[i].pages.size();
        }
    }
#endif

    /** @brief Write binary cache: metadata + padding + column pages. Stream must be binary mode. */
    void dump(std::ostream &out) {
        out.write(reinterpret_cast<const char *>(&tablemeta),
                  sizeof(struct TableMeta));
        char filler[FILLER_SIZE] = {0};
        out.write(filler, FILLER_SIZE);

        for (size_t i = 0; i < tablemeta.num_cols; ++i) {
            for (auto *page : table->columns[i].pages) {
                out.write(reinterpret_cast<const char *>(page->data),
                          PAGE_SIZE);
            }
        }
    }
};
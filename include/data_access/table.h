/**
 * @file table.h
 * @brief Row-oriented table representation and format conversion utilities.
 *
 * Provides the Table class for row-major storage of query results, plus
 * conversion utilities between row-oriented and columnar formats. Also
 * includes DumpTable for serializing tables to binary cache files.
 *
 * ### Table Formats
 * - **Table**: Row-oriented storage using std::vector<std::vector<Data>>,
 *   suitable for final result output and small intermediate results.
 * - **ColumnarTable**: Column-oriented storage with 8KB pages, optimized
 *   for scanning and cache-based persistence.
 *
 * ### Binary Cache Format
 * The binary cache format (used by DumpTable and from_cache) stores tables
 * as:
 * 1. TableMeta header (padded to PAGE_SIZE)
 * 2. Column pages concatenated in column-major order
 *
 * @see ColumnarTable for the column-oriented representation
 * @see CSVParser for parsing CSV input files
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

/**
 * @namespace FNVHash
 * @brief FNV-1a hash function implementation.
 *
 * Provides a fast, non-cryptographic hash function suitable for hash tables.
 * FNV-1a (Fowler-Noll-Vo) is a simple XOR-then-multiply hash with good
 * distribution properties.
 *
 * @note For join hash tables, see Contest::FNVHash in hashtable.h which
 *       provides typed overloads for Data values.
 */
namespace FNVHash {
/// FNV-1a prime multiplier for 64-bit hashes.
constexpr uint64_t FNV_prime = 1099511628211u;
/// FNV-1a initial hash value (offset basis).
constexpr uint64_t offset_basis = 14695981039346656037u;

/**
 * @brief Compute FNV-1a hash of arbitrary byte sequence.
 *
 * @param key Pointer to data to hash.
 * @param len Number of bytes to hash.
 * @return 64-bit FNV-1a hash value.
 *
 * @note This is the byte-oriented version; for structured data,
 *       consider type-specific hash functions.
 */
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

/**
 * @struct TableMeta
 * @brief Metadata header for binary table cache files.
 *
 * Stored at the beginning of cache files (padded to PAGE_SIZE) to allow
 * quick table reconstruction without parsing CSV. Supports up to 16 columns.
 */
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
 * @brief Row-oriented table for result output and format conversion.
 *
 * Stores table data in row-major order using nested vectors. While less
 * cache-efficient than ColumnarTable for scans, this format is convenient
 * for:
 * - Final query result output (row-by-row printing)
 * - Small intermediate results
 * - Converting between row and columnar formats
 *
 * ### Data Storage
 * Each row is a vector<Data>, where Data is a variant that can hold:
 * - std::monostate (NULL)
 * - int32_t (INT32)
 * - int64_t (INT64)
 * - double (DOUBLE)
 * - std::string (VARCHAR)
 *
 * @see ColumnarTable for the column-oriented storage format
 * @see DumpTable for serializing to binary cache files
 */
struct Table {
  public:
    Table() = default;

    /**
     * @brief Construct table from row data and column types.
     *
     * @param data Row-major data: data[row][col] is the value at (row, col).
     * @param types Column types in order.
     */
    Table(std::vector<std::vector<Data>> data, std::vector<DataType> types)
        : types_(types), data_(data) {}

    /**
     * @brief Load a ColumnarTable from a binary cache file.
     *
     * Reads the TableMeta header and column pages from a file previously
     * written by DumpTable::dump().
     *
     * @param path Path to the binary cache file.
     * @return ColumnarTable reconstructed from the cache.
     *
     * @see DumpTable::dump() for the cache file format
     */
    static ColumnarTable from_cache(const std::filesystem::path &path);

    /**
     * @brief Parse a CSV file into a ColumnarTable with optional filtering.
     *
     * Uses CSVParser to stream through the file, applying the filter
     * predicate to each row and converting matching rows to columnar
     * format.
     *
     * @param attributes Schema defining column names and types.
     * @param path       Path to the CSV file.
     * @param filter     Optional filter predicate (nullptr to load all rows).
     * @param header     If true, skip the first line as a header row.
     * @return ColumnarTable containing filtered data.
     *
     * @see CSVParser for details on supported CSV formats
     */
    static ColumnarTable from_csv(const std::vector<Attribute> &attributes,
                                  const std::filesystem::path &path,
                                  Statement *filter, bool header = false);

    /**
     * @brief Convert a ColumnarTable to row-oriented Table.
     *
     * Scans all columns and assembles rows for output. Used when the
     * final result needs row-by-row access.
     *
     * @param input The columnar table to convert.
     * @return Row-oriented Table with the same data.
     */
    static Table from_columnar(const ColumnarTable &input);

    /**
     * @brief Extract selected columns from a ColumnarTable as row data.
     *
     * Performs a column projection during the scan, useful for
     * materializing only needed attributes.
     *
     * @param table       Source columnar table.
     * @param output_attrs List of (column_index, type) pairs to extract.
     * @return Row-major data for the selected columns.
     */
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

    /**
     * @brief Print row data to stdout in pipe-delimited format.
     *
     * Formats each row with fields separated by '|', properly escaping
     * special characters in VARCHAR fields. NULL values print as "NULL",
     * strings are double-quoted.
     *
     * @param data Row-major data to print.
     */
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
                                // return fmt::format("{}", arg);
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

    /**
     * @brief Set column types from attribute list.
     * @param attributes Schema attributes to extract types from.
     */
    void set_attributes(const std::vector<Attribute> &attributes) {
        this->types_.clear();
        for (auto &attr : attributes) {
            this->types_.push_back(attr.type);
        }
    }
};

/**
 * @class DumpTable
 * @brief Serializes tables to binary cache format for fast reloading.
 *
 * Converts ColumnarTable (or DuckDB query results) to a binary format that
 * can be quickly memory-mapped and reconstructed. This avoids CSV parsing
 * overhead for frequently-accessed tables.
 *
 * ### Cache File Format
 * 1. TableMeta header (sizeof(TableMeta) bytes)
 * 2. Zero padding to PAGE_SIZE boundary
 * 3. Column 0 pages (num_pages[0] * PAGE_SIZE bytes)
 * 4. Column 1 pages (num_pages[1] * PAGE_SIZE bytes)
 * 5. ... and so on for each column
 *
 * @see Table::from_cache() to load cached tables
 */
class DumpTable {
  private:
    TableMeta tablemeta = {0}; ///< Metadata to write at file start.
    ColumnarTable *table;      ///< Table to serialize (owned externally).

  public:
    /**
     * @brief Construct serializer from a ColumnarTable.
     *
     * Extracts metadata from the table for the cache header. The table
     * pointer must remain valid until dump() is called.
     *
     * @param table Pointer to the table to serialize.
     */
    DumpTable(ColumnarTable *table) : table(table) {
        tablemeta.num_rows = table->num_rows;
        tablemeta.num_cols = table->columns.size();
        for (size_t i = 0; i < tablemeta.num_cols; ++i) {
            tablemeta.types[i] = table->columns[i].type;
            tablemeta.num_pages[i] = table->columns[i].pages.size();
        }
    }
#ifdef TEAMOPT_USE_DUCKDB
    /**
     * @brief Construct serializer from DuckDB query results.
     *
     * Converts DuckDB's MaterializedQueryResult to ColumnarTable format,
     * sorting rows for deterministic output. Supports INT32 and VARCHAR
     * column types.
     *
     * @param duckdb_results Query results to convert and serialize.
     * @throws std::runtime_error If unsupported column types are encountered.
     *
     * @note Only available when compiled with TEAMOPT_USE_DUCKDB.
     */
    DumpTable(duckdb::MaterializedQueryResult &duckdb_results) {
        std::vector<std::vector<Data>> data;
        std::vector<DataType> types;

        /* Setup metadata */
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

        /* Convert duckdb to ColumnarTable */
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

    /**
     * @brief Write the table to a binary stream.
     *
     * Outputs the cache format: metadata header, padding, then all column
     * pages in order. The stream should be opened in binary mode.
     *
     * @param out Output stream to write to.
     */
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
/**
 * @file csv_parser.h
 * @brief Streaming CSV parser with configurable delimiters and escape handling.
 *
 * Provides a base class for parsing CSV files in a streaming fashion, suitable
 * for processing large files without loading them entirely into memory. The
 * parser handles quoted fields, escape sequences, and validates column
 * consistency across rows.
 *
 * @note This is an abstract class; subclasses must implement on_field() to
 *       handle parsed field data.
 */

#pragma once

#include <vector>

#include <cstdlib>

/**
 * @class CSVParser
 * @brief Abstract streaming CSV parser with configurable format options.
 *
 * Parses CSV data incrementally via execute() calls, invoking the virtual
 * on_field() callback for each parsed field. Supports:
 * - Configurable field separator (default: comma)
 * - Configurable escape/quote character (default: double-quote)
 * - Optional trailing comma after last field
 * - Proper handling of quoted fields containing separators/newlines
 * - CRLF and LF line endings
 *
 * ### Usage Pattern
 * @code
 * class MyParser : public CSVParser {
 *     void on_field(size_t col, size_t row, const char* data, size_t len)
 * override {
 *         // Process field at (col, row)
 *     }
 * };
 *
 * MyParser parser;
 * parser.execute(buffer, length);
 * parser.finish();  // Flush any remaining field
 * @endcode
 *
 * @see Table::from_csv() for high-level CSV loading with schema validation
 */
class CSVParser {
  public:
    /**
     * @enum Error
     * @brief Parse error codes returned by execute() and finish().
     */
    enum Error {
        Ok,                  ///< No error, parsing succeeded.
        QuoteNotClosed,      ///< Quoted field missing closing quote at EOF.
        InconsistentColumns, ///< Row has different column count than first row.
        NoTrailingComma,     ///< Expected trailing comma not found (when
                             ///< has_trailing_comma=true).
    };

    /**
     * @brief Construct a CSV parser with format options.
     *
     * @param escape Quote/escape character for fields containing special
     *               characters. Use '"' for standard CSV or '\\' for
     *               backslash-escaped formats.
     * @param sep    Field separator character. Use ',' for CSV or '|' for
     *               pipe-delimited formats.
     * @param has_trailing_comma If true, expects each line to end with the
     *                           separator before the newline (# commas = #
     *                           columns). If false, the last field ends at
     *                           newline (# commas + 1 = # columns).
     */
    CSVParser(char escape = '"', char sep = ',',
              bool has_trailing_comma = false)
        : escape_(escape), comma_(sep),
          has_trailing_comma_(has_trailing_comma) {}

    /**
     * @brief Parse a chunk of CSV data.
     *
     * Processes the buffer incrementally, calling on_field() for each
     * completed field. May be called multiple times with consecutive
     * chunks of a large file. Maintains internal state between calls.
     *
     * @param buffer Pointer to CSV data (not null-terminated).
     * @param len    Number of bytes in buffer.
     * @return Error::Ok on success, or an error code on parse failure.
     *
     * @note The parser copies incomplete field data internally, so the
     *       buffer need not remain valid after this call returns.
     */
    [[nodiscard]] Error execute(const char *buffer, size_t len);

    /**
     * @brief Finalize parsing and flush any remaining field.
     *
     * Must be called after all data has been passed to execute() to
     * handle any final field that wasn't terminated by a newline.
     *
     * @return Error::Ok on success, Error::QuoteNotClosed if inside a
     *         quoted field, or Error::InconsistentColumns if the final
     *         row has wrong column count.
     */
    [[nodiscard]] Error finish();

    /**
     * @brief Callback invoked for each parsed field.
     *
     * Subclasses must implement this to process field data. Called in
     * row-major order (all fields of row 0, then row 1, etc.).
     *
     * @param col_idx Zero-based column index within the row.
     * @param row_idx Zero-based row index (first data row is 0).
     * @param begin   Pointer to unescaped field content.
     * @param len     Length of field content in bytes.
     *
     * @note The pointer is valid only during this callback; copy the data
     *       if needed later. Escape sequences have already been processed.
     */
    virtual void on_field(size_t col_idx, size_t row_idx, const char *begin,
                          size_t len) = 0;

  private:
    /// @name Configuration
    /// @{
    char escape_{'"'}; ///< Quote/escape character (may also be '\\').
    char comma_{','};  ///< Field separator (may also be '|').
    /// True means # commas = # columns (trailing comma before newline);
    /// false means # commas + 1 = # columns.
    bool has_trailing_comma_{false};
    /// @}

    /// @name Parser State
    /// @{
    std::vector<char> current_field_; ///< Buffer for field being assembled.
    size_t col_idx_{0};               ///< Current column index within row.
    size_t row_idx_{0};               ///< Current row index.
    size_t num_cols_{0}; ///< Expected column count (set from first row).
    bool after_first_row_{false};  ///< True after first row parsed.
    bool quoted_{false};           ///< Currently inside quoted field.
    bool after_field_sep_{false};  ///< Just saw field separator.
    bool after_record_sep_{false}; ///< Just saw record separator (newline).
    bool escaping_{false};         ///< Next char is escaped.
    bool newlining_{false};        ///< Processing CRLF sequence.
    /// @}
};

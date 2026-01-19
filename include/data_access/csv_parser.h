/**
 * @file csv_parser.h
 * @brief Streaming CSV parser. Configurable escape/delim, validates columns.
 * Abstract: subclass implements on_field().
 */

#pragma once

#include <vector>

#include <cstdlib>

/**
 * @class CSVParser
 * @brief Abstract streaming CSV parser. Subclass implements on_field().
 *
 * Configurable separator/escape, quoted fields, CRLF/LF. Call execute() then finish().
 * @see Table::from_csv()
 */
class CSVParser {
  public:
    /** @brief Parse error codes. */
    enum Error {
        Ok,                  ///< Success.
        QuoteNotClosed,      ///< Missing closing quote at EOF.
        InconsistentColumns, ///< Row column count mismatch.
        NoTrailingComma,     ///< Missing trailing comma (when enabled).
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

    /** @brief Parse chunk incrementally. Calls on_field() per field. Maintains state across calls. */
    [[nodiscard]] Error execute(const char *buffer, size_t len);

    /** @brief Finalize parsing, flush remaining field. Returns error if inside quote or column mismatch. */
    [[nodiscard]] Error finish();

    /** @brief Field callback. Row-major order. Pointer valid only during call; escapes pre-processed. */
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

    std::vector<char> current_field_; ///< Field buffer.
    size_t col_idx_{0};               ///< Current column.
    size_t row_idx_{0};               ///< Current row.
    size_t num_cols_{0};              ///< Expected columns (from row 0).
    bool after_first_row_{false};     ///< First row parsed.
    bool quoted_{false};              ///< Inside quoted field.
    bool after_field_sep_{false};     ///< After field separator.
    bool after_record_sep_{false};    ///< After newline.
    bool escaping_{false};            ///< Next char escaped.
    bool newlining_{false};           ///< Processing CRLF.
};

/**
 * @file inner_column.h
 * @brief Columnar storage with parallel filter evaluation for CSV loading.
 *
 * Provides an in-memory columnar representation optimized for parallel
 * predicate evaluation during CSV loading. Unlike ColumnarTable (which uses
 * 8KB pages), InnerColumn stores data in flat vectors with separate null
 * bitmaps for efficient SIMD-style filtering.
 *
 * ### Design Overview
 * - **InnerColumn<T>**: Type-specific column with contiguous data vector and
 *   packed null bitmap (1 bit per row). Supports parallel comparison
 * operations.
 * - **InnerColumn<std::string>**: Specialization using a flat char buffer with
 *   offset array for variable-length strings.
 * - **FilterThreadPool**: Simple thread pool for parallel filter evaluation,
 *   partitioning work by bitmap bytes (8 rows per unit).
 *
 * ### Null Handling
 * Null values are tracked in a separate bitmap rather than using sentinel
 * values. Comparisons with null values always return false (SQL semantics).
 *
 * @see ColumnarTable for the page-based columnar format used after loading
 * @see Statement for the filter predicate AST
 */

#pragma once
#include <condition_variable>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

#include <cstdint>

#include "statement.h"
#include <foundation/attribute.h>

/**
 * @struct FilterThreadPool
 * @brief Simple thread pool for parallel filter predicate evaluation.
 *
 * Manages a fixed set of worker threads that execute filter operations in
 * parallel. Work is partitioned by bitmap byte index, so each thread handles
 * groups of 8 rows. Threads sleep between tasks and wake on condition variable.
 *
 * ### Synchronization Model
 * Each thread has its own mutex and condition variable. The main thread sets
 * `has_function` flags, notifies workers, then waits for all `finished` flags.
 *
 * @note This is a simpler pool than Contest::WorkerPool, designed specifically
 *       for the filter evaluation use case during CSV loading.
 */
struct FilterThreadPool {
    std::vector<std::thread> threads;               ///< Worker threads.
    std::vector<std::unique_ptr<std::mutex>> mtxes; ///< Per-thread mutexes.
    std::vector<std::unique_ptr<std::condition_variable>>
        cvs;                           ///< Per-thread CVs.
    std::vector<uint8_t> has_function; ///< True if thread has work pending.
    std::vector<uint8_t> finished;   ///< True if thread finished current work.
    std::vector<uint8_t> destructed; ///< True to signal thread shutdown.
    std::function<void(size_t, size_t)> function; ///< Current task to execute.
    size_t num_tasks; ///< Total number of work units.

    /**
     * @brief Calculate the starting work unit for a thread.
     *
     * Distributes work evenly with remainder going to earlier threads.
     *
     * @param thread_id Thread index (0-based).
     * @return Starting task index for this thread.
     */
    size_t begin_idx(size_t thread_id) {
        size_t base = num_tasks / threads.size();
        size_t rem = num_tasks % threads.size();
        return thread_id * base + std::min(thread_id, rem);
    }

    /**
     * @brief Worker thread main loop.
     *
     * Waits for work, executes assigned range, signals completion, repeats.
     * Exits when destructed flag is set.
     *
     * @param thread_id This thread's index.
     */
    void run_loop(size_t thread_id) {
        auto &mtx = *mtxes[thread_id];
        auto &cv = *cvs[thread_id];
        for (;;) {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [this, thread_id] {
                return destructed[thread_id] or
                       (has_function[thread_id] and not finished[thread_id]);
            });
            if (destructed[thread_id]) {
                break;
            }
            size_t begin = begin_idx(thread_id);
            size_t end = begin_idx(thread_id + 1);
            if (begin < end) {
                function(begin, end);
            }
            finished[thread_id] = true;
            lk.unlock();
            cv.notify_one();
        }
    }

    /**
     * @brief Create a thread pool with the specified number of workers.
     * @param num_threads Number of worker threads to spawn.
     */
    FilterThreadPool(unsigned num_threads) {
        for (unsigned i = 0; i < num_threads; ++i) {
            mtxes.emplace_back(std::make_unique<std::mutex>());
            cvs.emplace_back(std::make_unique<std::condition_variable>());
        }
        has_function.resize(num_threads);
        finished.resize(num_threads);
        destructed.resize(num_threads, 0);
        for (unsigned i = 0; i < num_threads; ++i) {
            threads.emplace_back([this, i] { run_loop(i); });
        }
    }

    FilterThreadPool(const FilterThreadPool &) = delete;
    FilterThreadPool(FilterThreadPool &&) = delete;
    FilterThreadPool &operator=(const FilterThreadPool &) = delete;
    FilterThreadPool &operator=(FilterThreadPool &&) = delete;

    /**
     * @brief Destroy the thread pool, joining all workers.
     */
    ~FilterThreadPool() {
        for (size_t i = 0; i < threads.size(); ++i) {
            auto &mtx = *mtxes[i];
            std::lock_guard<std::mutex> lk(mtx);
            destructed[i] = true;
        }
        for (size_t i = 0; i < threads.size(); ++i) {
            auto &cv = *cvs[i];
            cv.notify_one();
        }
        for (size_t i = 0; i < threads.size(); ++i) {
            threads[i].join();
        }
    }

    /**
     * @brief Execute a function in parallel across all workers.
     *
     * Partitions `num_tasks` work units across threads and blocks until
     * all threads complete.
     *
     * @param function Task to execute; receives (begin_idx, end_idx) range.
     * @param num_tasks Total number of work units to distribute.
     */
    void run(std::function<void(size_t, size_t)> function, size_t num_tasks) {
        this->function = std::move(function);
        this->num_tasks = num_tasks;
        for (size_t i = 0; i < threads.size(); ++i) {
            auto &mtx = *mtxes[i];
            std::lock_guard<std::mutex> lk(mtx);
            has_function[i] = true;
            finished[i] = false;
        }
        for (size_t i = 0; i < threads.size(); ++i) {
            auto &cv = *cvs[i];
            cv.notify_one();
        }
        for (size_t i = 0; i < threads.size(); ++i) {
            auto &mtx = *mtxes[i];
            auto &cv = *cvs[i];
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [this, i] { return finished[i]; });
        }
    }
};

/// Global filter thread pool with 12 threads for CSV loading.
inline FilterThreadPool filter_tp(12);

/**
 * @struct InnerColumnBase
 * @brief Base class for type-erased columnar data access.
 *
 * Provides a common interface for heterogeneous column storage in InnerTable.
 * Derived classes (InnerColumn<T>) provide type-specific data and operations.
 */
struct InnerColumnBase {
    DataType type; ///< Runtime type tag for downcasting.

    /**
     * @brief Construct with the specified data type.
     * @param type The column's data type.
     */
    InnerColumnBase(DataType type) : type(type) {}

    virtual ~InnerColumnBase() {}
};

/**
 * @struct InnerColumn
 * @brief Type-specific columnar storage with parallel filter operations.
 *
 * Stores column data in a contiguous vector with a separate null bitmap.
 * Provides comparison operations that return a bitmap of matching rows,
 * executed in parallel using the global filter_tp thread pool.
 *
 * ### Storage Layout
 * - `data`: Contiguous vector of T values (undefined for null rows)
 * - `bitmap`: Packed bits, 1 = non-null, 0 = null; bit i in byte i/8
 *
 * ### Comparison Semantics
 * All comparisons return false for null values (SQL three-valued logic).
 * Results are packed bitmaps where bit i = 1 means row i matches.
 *
 * @tparam T The element type (int32_t, int64_t, or double).
 *
 * @see InnerColumn<std::string> for the VARCHAR specialization
 */
template <class T> struct InnerColumn : InnerColumnBase {
    /**
     * @brief Get the DataType enum value for this column type.
     * @return DataType::INT32, INT64, or FP64 based on T.
     */
    static constexpr DataType data_type() {
        if constexpr (std::is_same_v<T, int32_t>) {
            return DataType::INT32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return DataType::INT64;
        } else if constexpr (std::is_same_v<T, double>) {
            return DataType::FP64;
        }
    }

    InnerColumn() : InnerColumnBase(data_type()) {}

    std::vector<T> data;         ///< Column values (index = row number).
    std::vector<uint8_t> bitmap; ///< Null bitmap (1 bit per row, packed).

    /**
     * @brief Update the null bitmap for the current row.
     * @param not_null True if the value is non-null.
     */
    void bitmap_push_back(bool not_null) {
        if ((data.size() + 7) / 8 > bitmap.size()) {
            if (not_null) {
                bitmap.push_back(0x01);
            } else {
                bitmap.push_back(0x00);
            }
        } else {
            size_t byte_idx = (data.size() - 1) / 8;
            size_t bit_idx = (data.size() - 1) % 8;
            if (not_null) {
                bitmap[byte_idx] |= (0x1 << bit_idx);
            } else {
                bitmap[byte_idx] &= ~(0x1 << bit_idx);
            }
        }
    }

    /**
     * @brief Append a non-null value to the column.
     * @param value The value to append.
     */
    void push_back(T value) {
        data.emplace_back(value);
        bitmap_push_back(true);
    }

    /**
     * @brief Append a null value to the column.
     */
    void push_back_null() {
        data.emplace_back();
        bitmap_push_back(false);
    }

    /**
     * @brief Check if a row contains a non-null value.
     * @param idx Row index.
     * @return True if the value is non-null.
     */
    bool is_not_null(size_t idx) const {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        return bitmap[byte_idx] & (0x1 << bit_idx);
    }

    /**
     * @brief Get the value at a row (undefined if null).
     * @param idx Row index.
     * @return The value (check is_not_null first).
     */
    T get(size_t idx) const { return data[idx]; }

    /// @name Parallel Comparison Operations
    /// @brief Return bitmap where bit i = 1 if row i matches the predicate.
    /// @{

    /** @brief Find rows where value < rhs. */
    std::vector<uint8_t> less(T rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, rhs, &ret](size_t byte_begin, size_t byte_end) {
            less(data.data() + byte_begin * 8, bitmap.data() + byte_begin,
                 ret.data() + byte_begin,
                 std::min(byte_end * 8, data.size()) - byte_begin * 8, rhs);
        };
        filter_tp.run(task, (data.size() + 7) / 8);
        return ret;
    }

    /** @brief Find rows where value > rhs. */
    std::vector<uint8_t> greater(T rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, rhs, &ret](size_t byte_begin, size_t byte_end) {
            greater(data.data() + byte_begin * 8, bitmap.data() + byte_begin,
                    ret.data() + byte_begin,
                    std::min(byte_end * 8, data.size()) - byte_begin * 8, rhs);
        };
        filter_tp.run(task, (data.size() + 7) / 8);
        return ret;
    }

    /** @brief Find rows where value <= rhs. */
    std::vector<uint8_t> less_equal(T rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, rhs, &ret](size_t byte_begin, size_t byte_end) {
            less_equal(data.data() + byte_begin * 8, bitmap.data() + byte_begin,
                       ret.data() + byte_begin,
                       std::min(byte_end * 8, data.size()) - byte_begin * 8,
                       rhs);
        };
        filter_tp.run(task, (data.size() + 7) / 8);
        return ret;
    }

    /** @brief Find rows where value >= rhs. */
    std::vector<uint8_t> greater_equal(T rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, rhs, &ret](size_t byte_begin, size_t byte_end) {
            greater_equal(data.data() + byte_begin * 8,
                          bitmap.data() + byte_begin, ret.data() + byte_begin,
                          std::min(byte_end * 8, data.size()) - byte_begin * 8,
                          rhs);
        };
        filter_tp.run(task, (data.size() + 7) / 8);
        return ret;
    }

    /** @brief Find rows where value == rhs. */
    std::vector<uint8_t> equal(T rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, rhs, &ret](size_t byte_begin, size_t byte_end) {
            equal(data.data() + byte_begin * 8, bitmap.data() + byte_begin,
                  ret.data() + byte_begin,
                  std::min(byte_end * 8, data.size()) - byte_begin * 8, rhs);
        };
        filter_tp.run(task, (data.size() + 7) / 8);
        return ret;
    }

    /** @brief Find rows where value != rhs. */
    std::vector<uint8_t> not_equal(T rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, rhs, &ret](size_t byte_begin, size_t byte_end) {
            not_equal(data.data() + byte_begin * 8, bitmap.data() + byte_begin,
                      ret.data() + byte_begin,
                      std::min(byte_end * 8, data.size()) - byte_begin * 8,
                      rhs);
        };
        filter_tp.run(task, (data.size() + 7) / 8);
        return ret;
    }

    /// @}

    /// @name Low-level Comparison Kernels
    /// @brief Static functions for batch comparison; called by parallel tasks.
    /// @{

    static void less(const T *__restrict__ data,
                     const uint8_t *__restrict__ bitmap,
                     uint8_t *__restrict__ output, size_t size, T rhs) {
        for (size_t i = 0; i < size; ++i) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            output[byte_idx] |=
                (bitmap[byte_idx] & (0x1 << bit_idx) &
                 (static_cast<uint8_t>(data[i] < rhs) << bit_idx));
        }
    }

    static void greater(const T *__restrict__ data,
                        const uint8_t *__restrict__ bitmap,
                        uint8_t *__restrict__ output, size_t size, T rhs) {
        for (size_t i = 0; i < size; ++i) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            output[byte_idx] |=
                (bitmap[byte_idx] & (0x1 << bit_idx) &
                 (static_cast<uint8_t>(data[i] > rhs) << bit_idx));
        }
    }

    static void less_equal(const T *__restrict__ data,
                           const uint8_t *__restrict__ bitmap,
                           uint8_t *__restrict__ output, size_t size, T rhs) {
        for (size_t i = 0; i < size; ++i) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            output[byte_idx] |=
                (bitmap[byte_idx] & (0x1 << bit_idx) &
                 (static_cast<uint8_t>(data[i] <= rhs) << bit_idx));
        }
    }

    static void greater_equal(const T *__restrict__ data,
                              const uint8_t *__restrict__ bitmap,
                              uint8_t *__restrict__ output, size_t size,
                              T rhs) {
        for (size_t i = 0; i < size; ++i) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            output[byte_idx] |=
                (bitmap[byte_idx] & (0x1 << bit_idx) &
                 (static_cast<uint8_t>(data[i] >= rhs) << bit_idx));
        }
    }

    static void equal(const T *__restrict__ data,
                      const uint8_t *__restrict__ bitmap,
                      uint8_t *__restrict__ output, size_t size, T rhs) {
        for (size_t i = 0; i < size; ++i) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            output[byte_idx] |=
                (bitmap[byte_idx] & (0x1 << bit_idx) &
                 (static_cast<uint8_t>(data[i] == rhs) << bit_idx));
        }
    }

    static void not_equal(const T *__restrict__ data,
                          const uint8_t *__restrict__ bitmap,
                          uint8_t *__restrict__ output, size_t size, T rhs) {
        for (size_t i = 0; i < size; ++i) {
            size_t byte_idx = i / 8;
            size_t bit_idx = i % 8;
            output[byte_idx] |=
                (bitmap[byte_idx] & (0x1 << bit_idx) &
                 (static_cast<uint8_t>(data[i] != rhs) << bit_idx));
        }
    }

    /// @}
};

/**
 * @struct InnerColumn<std::string>
 * @brief Specialization of InnerColumn for variable-length strings (VARCHAR).
 *
 * Uses a different storage layout than the fixed-size template:
 * - `data`: Flat char buffer containing all strings concatenated
 * - `offsets`: End offset of each string (offsets[i] = end of string i)
 * - `bitmap`: Null bitmap as in the base template
 *
 * String i spans bytes [offsets[i-1], offsets[i]) in the data buffer
 * (with offsets[-1] implicitly 0).
 *
 * Provides the same comparison interface as InnerColumn<T>, plus
 * LIKE/NOT LIKE pattern matching for SQL LIKE predicates.
 */
template <> struct InnerColumn<std::string> : InnerColumnBase {
    /** @brief Get the DataType for string columns. */
    static constexpr DataType data_type() { return DataType::VARCHAR; }

    InnerColumn() : InnerColumnBase(data_type()) {}

    std::vector<char> data;      ///< Concatenated string data.
    std::vector<size_t> offsets; ///< End offset of each string in data.
    std::vector<uint8_t> bitmap; ///< Null bitmap (1 bit per row).
    size_t row = 0;              ///< Current row count.

    /**
     * @brief Update the null bitmap for the current row.
     * @param not_null True if the value is non-null.
     */
    void bitmap_push_back(bool not_null) {
        if (row / 8 + 1 > bitmap.size()) {
            if (not_null) {
                bitmap.push_back(0x01);
            } else {
                bitmap.push_back(0x00);
            }
        } else {
            size_t byte_idx = row / 8;
            size_t bit_idx = row % 8;
            if (not_null) {
                bitmap[byte_idx] |= (0x1 << bit_idx);
            } else {
                bitmap[byte_idx] &= ~(0x1 << bit_idx);
            }
        }
        row += 1;
    }

    /**
     * @brief Append a non-null string value.
     * @param value The string to append.
     */
    void push_back(std::string_view value) {
        data.insert(data.end(), value.begin(), value.end());
        offsets.emplace_back(data.size());
        bitmap_push_back(true);
    }

    /** @brief Append a null value. */
    void push_back_null() {
        offsets.emplace_back(data.size());
        bitmap_push_back(false);
    }

    /**
     * @brief Check if a row contains a non-null value.
     * @param idx Row index.
     * @return True if the value is non-null.
     */
    bool is_not_null(size_t idx) const {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        return bitmap[byte_idx] & (0x1 << bit_idx);
    }

    /**
     * @brief Get the string value at a row.
     * @param idx Row index.
     * @return View of the string (check is_not_null first).
     */
    std::string_view get(size_t idx) const {
        size_t begin;
        if (idx == 0) [[unlikely]] {
            begin = 0;
        } else {
            begin = offsets[idx - 1];
        }
        size_t end = offsets[idx];
        return std::string_view{data.data() + begin, end - begin};
    }

    /// @name Parallel Comparison Operations
    /// @{

    /** @brief Find rows where value < rhs (lexicographic). */
    std::vector<uint8_t> less(std::string_view rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, &ret, &rhs](size_t byte_begin, size_t byte_end) {
            for (size_t byte_idx = byte_begin; byte_idx < byte_end;
                 ++byte_idx) {
                for (size_t bit_idx = 0; bit_idx < 8; ++bit_idx) {
                    size_t i = byte_idx * 8 + bit_idx;
                    if (i >= row) {
                        break;
                    }
                    bool non_null = bitmap[byte_idx] & (0x1 << bit_idx);
                    size_t begin = i == 0 ? (size_t)0 : offsets[i - 1];
                    size_t end = offsets[i];
                    std::string_view value{data.data() + begin, end - begin};
                    ret[byte_idx] |=
                        (static_cast<uint8_t>(non_null and value < rhs)
                         << bit_idx);
                }
            }
        };
        filter_tp.run(task, (row + 7) / 8);
        return ret;
    }

    /** @brief Find rows where value > rhs (lexicographic). */
    std::vector<uint8_t> greater(std::string_view rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, &ret, &rhs](size_t byte_begin, size_t byte_end) {
            for (size_t byte_idx = byte_begin; byte_idx < byte_end;
                 ++byte_idx) {
                for (size_t bit_idx = 0; bit_idx < 8; ++bit_idx) {
                    size_t i = byte_idx * 8 + bit_idx;
                    if (i >= row) {
                        break;
                    }
                    bool non_null = bitmap[byte_idx] & (0x1 << bit_idx);
                    size_t begin = i == 0 ? (size_t)0 : offsets[i - 1];
                    size_t end = offsets[i];
                    std::string_view value{data.data() + begin, end - begin};
                    ret[byte_idx] |=
                        (static_cast<uint8_t>(non_null and value > rhs)
                         << bit_idx);
                }
            }
        };
        filter_tp.run(task, (row + 7) / 8);
        return ret;
    }

    /** @brief Find rows where value <= rhs (lexicographic). */
    std::vector<uint8_t> less_equal(std::string_view rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, &ret, &rhs](size_t byte_begin, size_t byte_end) {
            for (size_t byte_idx = byte_begin; byte_idx < byte_end;
                 ++byte_idx) {
                for (size_t bit_idx = 0; bit_idx < 8; ++bit_idx) {
                    size_t i = byte_idx * 8 + bit_idx;
                    if (i >= row) {
                        break;
                    }
                    bool non_null = bitmap[byte_idx] & (0x1 << bit_idx);
                    size_t begin = i == 0 ? (size_t)0 : offsets[i - 1];
                    size_t end = offsets[i];
                    std::string_view value{data.data() + begin, end - begin};
                    ret[byte_idx] |=
                        (static_cast<uint8_t>(non_null and value <= rhs)
                         << bit_idx);
                }
            }
        };
        filter_tp.run(task, (row + 7) / 8);
        return ret;
    }

    /** @brief Find rows where value >= rhs (lexicographic). */
    std::vector<uint8_t> greater_equal(std::string_view rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, &ret, &rhs](size_t byte_begin, size_t byte_end) {
            for (size_t byte_idx = byte_begin; byte_idx < byte_end;
                 ++byte_idx) {
                for (size_t bit_idx = 0; bit_idx < 8; ++bit_idx) {
                    size_t i = byte_idx * 8 + bit_idx;
                    if (i >= row) {
                        break;
                    }
                    bool non_null = bitmap[byte_idx] & (0x1 << bit_idx);
                    size_t begin = i == 0 ? (size_t)0 : offsets[i - 1];
                    size_t end = offsets[i];
                    std::string_view value{data.data() + begin, end - begin};
                    ret[byte_idx] |=
                        (static_cast<uint8_t>(non_null and value >= rhs)
                         << bit_idx);
                }
            }
        };
        filter_tp.run(task, (row + 7) / 8);
        return ret;
    }

    /** @brief Find rows where value == rhs. */
    std::vector<uint8_t> equal(std::string_view rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, &ret, &rhs](size_t byte_begin, size_t byte_end) {
            for (size_t byte_idx = byte_begin; byte_idx < byte_end;
                 ++byte_idx) {
                for (size_t bit_idx = 0; bit_idx < 8; ++bit_idx) {
                    size_t i = byte_idx * 8 + bit_idx;
                    if (i >= row) {
                        break;
                    }
                    bool non_null = bitmap[byte_idx] & (0x1 << bit_idx);
                    size_t begin = i == 0 ? (size_t)0 : offsets[i - 1];
                    size_t end = offsets[i];
                    std::string_view value{data.data() + begin, end - begin};
                    ret[byte_idx] |=
                        (static_cast<uint8_t>(non_null and value == rhs)
                         << bit_idx);
                }
            }
        };
        filter_tp.run(task, (row + 7) / 8);
        return ret;
    }

    /** @brief Find rows where value != rhs. */
    std::vector<uint8_t> not_equal(std::string_view rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, &ret, &rhs](size_t byte_begin, size_t byte_end) {
            for (size_t byte_idx = byte_begin; byte_idx < byte_end;
                 ++byte_idx) {
                for (size_t bit_idx = 0; bit_idx < 8; ++bit_idx) {
                    size_t i = byte_idx * 8 + bit_idx;
                    if (i >= row) {
                        break;
                    }
                    bool non_null = bitmap[byte_idx] & (0x1 << bit_idx);
                    size_t begin = i == 0 ? (size_t)0 : offsets[i - 1];
                    size_t end = offsets[i];
                    std::string_view value{data.data() + begin, end - begin};
                    ret[byte_idx] |=
                        (static_cast<uint8_t>(non_null and value != rhs)
                         << bit_idx);
                }
            }
        };
        filter_tp.run(task, (row + 7) / 8);
        return ret;
    }

    /**
     * @brief Find rows where value matches SQL LIKE pattern.
     *
     * Converts LIKE pattern to regex (% -> .*, _ -> .) and caches
     * compiled regex per thread for performance.
     *
     * @param rhs LIKE pattern (e.g., "%foo%", "bar_").
     * @return Bitmap of matching rows.
     */
    std::vector<uint8_t> like(const std::string &rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, &ret, &rhs](size_t byte_begin, size_t byte_end) {
            for (size_t byte_idx = byte_begin; byte_idx < byte_end;
                 ++byte_idx) {
                for (size_t bit_idx = 0; bit_idx < 8; ++bit_idx) {
                    size_t i = byte_idx * 8 + bit_idx;
                    if (i >= row) {
                        break;
                    }
                    bool non_null = bitmap[byte_idx] & (0x1 << bit_idx);
                    size_t begin = i == 0 ? (size_t)0 : offsets[i - 1];
                    size_t end = offsets[i];
                    std::string_view value{data.data() + begin, end - begin};
                    ret[byte_idx] |=
                        (static_cast<uint8_t>(
                             non_null and Comparison::like_match(value, rhs))
                         << bit_idx);
                }
            }
        };
        filter_tp.run(task, (row + 7) / 8);
        return ret;
    }

    /**
     * @brief Find rows where value does NOT match SQL LIKE pattern.
     * @param rhs LIKE pattern to negate.
     * @return Bitmap of non-matching rows.
     */
    std::vector<uint8_t> not_like(const std::string &rhs) const {
        std::vector<uint8_t> ret(bitmap.size());
        auto task = [this, &ret, &rhs](size_t byte_begin, size_t byte_end) {
            for (size_t byte_idx = byte_begin; byte_idx < byte_end;
                 ++byte_idx) {
                for (size_t bit_idx = 0; bit_idx < 8; ++bit_idx) {
                    size_t i = byte_idx * 8 + bit_idx;
                    if (i >= row) {
                        break;
                    }
                    bool non_null = bitmap[byte_idx] & (0x1 << bit_idx);
                    size_t begin = i == 0 ? (size_t)0 : offsets[i - 1];
                    size_t end = offsets[i];
                    std::string_view value{data.data() + begin, end - begin};
                    ret[byte_idx] |=
                        (static_cast<uint8_t>(
                             non_null and
                             not Comparison::like_match(value, rhs))
                         << bit_idx);
                }
            }
        };
        filter_tp.run(task, (row + 7) / 8);
        return ret;
    }
    /// @}
};

/**
 * @struct InnerTable
 * @brief Collection of InnerColumn objects representing a table.
 *
 * Owns the column storage via unique_ptr. Used during CSV loading to
 * accumulate rows before converting to ColumnarTable format.
 */
struct InnerTable {
    size_t rows; ///< Number of rows in the table.
    std::vector<std::unique_ptr<InnerColumnBase>> columns; ///< Owned columns.
};

/**
 * @struct InnerTableView
 * @brief Non-owning view into an InnerTable's columns.
 *
 * Provides read-only access to columns for filter evaluation without
 * copying or transferring ownership.
 */
struct InnerTableView {
    size_t rows; ///< Number of rows in the referenced table.
    std::vector<const InnerColumnBase *> columns; ///< Non-owning column ptrs.

    InnerTableView() = default;

    /**
     * @brief Construct a view from an InnerTable.
     * @param table The table to create a view of.
     */
    InnerTableView(const InnerTable &table) : rows(table.rows) {
        for (auto &c : table.columns) {
            columns.push_back(c.get());
        }
    }
};

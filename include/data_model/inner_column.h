/**
 * @file inner_column.h
 * @brief Flat columnar storage with parallel filter evaluation for CSV loading.
 *
 * Unlike ColumnarTable (8KB pages), uses flat vectors with packed null bitmaps
 * for SIMD-style filtering during CSV load.
 *
 * - InnerColumn<T>: contiguous data + 1-bit/row null bitmap, parallel comparisons
 * - InnerColumn<string>: flat char buffer + offset array for VARCHARs
 * - FilterThreadPool: partitions work by bitmap bytes (8 rows/unit)
 *
 * Null semantics: comparisons with NULL return false (SQL three-valued logic).
 *
 * @see ColumnarTable, Statement
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
 * @brief Thread pool for parallel filter evaluation during CSV loading.
 *
 * Per-thread mutex/CV synchronization. Work partitioned by bitmap byte (8 rows).
 * Simpler than Contest::WorkerPool, specialized for filter predicates.
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

    /** @brief Starting task index for thread (even distribution, remainder to earlier). */
    size_t begin_idx(size_t thread_id) {
        size_t base = num_tasks / threads.size();
        size_t rem = num_tasks % threads.size();
        return thread_id * base + std::min(thread_id, rem);
    }

    /** @brief Worker loop: wait → execute range → signal done. Exits on destructed. */
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

    /** @brief Spawn num_threads workers. */
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

    /** @brief Signal shutdown and join all workers. */
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

    /** @brief Execute function(begin, end) in parallel. Blocks until complete. */
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

/// Global filter thread pool (12 threads) for CSV loading.
inline FilterThreadPool filter_tp(12);

/**
 * @struct InnerColumnBase
 * @brief Type-erased column base. Runtime type tag for downcasting.
 */
struct InnerColumnBase {
    DataType type; ///< Runtime type tag.

    InnerColumnBase(DataType type) : type(type) {}
    virtual ~InnerColumnBase() {}
};

/**
 * @struct InnerColumn
 * @brief Typed column: contiguous data vector + packed null bitmap.
 *
 * Parallel comparisons via filter_tp return result bitmaps (bit i=1 → match).
 * NULL comparisons always false (SQL semantics).
 *
 * @tparam T int32_t, int64_t, or double.
 * @see InnerColumn<std::string> for VARCHAR specialization.
 */
template <class T> struct InnerColumn : InnerColumnBase {
    /** @brief DataType for T. */
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

    std::vector<T> data;         ///< Column values (row index).
    std::vector<uint8_t> bitmap; ///< Null bitmap (1 bit/row, packed).

    /** @brief Update null bitmap for current row. */
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

    /** @brief Append non-null value. */
    void push_back(T value) {
        data.emplace_back(value);
        bitmap_push_back(true);
    }

    /** @brief Append null. */
    void push_back_null() {
        data.emplace_back();
        bitmap_push_back(false);
    }

    /** @brief True if row idx is non-null. */
    bool is_not_null(size_t idx) const {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        return bitmap[byte_idx] & (0x1 << bit_idx);
    }

    /** @brief Value at idx (undefined if null). */
    T get(size_t idx) const { return data[idx]; }

    /// @name Parallel Comparisons (return bitmap: bit i=1 → row i matches)
    /// @{

    /** @brief Rows where value < rhs. */
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

    /** @brief Rows where value > rhs. */
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

    /** @brief Rows where value <= rhs. */
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

    /** @brief Rows where value >= rhs. */
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

    /** @brief Rows where value == rhs. */
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

    /** @brief Rows where value != rhs. */
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

    /// @name Low-level Kernels (static, batch comparison for parallel tasks)
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
 * @brief VARCHAR specialization: flat char buffer + offset array.
 *
 * String i = data[offsets[i-1]..offsets[i]). Same comparison interface
 * as InnerColumn<T> plus LIKE/NOT LIKE pattern matching.
 */
template <> struct InnerColumn<std::string> : InnerColumnBase {
    /** @brief Returns DataType::VARCHAR. */
    static constexpr DataType data_type() { return DataType::VARCHAR; }

    InnerColumn() : InnerColumnBase(data_type()) {}

    std::vector<char> data;      ///< Concatenated string bytes.
    std::vector<size_t> offsets; ///< End offset per string.
    std::vector<uint8_t> bitmap; ///< Null bitmap (1 bit/row).
    size_t row = 0;              ///< Row count.

    /** @brief Update null bitmap for current row. */
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

    /** @brief Append non-null string. */
    void push_back(std::string_view value) {
        data.insert(data.end(), value.begin(), value.end());
        offsets.emplace_back(data.size());
        bitmap_push_back(true);
    }

    /** @brief Append null. */
    void push_back_null() {
        offsets.emplace_back(data.size());
        bitmap_push_back(false);
    }

    /** @brief True if row idx is non-null. */
    bool is_not_null(size_t idx) const {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        return bitmap[byte_idx] & (0x1 << bit_idx);
    }

    /** @brief String view at idx (check is_not_null first). */
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

    /// @name Parallel Comparisons (lexicographic, return bitmap)
    /// @{

    /** @brief Rows where value < rhs. */
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

    /** @brief Rows where value > rhs. */
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

    /** @brief Rows where value <= rhs. */
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

    /** @brief Rows where value >= rhs. */
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

    /** @brief Rows where value == rhs. */
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

    /** @brief Rows where value != rhs. */
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

    /** @brief Rows matching SQL LIKE pattern. % → .*, _ → . @see Comparison::like_match */
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

    /** @brief Rows NOT matching SQL LIKE pattern. */
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
 * @brief Owned collection of InnerColumns. Used during CSV load before
 * conversion to ColumnarTable.
 */
struct InnerTable {
    size_t rows;                                           ///< Row count.
    std::vector<std::unique_ptr<InnerColumnBase>> columns; ///< Owned columns.
};

/**
 * @struct InnerTableView
 * @brief Non-owning view into InnerTable columns for filter evaluation.
 */
struct InnerTableView {
    size_t rows;                                  ///< Row count.
    std::vector<const InnerColumnBase *> columns; ///< Non-owning ptrs.

    InnerTableView() = default;

    /** @brief Create view from InnerTable. */
    InnerTableView(const InnerTable &table) : rows(table.rows) {
        for (auto &c : table.columns) {
            columns.push_back(c.get());
        }
    }
};

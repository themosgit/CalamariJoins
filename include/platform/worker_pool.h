/**
 * @file worker_pool.h
 * @brief Global thread pool for parallel join and materialization operations.
 *
 * Provides persistent worker threads (SPC__THREAD_COUNT) to avoid thread
 * creation overhead. Uses generation-based task dispatch with condition
 * variables for synchronization.
 *
 * @see execute() for dispatching parallel work.
 */
#pragma once

#if defined(__APPLE__) && defined(__aarch64__)
#include <platform/hardware_darwin.h>
#elif defined(SPC__USE_BENCHMARKVM_HARDWARE)
#include <platform/hardware_benchmarkvm.h>
#else
#include <platform/hardware.h>
#endif

#include <atomic>
#include <condition_variable>
#include <foundation/common.h>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
/**
 * @namespace Contest::platform
 * @brief Platform runtime and threading infrastructure.
 *
 * Key components in this file:
 * - WorkerThreadPool: Generation-based parallel task dispatch
 * - worker_pool: Global thread pool instance (SPC__THREAD_COUNT threads)
 * - execute(): Barrier-style parallel work dispatch
 *
 * @see Contest::join for hash join algorithms using this pool
 * @see Contest::materialize for result materialization using this pool
 */
namespace Contest::platform {

/**
 * @brief Global thread pool for parallel join and materialization operations.
 *
 * Creates SPC__THREAD_COUNT persistent worker threads at startup to eliminate
 * thread creation/destruction overhead for each parallel phase.
 *
 * **Generation-Based Task Dispatch**:
 * Uses a generation counter to prevent the ABA problem where workers might
 * execute stale tasks. Each execute() call increments the generation, ensuring
 * workers only process the most recent task even if they wake spuriously.
 *
 * **Synchronization Primitives**:
 * - pool_mutex: Protects task_generation, current_task, and should_exit
 * - worker_cv: Signals workers when new task is available
 * - main_cv: Signals main thread when all workers complete
 * - tasks_remaining: Lock-free atomic counter for completion detection
 *
 * **Thread Safety**:
 * Provides barrier semantics - execute() blocks until all workers complete
 * their assigned work. Safe for concurrent execute() calls (serialized by
 * mutex).
 *
 * @example
 * @code
 * // Parallel array processing across worker threads
 * worker_pool.execute([&](size_t thread_id) {
 *     size_t start = thread_id * batch_size;
 *     size_t end = std::min(start + batch_size, total_work);
 *     for (size_t i = start; i < end; ++i) {
 *         process_item(i);
 *     }
 * });
 * // All threads guaranteed complete here
 * @endcode
 */
class WorkerThreadPool {
  private:
    /** @brief Number of worker threads - constexpr for compile-time
     * optimization */
    static constexpr int NUM_THREADS = SPC__THREAD_COUNT;

    /** @brief Persistent worker thread storage */
    std::vector<std::thread> threads;

    /**
     * @brief Protects task_generation, current_task, and should_exit.
     * Serializes execute() calls and coordinates task dispatch.
     */
    std::mutex pool_mutex;

    /**
     * @brief Signals workers when new task available (main → workers).
     * Workers wait on this when idle, woken by execute().
     */
    std::condition_variable worker_cv;

    /**
     * @brief Signals main thread when all workers complete (workers → main).
     * Main thread waits on this in execute() for barrier semantics.
     */
    std::condition_variable main_cv;

    /**
     * @brief Lock-free completion counter with acquire/release ordering.
     * Decremented atomically by workers, preventing mutex contention during
     * task completion. Only the last worker (counter reaches 0) signals
     * main_cv.
     */
    std::atomic<int> tasks_remaining{0};

    /**
     * @brief Generation counter preventing ABA problem in task dispatch.
     * Incremented on each execute() call. Workers compare against their
     * last_generation to detect new tasks, avoiding stale task execution
     * on spurious wakeups or delayed scheduling.
     */
    int task_generation = 0;

    /** @brief Shutdown flag - signals workers to exit from their loop */
    bool should_exit = false;

    /** @brief Current task callable - copied by workers while holding lock */
    std::function<void(size_t)> current_task;

    /**
     * @brief Worker thread main loop - waits for tasks and executes them.
     *
     * @param thread_id Zero-based thread identifier (0 to NUM_THREADS-1),
     *                  passed to task callable for work partitioning.
     *
     * **Generation Tracking**:
     * Maintains last_generation to detect new tasks. Wait condition triggers
     * when task_generation > last_generation, preventing stale task execution.
     *
     * **Execution Flow**:
     * 1. Wait on worker_cv for new task or shutdown signal
     * 2. Copy task callable while holding lock (safe concurrent access)
     * 3. Release lock before executing task (minimizes contention)
     * 4. Execute task with thread_id for work partitioning
     * 5. Atomically decrement tasks_remaining (acq_rel ordering)
     * 6. Last worker (counter reaches 0) signals main_cv
     *
     * **Why acq_rel ordering**:
     * Ensures all task writes visible to main thread when counter reaches 0.
     */
    void worker_loop(size_t thread_id) {
        int last_generation = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(pool_mutex);
            worker_cv.wait(lock, [&] {
                return task_generation > last_generation || should_exit;
            });
            if (should_exit)
                break;

            auto local_task = current_task;
            int current_gen = task_generation;
            lock.unlock();

            local_task(thread_id);
            last_generation = current_gen;
            if (tasks_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> notify_lock(pool_mutex);
                main_cv.notify_one();
            }
        }
    }

  public:
    WorkerThreadPool() {
        threads.reserve(NUM_THREADS);

        for (int t = 0; t < NUM_THREADS; ++t) {
            threads.emplace_back([this, t] { worker_loop(t); });
        }
    }

    ~WorkerThreadPool() {
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            should_exit = true;
        }
        worker_cv.notify_all();
        for (auto &thread : threads) {
            thread.join();
        }
    }

    /**
     * @brief Dispatches task to all workers and waits for completion.
     *
     * @param task Callable invoked with thread_id (0 to NUM_THREADS-1).
     *             Each worker receives unique thread_id for work partitioning.
     *
     * **Barrier Semantics**:
     * Blocks until all workers complete their assigned work. Provides full
     * memory barrier - all task writes visible to caller upon return.
     *
     * **Bidirectional Synchronization**:
     * - worker_cv: Main thread signals workers to start (main → workers)
     * - main_cv: Workers signal main thread on completion (workers → main)
     * Both condition variables needed for full bidirectional handshake.
     *
     * **Atomic Counter for Completion**:
     * tasks_remaining tracks active workers lock-free. Release store ensures
     * task/generation visible to workers. Acquire load ensures worker writes
     * visible to main thread. Only last worker signals main_cv to minimize
     * contention.
     *
     * **Thread Safety**:
     * Multiple execute() calls serialized by pool_mutex. Safe but not
     * concurrent - callers should avoid overlapping execute() calls.
     */
    void execute(std::function<void(size_t)> task) {
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            current_task = task;
            tasks_remaining.store(NUM_THREADS, std::memory_order_release);
            task_generation++;
        }
        worker_cv.notify_all();

        std::unique_lock<std::mutex> lock(pool_mutex);
        main_cv.wait(lock, [&] {
            return tasks_remaining.load(std::memory_order_acquire) == 0;
        });
    }

    constexpr int thread_count() const { return NUM_THREADS; }
};

inline WorkerThreadPool worker_pool;

} // namespace Contest::platform

/**
 *
 * @file worker_pool.h
 * @brief Global thread pool for parallel join and materialization operations.
 *
 * Provides persistent worker threads (SPC__THREAD_COUNT) to avoid thread
 * creation overhead. Uses generation-based task dispatch with condition
 * variables for synchronization.
 *
 * @see execute() for dispatching parallel work.
 *
 **/
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

/* @namespace Contest::platform @brief Platform runtime and threading. */
namespace Contest::platform {

inline constexpr int THREAD_COUNT = SPC__THREAD_COUNT;

/**
 *
 * @brief Global thread pool for parallel join and materialization.
 *
 * SPC__THREAD_COUNT persistent workers with generation-based task dispatch.
 * Provides barrier semantics - execute() blocks until all workers complete.
 *
 **/
class WorkerThreadPool {
  private:
    static constexpr int NUM_THREADS = SPC__THREAD_COUNT;

    /* @brief Persistent worker thread storage */
    std::vector<std::thread> threads;

    /* @brief Protects task_generation, current_task, should_exit. */
    std::mutex pool_mutex;

    /* @brief Signals workers when new task available. */
    std::condition_variable worker_cv;

    /* @brief Signals main thread when all workers complete */
    std::condition_variable main_cv;

    /* @brief Lock-free completion counter with acq_rel ordering. */
    std::atomic<int> tasks_remaining{0};

    /* @brief Generation counter preventing ABA problem in task dispatch. */
    int task_generation = 0;

    /* @brief Shutdown flag - signals workers to exit from their loop */
    bool should_exit = false;

    /* @brief Current task callable - copied by workers while holding lock */
    std::function<void(size_t)> current_task;

    /**
     *
     * @brief Worker thread main loop - waits for tasks and executes them.
     *
     * @param thread_id Zero-based thread identifier for work partitioning.
     *
     **/
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
     *
     * @brief Dispatches task to all workers and waits for completion.
     *
     * @param task Callable invoked with thread_id (0 to NUM_THREADS-1).
     * 
     **/
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

inline WorkerThreadPool g_worker_pool{};

inline WorkerThreadPool &worker_pool() { return g_worker_pool; }

} // namespace Contest::platform

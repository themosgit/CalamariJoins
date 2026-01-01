#pragma once

#if defined(__APPLE__) && defined(__aarch64__)
    #include <hardware_darwin.h>
#else
    #include <hardware.h>
#endif

#include <common.h>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>
namespace Contest {

/**
 *
 *  global thread pool for join/materialization operations
 *  creates SPC__CORE_COUNT persistent worker threads
 *  eliminates thread creation/destruction overhead
 *
 **/
class WorkerThreadPool {
  private:
    static constexpr int NUM_THREADS = SPC__CORE_COUNT;

    std::vector<std::thread> threads;
    std::mutex pool_mutex;
    std::condition_variable worker_cv;
    std::condition_variable main_cv;
    std::atomic<int> tasks_remaining{0};
    int task_generation = 0;
    bool should_exit = false;

    std::function<void(size_t, size_t)> current_task;

    void worker_loop(size_t thread_id) {
        int last_generation = 0;
        while (true) {
            std::unique_lock<std::mutex> lock(pool_mutex);
            worker_cv.wait(lock, [&] {
                return task_generation > last_generation || should_exit;
            });
            if (should_exit)
                break;

            /* copy task while holding lock, then unlock */
            auto local_task = current_task;
            int current_gen = task_generation;
            lock.unlock();

            local_task(thread_id, NUM_THREADS);
            last_generation = current_gen;
            if (tasks_remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
                main_cv.notify_one();
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

    void execute(std::function<void(size_t, size_t)> task) {
        {
            std::lock_guard<std::mutex> lock(pool_mutex);
            current_task = task;
            tasks_remaining.store(NUM_THREADS, std::memory_order_release);
            task_generation++;
        }
        worker_cv.notify_all();
        
        /* wait for atomic counter to reach 0 */
        std::unique_lock<std::mutex> lock(pool_mutex);
        main_cv.wait(lock, [&] {
                return tasks_remaining.load(std::memory_order_acquire) == 0;
        });

    }

    constexpr int thread_count() const { return NUM_THREADS; }
};

/* global instance - instantiated once at program startup */
inline WorkerThreadPool worker_pool;

} // namespace Contest

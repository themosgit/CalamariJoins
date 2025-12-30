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
    std::vector<std::unique_ptr<std::mutex>> mtxes;
    std::vector<std::unique_ptr<std::condition_variable>> cvs;
    std::vector<uint8_t> has_task;
    std::vector<uint8_t> finished;
    std::vector<bool> should_exit;

    std::mutex task_mutex;
    std::function<void(size_t, size_t)> current_task;

    void worker_loop(size_t thread_id) {
        while (true) {
            std::unique_lock<std::mutex> lock(*mtxes[thread_id]);
            cvs[thread_id]->wait(lock, [&] {
                return has_task[thread_id] || should_exit[thread_id];
            });

            if (should_exit[thread_id])
                break;

            std::function<void(size_t, size_t)> local_task;
            /* execute task under lock to ensure proper access */
            {
                std::lock_guard<std::mutex> task_lock(task_mutex);
                local_task = current_task;
            }
            local_task(thread_id, NUM_THREADS);

            has_task[thread_id] = 0;
            finished[thread_id] = 1;
            cvs[thread_id]->notify_one();
        }
    }

  public:
    WorkerThreadPool() {
        threads.reserve(NUM_THREADS);
        mtxes.reserve(NUM_THREADS);
        cvs.reserve(NUM_THREADS);
        has_task.resize(NUM_THREADS, 0);
        finished.resize(NUM_THREADS, 1);
        should_exit.resize(NUM_THREADS, false);

        for (int t = 0; t < NUM_THREADS; ++t) {
            mtxes.push_back(std::make_unique<std::mutex>());
            cvs.push_back(std::make_unique<std::condition_variable>());
            threads.emplace_back([this, t] { worker_loop(t); });
        }
    }

    ~WorkerThreadPool() {
        for (int t = 0; t < NUM_THREADS; ++t) {
            {
                std::lock_guard<std::mutex> lock(*mtxes[t]);
                should_exit[t] = true;
                cvs[t]->notify_one();
            }
        }
        for (auto &thread : threads) {
            thread.join();
        }
    }

    void execute(std::function<void(size_t, size_t)> task) {
        /* assign task under lock to prevent data race */
        {
            std::lock_guard<std::mutex> task_lock(task_mutex);
            current_task = task;
        }

        /* wake all threads */
        for (int t = 0; t < NUM_THREADS; ++t) {
            std::lock_guard<std::mutex> lock(*mtxes[t]);
            has_task[t] = 1;
            finished[t] = 0;
            cvs[t]->notify_one();
        }

        /* wait for all threads to finish */
        for (int t = 0; t < NUM_THREADS; ++t) {
            std::unique_lock<std::mutex> lock(*mtxes[t]);
            const int thread_idx = t;  // Capture by value
            cvs[t]->wait(lock, [&, thread_idx] { return finished[thread_idx]; });
        }

    }

    constexpr int thread_count() const { return NUM_THREADS; }
};

/* global instance - instantiated once at program startup */
inline WorkerThreadPool worker_pool;

} // namespace Contest

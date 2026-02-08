/*
 * Copyright 2025 The llm-d Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "simple_thread_pool.hpp"
#include <iostream>

// SimpleThreadPool constructor
SimpleThreadPool::SimpleThreadPool(size_t num_threads) {
  // Create all worker threads
  for (size_t i = 0; i < num_threads; ++i) {
    m_workers.emplace_back([this] {
      // Worker loop
      while (true) {
        std::function<void()> task;
        {
          // Lock the task queue before checking it
          std::unique_lock<std::mutex> lock(m_queue_mutex);

          // Wait until either a new task arrives or the pool is stopping
          // (wait() unlocks the mutex while sleeping and re-locks it when waking)
          m_condition.wait(lock, [this] { return m_stop || !m_tasks.empty(); });

          // Exit thread if pool is stopping and no more tasks
          if (m_stop && m_tasks.empty()) {
            return;
          }

          // Fetch next task from the queue
          task = std::move(m_tasks.front());
          m_tasks.pop();
        }

        try {
          // Execute the task
          task();
        } catch (const std::exception& e) {
          std::cerr << "[ERROR] Exception in worker thread: " << e.what()
                    << "\n";
        } catch (...) {
          std::cerr << "[ERROR] Unknown exception in worker thread\n";
        }
      }
    });
  }
}

// SimpleThreadPool destructor
SimpleThreadPool::~SimpleThreadPool() {
  {
    std::unique_lock<std::mutex> lock(m_queue_mutex);
    m_stop = true;
  }
  m_condition.notify_all();

  // Wait for all worker threads to exit
  for (std::thread& worker : m_workers) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

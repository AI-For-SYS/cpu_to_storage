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

#pragma once

#include <vector>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

// SimpleThreadPool: A lightweight CPU-only thread pool for file I/O operations
// without CUDA dependencies. Designed for parallel file read/write operations.
class SimpleThreadPool {
 public:
  // Constructor: creates num_threads worker threads
  explicit SimpleThreadPool(size_t num_threads);

  // Destructor: stops all worker threads and waits for them to finish
  ~SimpleThreadPool();

  // Enqueue a task to be executed by the thread pool
  // Returns a future that will contain the result of the task
  template <class F>
  auto enqueue(F&& f) -> std::future<std::invoke_result_t<F>>;

 private:
  std::vector<std::thread> m_workers;         // All worker threads
  std::queue<std::function<void()>> m_tasks;  // Queue of pending tasks

  std::mutex m_queue_mutex;  // Protects access to the task queue
  std::condition_variable
      m_condition;  // Signals workers when tasks are available

  std::atomic<bool> m_stop{false};  // Tells workers to stop and exit
};

// Template implementation must be in header file
template <class F>
auto SimpleThreadPool::enqueue(F&& f) -> std::future<std::invoke_result_t<F>> {
  // Get the return type of the submitted task
  using return_type = std::invoke_result_t<F>;

  // Wrap the callable into a packaged_task so we can return a future
  auto task =
      std::make_shared<std::packaged_task<return_type()>>(std::forward<F>(f));

  // Future for the caller to wait on
  std::future<return_type> res = task->get_future();

  {
    std::unique_lock<std::mutex> lock(m_queue_mutex);

    // Reject new tasks if the pool is shutting down
    if (m_stop) {
      throw std::runtime_error("Cannot enqueue task: thread pool is stopped");
    }

    // Push the task wrapper into the queue
    m_tasks.emplace([task]() { (*task)(); });
  }

  // Wake one worker thread to process the task
  m_condition.notify_one();

  return res;
}


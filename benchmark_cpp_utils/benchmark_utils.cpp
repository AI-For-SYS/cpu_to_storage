#include <torch/extension.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <future>
#include <thread>
#include <cstdlib>
#include <memory>
#include <mutex>

#include "file_io.hpp"
#include "simple_thread_pool.hpp"

// Global thread pool configuration
static std::mutex g_pool_mutex;
static std::unique_ptr<SimpleThreadPool> g_thread_pool;
static size_t g_thread_count = 0;

// Helper function to get thread count from environment or use default
static size_t get_default_thread_count() {
  const char* env_threads = std::getenv("IO_THREAD_COUNT");
  if (env_threads) {
    try {
      size_t count = std::stoul(env_threads);
      if (count > 0) {
        return count;
      }
    } catch (...) {
      // Fall through to default
    }
  }
  // Default to hardware concurrency
  size_t hw_threads = std::thread::hardware_concurrency();
  return hw_threads > 0 ? hw_threads : 8;
}

// Get or create the global thread pool (persistent across calls)
static SimpleThreadPool& get_thread_pool() {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  if (!g_thread_pool) {
    if (g_thread_count == 0) {
      g_thread_count = get_default_thread_count();
    }
    g_thread_pool = std::make_unique<SimpleThreadPool>(g_thread_count);
  }
  return *g_thread_pool;
}

// Set the number of threads for the thread pool
// Note: This will recreate the thread pool, so call before any I/O operations
void set_io_thread_count(size_t num_threads) {
  if (num_threads == 0) {
    throw std::runtime_error("Thread count must be greater than 0");
  }
  
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  g_thread_count = num_threads;
  // Destroy old pool and create new one with updated thread count
  g_thread_pool.reset();
  g_thread_pool = std::make_unique<SimpleThreadPool>(g_thread_count);
  
  std::cerr << "[INFO] Thread pool recreated with " << num_threads << " threads\n";
}

// Get the current thread pool size
size_t get_io_thread_count() {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  if (g_thread_count == 0) {
    g_thread_count = get_default_thread_count();
  }
  return g_thread_count;
}

// cpp_write_blocks: Write multiple blocks from a tensor to separate files
// Uses a persistent thread pool for parallel I/O operations
bool cpp_write_blocks(torch::Tensor buffer,
                        int64_t block_size,
                        std::vector<int64_t> block_indices,
                        std::vector<std::string> dest_files) {
  // Validate inputs
  if (!buffer.is_cpu()) {
    throw std::runtime_error("Buffer must be on CPU");
  }
  if (!buffer.is_contiguous()) {
    throw std::runtime_error("Buffer must be contiguous");
  }
  if (block_indices.size() != dest_files.size()) {
    throw std::runtime_error(
        "block_indices and dest_files must have the same size");
  }
  
  // Release GIL for true parallelism during I/O
  py::gil_scoped_release release;

  uint8_t* buffer_ptr = static_cast<uint8_t*>(buffer.data_ptr());
  int64_t buffer_size = buffer.numel() * buffer.element_size();
  SimpleThreadPool& pool = get_thread_pool();

  // Enqueue write tasks for each block
  std::vector<std::future<bool>> futures;
  futures.reserve(block_indices.size());

  for (size_t i = 0; i < block_indices.size(); i++) {
    int64_t block_idx = block_indices[i];
    std::string dest_file = std::move(dest_files[i]);

    // Validate block index
    int64_t block_offset = block_idx * block_size;
    if (block_offset + block_size > buffer_size) {
      throw std::runtime_error("Block index " + std::to_string(block_idx) +
                               " out of bounds for buffer size " +
                               std::to_string(buffer_size));
    }

    // Enqueue write task
    auto future = pool.enqueue([buffer_ptr, block_offset, block_size, dest_file]() -> bool {
      try {
        if (!write_buffer_to_file(buffer_ptr + block_offset, block_size, dest_file)) {
          std::cerr << "[ERROR] Failed to write to file: " << dest_file << "\n";
          return false;
        }
        return true;
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception writing block to " << dest_file << ": "
                  << e.what() << "\n";
        return false;
      }
    });

    futures.push_back(std::move(future));
  }

  // Wait for all write tasks to complete
  bool all_success = true;
  for (auto& fut : futures) {
    if (!fut.get()) {
      all_success = false;
    }
  }

  if (!all_success) {
    std::cerr << "[WARN] Some write operations failed\n";
  }

return all_success;
}

// cpp_read_blocks: Read multiple blocks from separate files into a tensor
// Uses a persistent thread pool for parallel I/O operations
bool cpp_read_blocks(torch::Tensor buffer,
                       int64_t block_size,
                       std::vector<int64_t> block_indices,
                       std::vector<std::string> source_files) {
  // Validate inputs
  if (!buffer.is_cpu()) {
    throw std::runtime_error("Buffer must be on CPU");
  }
  if (!buffer.is_contiguous()) {
    throw std::runtime_error("Buffer must be contiguous");
  }
  if (block_indices.size() != source_files.size()) {
    throw std::runtime_error(
        "block_indices and source_files must have the same size");
  }

  // Release GIL for true parallelism during I/O
  py::gil_scoped_release release;

  uint8_t* data_ptr = static_cast<uint8_t*>(buffer.data_ptr());
  int64_t buffer_size = buffer.numel() * buffer.element_size();
  SimpleThreadPool& pool = get_thread_pool();

  // Enqueue read tasks for each block
  std::vector<std::future<bool>> futures;
  futures.reserve(block_indices.size());

  for (size_t i = 0; i < block_indices.size(); i++) {
    int64_t block_idx = block_indices[i];
    std::string source_file = source_files[i];

    // Validate block index
    int64_t block_offset = block_idx * block_size;
    if (block_offset + block_size > buffer_size) {
      throw std::runtime_error("Block index " + std::to_string(block_idx) +
                               " out of bounds for buffer size " +
                               std::to_string(buffer_size));
    }

    // Enqueue read task
    auto future = pool.enqueue([data_ptr, block_offset, block_size, source_file]() -> bool {
      try {
        // Read file directly into buffer slice
        return read_file_to_buffer(source_file, data_ptr + block_offset, block_size);
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception reading block from " << source_file
                  << ": " << e.what() << "\n";
        return false;
      }
    });

    futures.push_back(std::move(future));
  }

  // Wait for all read tasks to complete
  bool all_success = true;
  for (auto& fut : futures) {
    if (!fut.get()) {
      all_success = false;
    }
  }

  if (!all_success) {
    std::cerr << "[WARN] Some read operations failed\n";
  }

  // Return time in seconds (matching Python's time.perf_counter)
  return all_success;
}

// PyBind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpp_write_blocks",
        &cpp_write_blocks,
        "Write multiple blocks from a tensor to separate files in parallel",
        py::arg("buffer"),
        py::arg("block_size"),
        py::arg("block_indices"),
        py::arg("dest_files"));

  m.def("cpp_read_blocks",
        &cpp_read_blocks,
        "Read multiple blocks from separate files into a tensor in parallel",
        py::arg("buffer"),
        py::arg("block_size"),
        py::arg("block_indices"),
        py::arg("source_files"));

  m.def("set_io_thread_count",
        &set_io_thread_count,
        "Set the number of threads for the I/O thread pool",
        py::arg("num_threads"));

  m.def("get_io_thread_count",
        &get_io_thread_count,
        "Get the current number of threads in the I/O thread pool");
}

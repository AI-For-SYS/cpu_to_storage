#include <torch/extension.h>

#include <iostream>
#include <string>
#include <vector>
#include <future>
#include <thread>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

namespace fs = std::filesystem;

#include "simple_thread_pool.hpp"

// Read block_size bytes from path into buffer_ptr using pread().
static bool pread_file(const std::string& path, uint8_t* buffer_ptr, size_t block_size) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cerr << "[ERROR] Failed to open file: " << path
              << " - " << std::strerror(errno) << "\n";
    return false;
  }
  size_t total_read = 0;
  while (total_read < block_size) {
    ssize_t n = pread(fd, buffer_ptr + total_read, block_size - total_read,
                      static_cast<off_t>(total_read));
    if (n < 0) {
      if (errno == EINTR) continue;
      std::cerr << "[ERROR] pread failed: " << path
                << " - " << std::strerror(errno) << "\n";
      close(fd);
      return false;
    }
    if (n == 0) {
      std::cerr << "[ERROR] Unexpected EOF: " << path
                << " (read " << total_read << "/" << block_size << " bytes)\n";
      close(fd);
      return false;
    }
    total_read += n;
  }
  close(fd);
  return true;
}

// Global thread pool configuration
static std::mutex g_pool_mutex;
static std::unique_ptr<SimpleThreadPool> g_thread_pool;
static size_t g_thread_count = 0;

static size_t get_default_thread_count() {
  const char* env_threads = std::getenv("IO_THREAD_COUNT");
  if (env_threads) {
    try {
      size_t count = std::stoul(env_threads);
      if (count > 0) return count;
    } catch (...) {}
  }
  size_t hw_threads = std::thread::hardware_concurrency();
  return hw_threads > 0 ? hw_threads : 8;
}

static SimpleThreadPool& get_thread_pool() {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  if (!g_thread_pool) {
    if (g_thread_count == 0) g_thread_count = get_default_thread_count();
    g_thread_pool = std::make_unique<SimpleThreadPool>(g_thread_count);
  }
  return *g_thread_pool;
}

void set_thread_count(size_t num_threads) {
  if (num_threads == 0)
    throw std::runtime_error("Thread count must be greater than 0");
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  g_thread_count = num_threads;
  g_thread_pool.reset();
  g_thread_pool = std::make_unique<SimpleThreadPool>(g_thread_count);
  std::cerr << "[INFO] Thread pool recreated with " << num_threads << " threads\n";
}

size_t get_io_thread_count() {
  std::lock_guard<std::mutex> lock(g_pool_mutex);
  if (g_thread_count == 0) g_thread_count = get_default_thread_count();
  return g_thread_count;
}

// cpp_read_blocks: each worker opens its own FD and uses pread().
// For reads, files already exist so there is no O_CREAT directory-inode
// contention. Opening FDs inside worker threads lets the opens happen in
// parallel, which is faster than pre-opening them sequentially in the main thread.
bool cpp_read_blocks(torch::Tensor buffer,
                     int64_t block_size,
                     std::vector<int64_t> block_indices,
                     std::vector<std::string> source_files) {
  if (!buffer.is_cpu())
    throw std::runtime_error("Buffer must be on CPU");
  if (!buffer.is_contiguous())
    throw std::runtime_error("Buffer must be contiguous");
  if (block_indices.size() != source_files.size())
    throw std::runtime_error("block_indices and source_files must have the same size");

  int64_t buffer_size = buffer.numel() * buffer.element_size();
  for (size_t i = 0; i < block_indices.size(); i++) {
    int64_t block_offset = block_indices[i] * block_size;
    if (block_offset + block_size > buffer_size)
      throw std::runtime_error("Block index " + std::to_string(block_indices[i]) +
                               " out of bounds for buffer size " +
                               std::to_string(buffer_size));
  }

  py::gil_scoped_release release;

  uint8_t* data_ptr = static_cast<uint8_t*>(buffer.data_ptr());
  SimpleThreadPool& pool = get_thread_pool();

  std::vector<std::future<bool>> futures;
  futures.reserve(block_indices.size());

  for (size_t i = 0; i < block_indices.size(); i++) {
    int64_t block_offset = block_indices[i] * block_size;
    std::string source_file = std::move(source_files[i]);

    futures.push_back(pool.enqueue([data_ptr, block_offset, block_size, source_file]() -> bool {
      return pread_file(source_file, data_ptr + block_offset, block_size);
    }));
  }

  bool all_success = true;
  for (auto& fut : futures) {
    if (!fut.get()) all_success = false;
  }
  if (!all_success) std::cerr << "[WARN] Some read operations failed\n";
  return all_success;
}

// cpp_write_blocks: opens all temp-file FDs sequentially in the main thread,
// then dispatches workers to do pure pwrite + close + rename.
// For writes, each new file requires O_CREAT which contends on the directory
// inode. Opening them one-by-one in the main thread before dispatch serializes
// that metadata work and avoids concurrent contention.
bool cpp_write_blocks(torch::Tensor buffer,
                      int64_t block_size,
                      std::vector<int64_t> block_indices,
                      std::vector<std::string> dest_files) {
  if (!buffer.is_cpu())
    throw std::runtime_error("Buffer must be on CPU");
  if (!buffer.is_contiguous())
    throw std::runtime_error("Buffer must be contiguous");
  if (block_indices.size() != dest_files.size())
    throw std::runtime_error("block_indices and dest_files must have the same size");

  int64_t buffer_size = buffer.numel() * buffer.element_size();
  for (size_t i = 0; i < block_indices.size(); i++) {
    int64_t block_offset = block_indices[i] * block_size;
    if (block_offset + block_size > buffer_size)
      throw std::runtime_error("Block index " + std::to_string(block_indices[i]) +
                               " out of bounds for buffer size " +
                               std::to_string(buffer_size));
  }

  py::gil_scoped_release release;

  uint8_t* buffer_ptr = static_cast<uint8_t*>(buffer.data_ptr());
  SimpleThreadPool& pool = get_thread_pool();
  size_t n = block_indices.size();

  // Open all temp file FDs sequentially before dispatching any task
  std::vector<int> fds(n, -1);
  std::vector<std::string> tmp_paths(n);

  for (size_t i = 0; i < n; i++) {
    fs::path parent_dir = fs::path(dest_files[i]).parent_path();
    if (!parent_dir.empty()) {
      std::error_code ec;
      fs::create_directories(parent_dir, ec);
      if (ec) {
        std::cerr << "[ERROR] Failed to create directories: " << ec.message() << "\n";
        for (size_t j = 0; j < i; j++) if (fds[j] >= 0) close(fds[j]);
        return false;
      }
    }
    tmp_paths[i] = dest_files[i] + ".tmp";
    fds[i] = open(tmp_paths[i].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fds[i] < 0) {
      std::cerr << "[ERROR] Failed to open tmp file: " << tmp_paths[i]
                << " - " << std::strerror(errno) << "\n";
      for (size_t j = 0; j < i; j++) if (fds[j] >= 0) close(fds[j]);
      return false;
    }
  }

  std::vector<std::future<bool>> futures;
  futures.reserve(n);

  for (size_t i = 0; i < n; i++) {
    int64_t block_offset = block_indices[i] * block_size;
    int fd = fds[i];
    std::string tmp_path  = std::move(tmp_paths[i]);
    std::string dest_file = std::move(dest_files[i]);

    futures.push_back(pool.enqueue([buffer_ptr, block_offset, block_size,
                                    fd, tmp_path, dest_file]() -> bool {
      size_t total_written = 0;
      while (total_written < static_cast<size_t>(block_size)) {
        ssize_t written = pwrite(fd,
                                 buffer_ptr + block_offset + total_written,
                                 block_size - total_written,
                                 static_cast<off_t>(total_written));
        if (written < 0) {
          if (errno == EINTR) continue;
          std::cerr << "[ERROR] pwrite failed: " << tmp_path
                    << " - " << std::strerror(errno) << "\n";
          close(fd);
          unlink(tmp_path.c_str());
          return false;
        }
        total_written += written;
      }
      if (close(fd) != 0) {
        std::cerr << "[ERROR] close failed: " << tmp_path
                  << " - " << std::strerror(errno) << "\n";
        unlink(tmp_path.c_str());
        return false;
      }
      if (std::rename(tmp_path.c_str(), dest_file.c_str()) != 0) {
        std::cerr << "[ERROR] rename failed: " << tmp_path << " -> "
                  << dest_file << " - " << std::strerror(errno) << "\n";
        unlink(tmp_path.c_str());
        return false;
      }
      return true;
    }));
  }

  bool all_success = true;
  for (auto& fut : futures) {
    if (!fut.get()) all_success = false;
  }
  if (!all_success) std::cerr << "[WARN] Some write operations failed\n";
  return all_success;
}

// PyBind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpp_read_blocks",
        &cpp_read_blocks,
        "Read multiple blocks from separate files into a tensor in parallel",
        py::arg("buffer"),
        py::arg("block_size"),
        py::arg("block_indices"),
        py::arg("source_files"));

  m.def("cpp_write_blocks",
        &cpp_write_blocks,
        "Write multiple blocks from a tensor to separate files in parallel",
        py::arg("buffer"),
        py::arg("block_size"),
        py::arg("block_indices"),
        py::arg("dest_files"));

  m.def("set_thread_count",
        &set_thread_count,
        "Set the number of threads for the I/O thread pool",
        py::arg("num_threads"));

  m.def("get_io_thread_count",
        &get_io_thread_count,
        "Get the current number of threads in the I/O thread pool");
}

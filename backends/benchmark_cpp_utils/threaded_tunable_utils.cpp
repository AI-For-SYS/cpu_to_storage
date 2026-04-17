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
#include <sched.h>
#include <sys/syscall.h>

namespace fs = std::filesystem;

#include "simple_thread_pool.hpp"

// ============================================================================
// I/O Configuration — tunable parameters grouped in a struct
// ============================================================================
struct IOConfig {
  // File descriptor flags
  bool     o_noatime          = false;
  bool     o_direct           = false;

  // posix_fadvise hints
  int      fadvise_read       = POSIX_FADV_NORMAL;
  int      fadvise_write      = POSIX_FADV_NORMAL;

  // I/O chunking: 0 = full block
  size_t   io_chunk_size      = 0;

  // Prefetching (reads only): 0 = disabled
  int      prefetch_depth     = 0;

  // Write-specific
  bool     fallocate_prealloc = false;
  int      sync_strategy      = 0;  // 0=none, 1=fdatasync, 2=sync_file_range

  // Thread/CPU affinity
  bool     cpu_affinity       = false;

  void reset() { *this = IOConfig{}; }
};

static std::mutex g_io_config_mutex;
static IOConfig   g_io_config;

// ============================================================================
// Thread pool — same pattern as baseline cpp_utils.cpp
// ============================================================================
static std::mutex g_pool_mutex;
static std::unique_ptr<SimpleThreadPool> g_thread_pool;
static size_t g_thread_count = 0;

// Track previous config to avoid unnecessary pool rebuilds
static size_t g_prev_thread_count  = 0;
static bool   g_prev_cpu_affinity  = false;

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

// ============================================================================
// configure_all / get_config — bulk setter/getter for Optuna
// ============================================================================
static int parse_fadvise(const std::string& hint) {
  if (hint == "sequential") return POSIX_FADV_SEQUENTIAL;
  if (hint == "random")     return POSIX_FADV_RANDOM;
  if (hint == "willneed")   return POSIX_FADV_WILLNEED;
  if (hint == "dontneed")   return POSIX_FADV_DONTNEED;
  if (hint == "noreuse")    return POSIX_FADV_NOREUSE;
  return POSIX_FADV_NORMAL;
}

static int parse_sync_strategy(const std::string& s) {
  if (s == "fdatasync")       return 1;
  if (s == "sync_file_range") return 2;
  return 0;
}

void configure_all(py::dict config) {
  // 1. Reset and apply I/O config
  {
    std::lock_guard<std::mutex> lock(g_io_config_mutex);
    g_io_config.reset();

    if (config.contains("o_noatime"))
      g_io_config.o_noatime = config["o_noatime"].cast<bool>();
    if (config.contains("o_direct"))
      g_io_config.o_direct = config["o_direct"].cast<bool>();
    if (config.contains("fadvise_hint")) {
      int hint = parse_fadvise(config["fadvise_hint"].cast<std::string>());
      g_io_config.fadvise_read = hint;
      g_io_config.fadvise_write = hint;
    }
    if (config.contains("fadvise_read"))
      g_io_config.fadvise_read = parse_fadvise(config["fadvise_read"].cast<std::string>());
    if (config.contains("fadvise_write"))
      g_io_config.fadvise_write = parse_fadvise(config["fadvise_write"].cast<std::string>());
    if (config.contains("io_chunk_kb"))
      g_io_config.io_chunk_size = config["io_chunk_kb"].cast<size_t>() * 1024;
    if (config.contains("io_chunk_size"))
      g_io_config.io_chunk_size = config["io_chunk_size"].cast<size_t>();
    if (config.contains("prefetch_depth"))
      g_io_config.prefetch_depth = config["prefetch_depth"].cast<int>();
    if (config.contains("fallocate"))
      g_io_config.fallocate_prealloc = config["fallocate"].cast<bool>();
    if (config.contains("sync_strategy")) {
      auto val = config["sync_strategy"];
      if (py::isinstance<py::int_>(val))
        g_io_config.sync_strategy = val.cast<int>();
      else
        g_io_config.sync_strategy = parse_sync_strategy(val.cast<std::string>());
    }
    if (config.contains("cpu_affinity"))
      g_io_config.cpu_affinity = config["cpu_affinity"].cast<bool>();
  }

  // 2. Rebuild thread pool only if thread_count or cpu_affinity changed
  {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    if (config.contains("thread_count"))
      g_thread_count = config["thread_count"].cast<size_t>();

    bool need_rebuild = (g_thread_count != g_prev_thread_count)
                     || (g_io_config.cpu_affinity != g_prev_cpu_affinity);
    if (need_rebuild && g_thread_count > 0) {
      g_thread_pool.reset();
      g_thread_pool = std::make_unique<SimpleThreadPool>(g_thread_count);
      g_prev_thread_count = g_thread_count;
      g_prev_cpu_affinity = g_io_config.cpu_affinity;
      std::cerr << "[INFO] Thread pool recreated with " << g_thread_count << " threads\n";
    }
  }
}

py::dict get_config() {
  py::dict d;
  d["thread_count"] = get_io_thread_count();
  {
    std::lock_guard<std::mutex> lock(g_io_config_mutex);
    d["o_noatime"]          = g_io_config.o_noatime;
    d["o_direct"]           = g_io_config.o_direct;
    d["fadvise_read"]       = g_io_config.fadvise_read;
    d["fadvise_write"]      = g_io_config.fadvise_write;
    d["io_chunk_size"]      = g_io_config.io_chunk_size;
    d["prefetch_depth"]     = g_io_config.prefetch_depth;
    d["fallocate_prealloc"] = g_io_config.fallocate_prealloc;
    d["sync_strategy"]      = g_io_config.sync_strategy;
    d["cpu_affinity"]       = g_io_config.cpu_affinity;
  }
  return d;
}

// ============================================================================
// Helper: pin current thread to a core (round-robin by task index)
// ============================================================================
static void maybe_pin_thread(bool cpu_affinity, size_t task_index) {
  if (!cpu_affinity) return;
  unsigned int num_cpus = std::thread::hardware_concurrency();
  if (num_cpus == 0) return;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(task_index % num_cpus, &cpuset);
  sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
}

// ============================================================================
// Read: pread with tunable parameters
// ============================================================================
static bool tunable_pread_file(const std::string& path, uint8_t* buffer_ptr,
                                size_t block_size, const IOConfig& cfg) {
  int flags = O_RDONLY;
  if (cfg.o_noatime) flags |= O_NOATIME;
  if (cfg.o_direct)  flags |= O_DIRECT;

  int fd = open(path.c_str(), flags);
  if (fd < 0) {
    // O_NOATIME may fail with EPERM if not file owner; retry without it
    if (cfg.o_noatime && errno == EPERM) {
      flags &= ~O_NOATIME;
      fd = open(path.c_str(), flags);
    }
    if (fd < 0) {
      std::cerr << "[ERROR] Failed to open file: " << path
                << " - " << std::strerror(errno) << "\n";
      return false;
    }
  }

  // fadvise hint
  if (cfg.fadvise_read != POSIX_FADV_NORMAL) {
    posix_fadvise(fd, 0, block_size, cfg.fadvise_read);
  }

  // Chunked pread loop
  size_t chunk = (cfg.io_chunk_size > 0) ? cfg.io_chunk_size : block_size;
  size_t total_read = 0;
  while (total_read < block_size) {
    size_t to_read = std::min(chunk, block_size - total_read);
    ssize_t n = pread(fd, buffer_ptr + total_read, to_read,
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

// ============================================================================
// threaded_tunable_read_blocks
// ============================================================================
bool threaded_tunable_read_blocks(torch::Tensor buffer,
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

  // Snapshot I/O config
  IOConfig cfg;
  {
    std::lock_guard<std::mutex> lock(g_io_config_mutex);
    cfg = g_io_config;
  }

  // O_DIRECT buffer alignment check
  uint8_t* data_ptr = static_cast<uint8_t*>(buffer.data_ptr());
  if (cfg.o_direct) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(data_ptr);
    if (addr % 4096 != 0)
      throw std::runtime_error("O_DIRECT requires 4096-byte aligned buffer");
  }

  py::gil_scoped_release release;

  SimpleThreadPool& pool = get_thread_pool();
  size_t n = block_indices.size();

  // Prefetch upcoming files via readahead()
  if (cfg.prefetch_depth > 0) {
    for (size_t i = 0; i < std::min(static_cast<size_t>(cfg.prefetch_depth), n); i++) {
      int pfd = open(source_files[i].c_str(), O_RDONLY);
      if (pfd >= 0) {
        readahead(pfd, 0, block_size);
        close(pfd);
      }
    }
  }

  std::vector<std::future<bool>> futures;
  futures.reserve(n);

  for (size_t i = 0; i < n; i++) {
    int64_t block_offset = block_indices[i] * block_size;
    std::string source_file = std::move(source_files[i]);

    futures.push_back(pool.enqueue([data_ptr, block_offset, block_size,
                                    source_file, cfg, i]() -> bool {
      maybe_pin_thread(cfg.cpu_affinity, i);
      return tunable_pread_file(source_file, data_ptr + block_offset, block_size, cfg);
    }));

    // Prefetch next batch of files as we go
    if (cfg.prefetch_depth > 0) {
      size_t prefetch_idx = i + cfg.prefetch_depth;
      if (prefetch_idx < n) {
        int pfd = open(source_files[prefetch_idx].c_str(), O_RDONLY);
        if (pfd >= 0) {
          readahead(pfd, 0, block_size);
          close(pfd);
        }
      }
    }
  }

  bool all_success = true;
  for (auto& fut : futures) {
    if (!fut.get()) all_success = false;
  }
  if (!all_success) std::cerr << "[WARN] Some read operations failed\n";
  return all_success;
}

// ============================================================================
// threaded_tunable_write_blocks
// ============================================================================
bool threaded_tunable_write_blocks(torch::Tensor buffer,
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

  // Snapshot I/O config
  IOConfig cfg;
  {
    std::lock_guard<std::mutex> lock(g_io_config_mutex);
    cfg = g_io_config;
  }

  // O_DIRECT buffer alignment check
  uint8_t* buffer_ptr = static_cast<uint8_t*>(buffer.data_ptr());
  if (cfg.o_direct) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer_ptr);
    if (addr % 4096 != 0)
      throw std::runtime_error("O_DIRECT requires 4096-byte aligned buffer");
  }

  py::gil_scoped_release release;

  SimpleThreadPool& pool = get_thread_pool();
  size_t n = block_indices.size();

  // Build open flags for writes
  int open_flags = O_WRONLY | O_CREAT | O_TRUNC;
  if (cfg.o_direct) open_flags |= O_DIRECT;

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
    fds[i] = open(tmp_paths[i].c_str(), open_flags, 0644);
    if (fds[i] < 0) {
      std::cerr << "[ERROR] Failed to open tmp file: " << tmp_paths[i]
                << " - " << std::strerror(errno) << "\n";
      for (size_t j = 0; j < i; j++) if (fds[j] >= 0) close(fds[j]);
      return false;
    }

    // Pre-allocate file space if enabled
    if (cfg.fallocate_prealloc) {
      if (fallocate(fds[i], 0, 0, block_size) != 0 && errno != EOPNOTSUPP) {
        std::cerr << "[WARN] fallocate failed: " << tmp_paths[i]
                  << " - " << std::strerror(errno) << " (continuing without)\n";
      }
    }

    // fadvise hint for writes
    if (cfg.fadvise_write != POSIX_FADV_NORMAL) {
      posix_fadvise(fds[i], 0, block_size, cfg.fadvise_write);
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
                                    fd, tmp_path, dest_file, cfg, i]() -> bool {
      maybe_pin_thread(cfg.cpu_affinity, i);

      // Chunked pwrite loop
      size_t chunk = (cfg.io_chunk_size > 0) ? cfg.io_chunk_size : static_cast<size_t>(block_size);
      size_t total_written = 0;
      while (total_written < static_cast<size_t>(block_size)) {
        size_t to_write = std::min(chunk, static_cast<size_t>(block_size) - total_written);
        ssize_t written = pwrite(fd,
                                 buffer_ptr + block_offset + total_written,
                                 to_write,
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

      // Sync strategy before close
      if (cfg.sync_strategy == 1) {
        if (fdatasync(fd) != 0 && errno != ENOSYS) {
          std::cerr << "[WARN] fdatasync failed: " << tmp_path
                    << " - " << std::strerror(errno) << " (continuing)\n";
        }
      } else if (cfg.sync_strategy == 2) {
        if (sync_file_range(fd, 0, block_size, SYNC_FILE_RANGE_WRITE) != 0) {
          // Not supported on NFS, S3 FUSE mounts — fall back silently
          if (errno != ENOSYS && errno != ESPIPE) {
            std::cerr << "[WARN] sync_file_range failed: " << tmp_path
                      << " - " << std::strerror(errno) << " (continuing)\n";
          }
        }
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

// ============================================================================
// Individual setters
// ============================================================================
void set_o_noatime(bool v)          { std::lock_guard<std::mutex> l(g_io_config_mutex); g_io_config.o_noatime = v; }
void set_o_direct(bool v)           { std::lock_guard<std::mutex> l(g_io_config_mutex); g_io_config.o_direct = v; }
void set_fadvise_read(int hint)     { std::lock_guard<std::mutex> l(g_io_config_mutex); g_io_config.fadvise_read = hint; }
void set_fadvise_write(int hint)    { std::lock_guard<std::mutex> l(g_io_config_mutex); g_io_config.fadvise_write = hint; }
void set_io_chunk_size(size_t sz)   { std::lock_guard<std::mutex> l(g_io_config_mutex); g_io_config.io_chunk_size = sz; }
void set_prefetch_depth(int d)      { std::lock_guard<std::mutex> l(g_io_config_mutex); g_io_config.prefetch_depth = d; }
void set_fallocate_prealloc(bool v) { std::lock_guard<std::mutex> l(g_io_config_mutex); g_io_config.fallocate_prealloc = v; }
void set_sync_strategy(int s)       { std::lock_guard<std::mutex> l(g_io_config_mutex); g_io_config.sync_strategy = s; }
void set_cpu_affinity(bool v)       { std::lock_guard<std::mutex> l(g_io_config_mutex); g_io_config.cpu_affinity = v; }

// ============================================================================
// PyBind11 bindings
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("threaded_tunable_read_blocks", &threaded_tunable_read_blocks,
        "Read multiple blocks from separate files into a tensor in parallel (tunable)",
        py::arg("buffer"), py::arg("block_size"),
        py::arg("block_indices"), py::arg("source_files"));

  m.def("threaded_tunable_write_blocks", &threaded_tunable_write_blocks,
        "Write multiple blocks from a tensor to separate files in parallel (tunable)",
        py::arg("buffer"), py::arg("block_size"),
        py::arg("block_indices"), py::arg("dest_files"));

  m.def("configure_all", &configure_all,
        "Configure all I/O parameters at once from a dict", py::arg("config"));
  m.def("get_config", &get_config,
        "Get current I/O configuration as a dict");

  m.def("set_thread_count", &set_thread_count,
        "Set the number of threads for the I/O thread pool", py::arg("num_threads"));
  m.def("get_io_thread_count", &get_io_thread_count,
        "Get the current number of threads in the I/O thread pool");

  m.def("set_o_noatime", &set_o_noatime, py::arg("enabled"));
  m.def("set_o_direct", &set_o_direct, py::arg("enabled"));
  m.def("set_fadvise_read", &set_fadvise_read, py::arg("hint"));
  m.def("set_fadvise_write", &set_fadvise_write, py::arg("hint"));
  m.def("set_io_chunk_size", &set_io_chunk_size, py::arg("bytes"));
  m.def("set_prefetch_depth", &set_prefetch_depth, py::arg("depth"));
  m.def("set_fallocate_prealloc", &set_fallocate_prealloc, py::arg("enabled"));
  m.def("set_sync_strategy", &set_sync_strategy, py::arg("strategy"));
  m.def("set_cpu_affinity", &set_cpu_affinity, py::arg("enabled"));
}

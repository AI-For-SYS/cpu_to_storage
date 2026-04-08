#include <torch/extension.h>
#include <liburing.h>

#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <cerrno>
#include <filesystem>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Global configuration (set from Python before each benchmark run)
// ---------------------------------------------------------------------------
static size_t g_queue_depth = 256;
static size_t g_batch_size  = 256;  // defaults to queue_depth

void set_queue_depth(size_t depth) {
  if (depth == 0)
    throw std::runtime_error("queue_depth must be greater than 0");
  g_queue_depth = depth;
  std::cerr << "[INFO] io_uring queue_depth set to " << depth << "\n";
}

size_t get_queue_depth() {
  return g_queue_depth;
}

void set_batch_size(size_t size) {
  if (size == 0)
    throw std::runtime_error("batch_size must be greater than 0");
  g_batch_size = size;
  std::cerr << "[INFO] io_uring batch_size set to " << size << "\n";
}

size_t get_batch_size() {
  return g_batch_size;
}

// ---------------------------------------------------------------------------
// Runtime probe — detects whether io_uring is available on this kernel.
// Throws std::runtime_error if blocked (e.g., kernel.io_uring_disabled != 0).
// ---------------------------------------------------------------------------
void iouring_probe() {
  struct io_uring ring;
  int ret = io_uring_queue_init(2, &ring, 0);
  if (ret < 0) {
    throw std::runtime_error(
        std::string("io_uring not available: ") + std::strerror(-ret));
  }
  io_uring_queue_exit(&ring);
}

// ---------------------------------------------------------------------------
// Validation (shared between read and write paths)
// ---------------------------------------------------------------------------
static void validate_buffer_args(torch::Tensor buffer,
                                 int64_t block_size,
                                 const std::vector<int64_t>& block_indices,
                                 const std::vector<std::string>& files) {
  if (!buffer.is_cpu())
    throw std::runtime_error("Buffer must be on CPU");
  if (!buffer.is_contiguous())
    throw std::runtime_error("Buffer must be contiguous");
  if (block_indices.size() != files.size())
    throw std::runtime_error("block_indices and files must have the same size");

  int64_t buffer_size = buffer.numel() * buffer.element_size();
  for (size_t i = 0; i < block_indices.size(); i++) {
    int64_t block_offset = block_indices[i] * block_size;
    if (block_offset + block_size > buffer_size)
      throw std::runtime_error("Block index " + std::to_string(block_indices[i]) +
                               " out of bounds for buffer size " +
                               std::to_string(buffer_size));
  }
}

// ---------------------------------------------------------------------------
// File open strategies
// ---------------------------------------------------------------------------

// Open files for reading. Files already exist so no O_CREAT contention —
// sequential open is simple and sufficient.
static std::vector<int> open_files_for_read(
    const std::vector<std::string>& source_files) {
  size_t n = source_files.size();
  std::vector<int> fds(n, -1);

  for (size_t i = 0; i < n; i++) {
    fds[i] = open(source_files[i].c_str(), O_RDONLY);
    if (fds[i] < 0) {
      std::cerr << "[ERROR] Failed to open file: " << source_files[i]
                << " - " << std::strerror(errno) << "\n";
      // Clean up already-opened FDs
      for (size_t j = 0; j < i; j++) {
        if (fds[j] >= 0) close(fds[j]);
      }
      return {};
    }
  }
  return fds;
}

// Open temp files for writing. Serialized O_CREAT to avoid directory inode
// contention — same strategy as cpp_ext. Creates parent directories if needed.
// Returns (fds, tmp_paths) pair. Empty vectors on failure.
static std::pair<std::vector<int>, std::vector<std::string>> open_files_for_write(
    const std::vector<std::string>& dest_files) {
  size_t n = dest_files.size();
  std::vector<int> fds(n, -1);
  std::vector<std::string> tmp_paths(n);

  for (size_t i = 0; i < n; i++) {
    fs::path parent_dir = fs::path(dest_files[i]).parent_path();
    if (!parent_dir.empty()) {
      std::error_code ec;
      fs::create_directories(parent_dir, ec);
      if (ec) {
        std::cerr << "[ERROR] Failed to create directories: "
                  << ec.message() << "\n";
        for (size_t j = 0; j < i; j++) if (fds[j] >= 0) close(fds[j]);
        return {{}, {}};
      }
    }
    tmp_paths[i] = dest_files[i] + ".tmp";
    fds[i] = open(tmp_paths[i].c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fds[i] < 0) {
      std::cerr << "[ERROR] Failed to open tmp file: " << tmp_paths[i]
                << " - " << std::strerror(errno) << "\n";
      for (size_t j = 0; j < i; j++) if (fds[j] >= 0) close(fds[j]);
      return {{}, {}};
    }
  }
  return {fds, tmp_paths};
}

// ---------------------------------------------------------------------------
// Close FDs helper (inline-worthy but used in multiple cleanup paths)
// ---------------------------------------------------------------------------
static void close_fds(std::vector<int>& fds) {
  for (int fd : fds) {
    if (fd >= 0) close(fd);
  }
}

// ---------------------------------------------------------------------------
// io_uring submission strategy
//
// Submits read or write SQEs in batches, reaping completions between batches
// to keep the ring flowing. Handles more ops than queue_depth by chunking.
// ---------------------------------------------------------------------------
static bool submit_io_ops(struct io_uring* ring,
                          const std::vector<int>& fds,
                          uint8_t* buffer_ptr,
                          const std::vector<int64_t>& block_indices,
                          int64_t block_size,
                          bool is_write) {
  size_t total_ops = fds.size();
  size_t submitted = 0;
  size_t completed = 0;
  size_t batch = std::min(g_batch_size, g_queue_depth);
  bool all_success = true;

  while (submitted < total_ops || completed < total_ops) {
    // Fill the SQ with as many ops as we can for this batch
    size_t batch_end = std::min(submitted + batch, total_ops);
    for (size_t i = submitted; i < batch_end; i++) {
      struct io_uring_sqe* sqe = io_uring_get_sqe(ring);
      if (!sqe) {
        // SQ is full — submit what we have and reap before continuing
        break;
      }

      int64_t offset = block_indices[i] * block_size;
      if (is_write) {
        io_uring_prep_write(sqe, fds[i], buffer_ptr + offset, block_size, 0);
      } else {
        io_uring_prep_read(sqe, fds[i], buffer_ptr + offset, block_size, 0);
      }
      io_uring_sqe_set_data64(sqe, i);
      submitted++;
    }

    // Submit the batch to the kernel
    int ret = io_uring_submit(ring);
    if (ret < 0) {
      std::cerr << "[ERROR] io_uring_submit failed: "
                << std::strerror(-ret) << "\n";
      return false;
    }

    // Reap available completions
    while (completed < submitted) {
      struct io_uring_cqe* cqe;
      // Block-wait for at least one if we've submitted everything we can
      // for now, otherwise just peek for available completions
      bool should_wait = (submitted >= total_ops) ||
                         (submitted - completed >= batch);
      if (should_wait) {
        ret = io_uring_wait_cqe(ring, &cqe);
      } else {
        ret = io_uring_peek_cqe(ring, &cqe);
      }

      if (ret < 0) {
        if (ret == -EAGAIN) break;  // no completions ready, submit more
        std::cerr << "[ERROR] io_uring CQE error: "
                  << std::strerror(-ret) << "\n";
        return false;
      }

      // Check for I/O errors
      if (cqe->res < 0) {
        uint64_t idx = io_uring_cqe_get_data64(cqe);
        std::cerr << "[ERROR] I/O op " << idx << " failed: "
                  << std::strerror(-cqe->res) << "\n";
        all_success = false;
      } else if (static_cast<size_t>(cqe->res) < static_cast<size_t>(block_size)) {
        // Short read/write — for simplicity we treat this as error for now.
        // A production version would resubmit the remainder.
        uint64_t idx = io_uring_cqe_get_data64(cqe);
        std::cerr << "[WARN] Short I/O on op " << idx << ": "
                  << cqe->res << "/" << block_size << " bytes\n";
        all_success = false;
      }

      io_uring_cqe_seen(ring, cqe);
      completed++;
    }
  }

  return all_success;
}

// ---------------------------------------------------------------------------
// Rename temp files to final destinations (atomic commit for writes)
// ---------------------------------------------------------------------------
static bool rename_temp_files(const std::vector<std::string>& tmp_paths,
                              const std::vector<std::string>& dest_files) {
  for (size_t i = 0; i < tmp_paths.size(); i++) {
    if (std::rename(tmp_paths[i].c_str(), dest_files[i].c_str()) != 0) {
      std::cerr << "[ERROR] rename failed: " << tmp_paths[i] << " -> "
                << dest_files[i] << " - " << std::strerror(errno) << "\n";
      // Clean up remaining temp files
      for (size_t j = i; j < tmp_paths.size(); j++) {
        unlink(tmp_paths[j].c_str());
      }
      return false;
    }
  }
  return true;
}

// ===========================================================================
// Orchestration: PyBind11-exposed functions
// ===========================================================================

bool iouring_read_blocks(torch::Tensor buffer,
                         int64_t block_size,
                         std::vector<int64_t> block_indices,
                         std::vector<std::string> source_files) {
  validate_buffer_args(buffer, block_size, block_indices, source_files);

  py::gil_scoped_release release;

  // Open all source files
  std::vector<int> fds = open_files_for_read(source_files);
  if (fds.empty()) return false;

  // Init io_uring ring
  struct io_uring ring;
  int ret = io_uring_queue_init(g_queue_depth, &ring, 0);
  if (ret < 0) {
    std::cerr << "[ERROR] io_uring_queue_init failed: "
              << std::strerror(-ret) << "\n";
    close_fds(fds);
    return false;
  }

  // Submit and reap read operations
  uint8_t* data_ptr = static_cast<uint8_t*>(buffer.data_ptr());
  bool success = submit_io_ops(&ring, fds, data_ptr, block_indices,
                               block_size, /*is_write=*/false);

  // Cleanup
  io_uring_queue_exit(&ring);
  close_fds(fds);

  if (!success) std::cerr << "[WARN] Some read operations failed\n";
  return success;
}

bool iouring_write_blocks(torch::Tensor buffer,
                          int64_t block_size,
                          std::vector<int64_t> block_indices,
                          std::vector<std::string> dest_files) {
  validate_buffer_args(buffer, block_size, block_indices, dest_files);

  py::gil_scoped_release release;

  // Open all temp files (serialized O_CREAT)
  auto [fds, tmp_paths] = open_files_for_write(dest_files);
  if (fds.empty()) return false;

  // Init io_uring ring
  struct io_uring ring;
  int ret = io_uring_queue_init(g_queue_depth, &ring, 0);
  if (ret < 0) {
    std::cerr << "[ERROR] io_uring_queue_init failed: "
              << std::strerror(-ret) << "\n";
    close_fds(fds);
    for (auto& p : tmp_paths) unlink(p.c_str());
    return false;
  }

  // Submit and reap write operations
  uint8_t* data_ptr = static_cast<uint8_t*>(buffer.data_ptr());
  bool success = submit_io_ops(&ring, fds, data_ptr, block_indices,
                               block_size, /*is_write=*/true);

  // Cleanup ring and close FDs before rename
  io_uring_queue_exit(&ring);
  close_fds(fds);

  if (!success) {
    std::cerr << "[WARN] Some write operations failed\n";
    for (auto& p : tmp_paths) unlink(p.c_str());
    return false;
  }

  // Atomic rename: temp files -> final destinations
  if (!rename_temp_files(tmp_paths, dest_files)) {
    return false;
  }

  return true;
}

// ===========================================================================
// PyBind11 bindings
// ===========================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("iouring_read_blocks",
        &iouring_read_blocks,
        "Read multiple blocks from separate files into a tensor via io_uring",
        py::arg("buffer"),
        py::arg("block_size"),
        py::arg("block_indices"),
        py::arg("source_files"));

  m.def("iouring_write_blocks",
        &iouring_write_blocks,
        "Write multiple blocks from a tensor to separate files via io_uring",
        py::arg("buffer"),
        py::arg("block_size"),
        py::arg("block_indices"),
        py::arg("dest_files"));

  m.def("iouring_probe",
        &iouring_probe,
        "Probe whether io_uring is available on this kernel");

  m.def("set_queue_depth",
        &set_queue_depth,
        "Set the io_uring submission/completion queue depth",
        py::arg("depth"));

  m.def("get_queue_depth",
        &get_queue_depth,
        "Get the current io_uring queue depth");

  m.def("set_batch_size",
        &set_batch_size,
        "Set how many SQEs to submit before reaping completions",
        py::arg("size"));

  m.def("get_batch_size",
        &get_batch_size,
        "Get the current batch size");
}

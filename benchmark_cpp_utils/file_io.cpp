#include "file_io.hpp"

#include <filesystem>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <random>
#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;

// Thread-local unique suffix for temporary files
thread_local std::string tmp_file_suffix =
    "_" + std::to_string(std::random_device{}()) + ".tmp";

// Write a buffer to disk using a temporary file and atomic rename
bool write_buffer_to_file(const uint8_t* buffer_ptr,
                          size_t size,
                          const std::string& target_path) {
  // Create parent directory if needed
  fs::path file_path(target_path);
  fs::path parent_dir = file_path.parent_path();
  
  if (!parent_dir.empty()) {
    try {
      fs::create_directories(parent_dir);
    } catch (const fs::filesystem_error& e) {
      std::cerr << "[ERROR] Failed to create directories: " << e.what() << "\n";
      return false;
    }
  }

  // Write to a temporary file to ensure atomic replace on rename
  std::string tmp_path = target_path + tmp_file_suffix;
  
  // Open file with POSIX API for direct writing (no intermediate buffering)
  int fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0) {
    std::cerr << "[ERROR] Failed to open temporary file for writing: "
              << tmp_path << " - " << std::strerror(errno) << "\n";
    return false;
  }

  // Write data directly from source buffer to file
  size_t total_written = 0;
  while (total_written < size) {
    ssize_t written = write(fd, buffer_ptr + total_written, size - total_written);
    if (written < 0) {
      if (errno == EINTR) {
        // Interrupted by signal, retry
        continue;
      }
      std::cerr << "[ERROR] Failed to write to temporary file: " << tmp_path
                << " - " << std::strerror(errno) << "\n";
      close(fd);
      unlink(tmp_path.c_str());
      return false;
    }
    total_written += written;
  }

  // Close the file
  if (close(fd) != 0) {
    std::cerr << "[ERROR] Failed to close temporary file: " << tmp_path
              << " - " << std::strerror(errno) << "\n";
    unlink(tmp_path.c_str());
    return false;
  }

  // Atomically rename temp file to final target name
  if (std::rename(tmp_path.c_str(), target_path.c_str()) != 0) {
    std::cerr << "[ERROR] Failed to rename " << tmp_path << " to "
              << target_path << " - " << std::strerror(errno) << "\n";
    unlink(tmp_path.c_str());
    return false;
  }

  return true;
}

// Read a file directly into a buffer
// Uses direct POSIX I/O for maximum performance (no intermediate buffering)
bool read_file_to_buffer(const std::string& path,
                         uint8_t* buffer_ptr,
                         size_t block_size) {
  // Open file with POSIX API
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cerr << "[ERROR] Failed to open file: " << path
              << " - " << std::strerror(errno) << "\n";
    return false;
  }

  // Read file directly into buffer
  size_t total_read = 0;
  while (total_read < block_size) {
    ssize_t bytes_read = read(fd, buffer_ptr + total_read,
                              block_size - total_read);
    if (bytes_read < 0) {
      if (errno == EINTR) {
        // Interrupted by signal, retry
        continue;
      }
      std::cerr << "[ERROR] Failed to read from file: " << path
                << " - " << std::strerror(errno) << "\n";
      close(fd);
      return false;
    }
    if (bytes_read == 0) {
      // Unexpected EOF
      std::cerr << "[ERROR] Unexpected EOF reading file: " << path
                << " (read " << total_read << "/" << block_size << " bytes)\n";
      close(fd);
      return false;
    }
    total_read += bytes_read;
  }

  // Close file
  if (close(fd) != 0) {
    std::cerr << "[ERROR] Failed to close file: " << path
              << " - " << std::strerror(errno) << "\n";
    return false;
  }

  return true;
}

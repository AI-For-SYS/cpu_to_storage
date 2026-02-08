#pragma once

#include <string>
#include <cstdint>

// Write a buffer to disk using a temporary file and atomic rename
// This ensures the file is either fully written or not present at all
bool write_buffer_to_file(const uint8_t* data, 
                          size_t size, 
                          const std::string& target_path);

// Read a file directly into a buffer
// Returns true on success, false on failure
bool read_file_to_buffer(const std::string& path, 
                         uint8_t* buffer, 
                         size_t buffer_size);

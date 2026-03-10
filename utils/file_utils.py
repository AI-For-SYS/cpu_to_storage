"""File utility functions for benchmark operations."""

import os
from utils.config import STORAGE_PATH
from typing import Any
import time
import asyncio

def generate_dest_file_names(name, num_files):
    return [f"{STORAGE_PATH}/{name}_{j}.bin" for j in range(num_files)]


def verify_file(original_data, filename):
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return False
    
    with open(filename, "rb") as f:
        file_data = f.read()
    
    # Convert memoryview to bytes
    original_bytes = bytes(original_data)
    
    # Check size first for better error messages
    if len(file_data) != len(original_bytes):
        print(f"Size mismatch in {filename}: file has {len(file_data)} bytes, expected {len(original_bytes)} bytes")
        return False
    
    return file_data == original_bytes


def verify_op(block_size, block_indices, view, dest_files, operation="operation", verify=False):
    if not verify:
        return
    
    for i, block_inx in enumerate(block_indices):
        start_byte = block_inx * block_size
        end_byte = (block_inx + 1) * block_size
        if not verify_file(view[start_byte:end_byte], dest_files[i]):
            print(f"{operation} block {block_inx} Failed")
            return
    
    print(f"Verified {operation} blocks")


def clean_files(file_names):
    for file_name in file_names:
        try:
            if os.path.exists(file_name):
                os.remove(file_name)
        except Exception as e:
            print(f"    Warning: Could not remove {file_name}: {e}")

def write_block_direct(fd, dest_name, buffer_view_slice):
    """Write cleaning files directly without temp file rename."""
    try:
        bytes_written = os.write(fd, buffer_view_slice)
        os.close(fd)
        return bytes_written == len(buffer_view_slice)
    except Exception as e:
        print(f"Error writing cleaning block: {e}")
        print(f"  Dest file: {dest_name}")
        try:
            os.close(fd)
        except:
            pass
        return False

async def write_blocks(block_size, view, block_indices, dest_files):
    tasks: list[Any] = []
    
    loop = asyncio.get_running_loop()
    start = time.perf_counter()

    # Open files directly in their destination location
    fds = [os.open(dest_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC) for dest_file in dest_files]

    tasks = [
        loop.run_in_executor(
            None,
            write_block_direct,
            fd, dest_file, view[start:start+block_size]
        )
        for fd, dest_file, start in zip(fds, dest_files, (i * block_size for i in block_indices))
    ]

    results = await asyncio.gather(*tasks)
    end = time.perf_counter()
    return (end - start)


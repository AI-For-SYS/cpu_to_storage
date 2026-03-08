"""Buffer allocation and management utilities for benchmarking."""

import torch
import os
import time
import asyncio
from typing import Any

def setup_cleaning_files():
    """Setup file names and indices for cache cleaning operations."""
    file_names_cleaning: list[str] = [f"/dev/shm/cleaning_{j}.bin" for j in range(3200)]
    indices_cleaning = list(range(3200))
    block_size_cleaning = 32*1024*1024
    
    return file_names_cleaning, indices_cleaning, block_size_cleaning

def write_cleaning_block_direct(fd, dest_name, buffer_view_slice):
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

async def write_cleaning_blocks(block_size, view, block_indices, dest_files):
    """Write cleaning files directly to /dev/shm without temp file rename."""
    tasks: list[Any] = []
    
    loop = asyncio.get_running_loop()
    start = time.perf_counter()

    # Open files directly in their destination location
    fds = [os.open(dest_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC) for dest_file in dest_files]

    tasks = [
        loop.run_in_executor(
            None,
            write_cleaning_block_direct,
            fd, dest_file, view[start:start+block_size]
        )
        for fd, dest_file, start in zip(fds, dest_files, (i * block_size for i in block_indices))
    ]

    results = await asyncio.gather(*tasks)
    end = time.perf_counter()
    return (end - start)
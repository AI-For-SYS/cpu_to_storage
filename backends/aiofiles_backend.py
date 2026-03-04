import aiofiles
import aiofiles.os 
from typing import Any
import time
import asyncio
import os

# Storage path configuration - can be overridden by STORAGE_PATH environment variable
STORAGE_PATH = os.environ.get('STORAGE_PATH', '/dev/shm')

async def write_and_rename(block_size, block_inx, buffer_view, temp_name, final_name):
    """Encapsulates the logic for a single buffer operation."""
    try:
        # Step A: Write to temporary file
        async with aiofiles.open(temp_name, "wb") as f:
            await f.write(buffer_view[block_inx*block_size: (block_inx+1)*block_size])

        # Step B: Atomically rename
        await aiofiles.os.replace(temp_name, final_name)
        return True
    except Exception as e:
        print(f"Error processing {final_name}: {e}")
        if await aiofiles.os.path.exists(temp_name):
            await aiofiles.os.unlink(temp_name)
        return False

async def read_block_from_file(block_size, block_inx, buffer_view, file_name):
    """Encapsulates the logic for reading a single block from file."""
    try:
        # Step A: Read from file
        async with aiofiles.open(file_name, "rb") as f:
            bytes_read = await f.readinto(buffer_view[block_inx*block_size : (block_inx+1)*block_size])
            if bytes_read != block_size:
                print(f"read {bytes_read} instead of {block_size}")
        return True
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return False

async def aiofiles_write_blocks(block_size, buffer_view, block_indices, dest_files):
    tasks: list[Any] = []

    start = time.perf_counter()
    for i,block_inx in enumerate(block_indices):
        tasks.append(write_and_rename(block_size, block_inx, buffer_view, f"{STORAGE_PATH}/temp_block_{block_inx}.bin", dest_files[i]))
    
    results = await asyncio.gather(*tasks)
    end: float = time.perf_counter()
    return (end -start)

async def aiofiles_read_blocks(block_size, buffer_view, block_indices, dest_files):
    tasks: list[Any] = []

    start = time.perf_counter()
    for i,block_inx in enumerate(block_indices):
        tasks.append(read_block_from_file(block_size, block_inx, buffer_view, dest_files[i]))
    
    results = await asyncio.gather(*tasks)
    end: float = time.perf_counter()
    for result in results:
        if not result:
            print(f"Reading blocks with aiofiles Failed")
            return
    return (end -start)
import os
import asyncio
import time
from typing import Any


# Storage path configuration - can be overridden by STORAGE_PATH environment variable
STORAGE_PATH = os.environ.get('STORAGE_PATH', '/dev/shm')

def read_block_direct(fd, buffer_view_slice):
    try:
        bytes_read = os.readv(fd,buffer_view_slice)
        return bytes_read == len(buffer_view_slice)
    except:
        raise
        return False

async def python_self_read_blocks(block_size, view, block_indices, dest_files):
    loop = asyncio.get_running_loop()

    start = time.perf_counter()

    # Pre-open all files
    fds: list[int] = [os.open(fn, os.O_RDONLY) for fn in dest_files]

    tasks = [
        loop.run_in_executor(
            None,  # Use default executor
            read_block_direct,
            fd, [view[start:start + block_size]]
        )
        for fd, start in zip(fds, (block_inx * block_size for block_inx in block_indices))
    ]

    results = await asyncio.gather(*tasks)
    # Close fds
    for fd in fds:
        os.close(fd)
    end: float = time.perf_counter()
    return (end-start)


def write_block(fd: int, tmp_path: str, dest_path: str,
                    buf: memoryview, offset: int, block_size: int) -> bool:
    total = 0
    while total < block_size:
        total += os.write(fd, buf[offset + total:offset + block_size])
    os.close(fd)
    os.replace(tmp_path, dest_path)
    return True


async def python_self_write_blocks(block_size, view, block_indices, dest_files):
    tasks: list[Any] = []

    loop = asyncio.get_running_loop()
    start = time.perf_counter()

    temp_names = [f"{STORAGE_PATH}/temp_block_{i}.bin" for i in block_indices]
    fds = [os.open(temp_name, os.O_CREAT | os.O_WRONLY | os.O_TRUNC) for temp_name in temp_names]

    tasks = [
        loop.run_in_executor(
            None,
            write_block,
            fd, temp_name, dest_file, view, start, block_size
        )
        for temp_name, fd, dest_file, start in zip(temp_names, fds, dest_files, (i * block_size for i in block_indices))
    ]

    results = await asyncio.gather(*tasks)
    end = time.perf_counter()
    return (end - start)

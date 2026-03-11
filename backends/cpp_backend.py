import time
CPP_AVAILABLE = False
try:
    import cpp_ext
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: C++ extension not available: {e}")
    print("Run 'python setup.py build_ext --inplace' to build it")

def set_thread_count_cpp(num_threads):
    cpp_ext.set_thread_count(num_threads)


async def cpp_write_blocks(block_size, buffer, block_indices, dest_files):
    """C++ implementation wrapper for writing blocks.

    Opens all temp-file FDs sequentially before dispatching workers,
    then does parallel pwrite + close + rename per block.
    """
    start = time.perf_counter()
    success = cpp_ext.cpp_write_blocks(
        buffer, block_size, block_indices, dest_files
    )
    end = time.perf_counter()

    if not success:
        print("Writing blocks with c++ Failed")
        return
    return (end-start)

async def cpp_read_blocks(block_size, buffer, block_indices, dest_files):
    """C++ implementation wrapper for reading blocks.

    Each worker opens its own FD and uses pread() for parallel reads.
    """
    start = time.perf_counter()
    success = cpp_ext.cpp_read_blocks(
        buffer, block_size, block_indices, dest_files
    )
    if not success:
        print("Reading blocks with c++ Failed")
        return
    end = time.perf_counter()
    return (end-start)

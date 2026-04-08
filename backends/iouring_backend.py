import time

IOURING_AVAILABLE = False
try:
    import iouring_ext
    iouring_ext.iouring_probe()
    IOURING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: io_uring extension not available: {e}")
    print("Run 'python setup_iouring.py build_ext --inplace' to build it")
except RuntimeError as e:
    print(f"Warning: io_uring blocked by kernel: {e}")


# Individual parameter setters for Optuna search space compatibility
def set_queue_depth(depth):
    iouring_ext.set_queue_depth(depth)

def set_batch_size(size):
    iouring_ext.set_batch_size(size)


async def iouring_write_blocks(block_size, buffer, block_indices, dest_files):
    """io_uring implementation wrapper for writing blocks.

    Opens all temp-file FDs sequentially (serialized O_CREAT),
    then batches pwrite ops through the io_uring submission queue.
    """
    start = time.perf_counter()
    success = iouring_ext.iouring_write_blocks(
        buffer, block_size, block_indices, dest_files
    )
    end = time.perf_counter()

    if not success:
        print("Writing blocks with io_uring failed")
        return
    return (end - start)


async def iouring_read_blocks(block_size, buffer, block_indices, dest_files):
    """io_uring implementation wrapper for reading blocks.

    Batches pread ops through the io_uring submission queue.
    """
    start = time.perf_counter()
    success = iouring_ext.iouring_read_blocks(
        buffer, block_size, block_indices, dest_files
    )
    if not success:
        print("Reading blocks with io_uring failed")
        return
    end = time.perf_counter()
    return (end - start)

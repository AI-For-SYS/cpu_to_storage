import asyncio
from sys import implementation
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiofiles.os  # Required for the atomic replace
import time
import os
import statistics
from numpy import block
import torch
import random
import json

from torch.library import impl
from checkpoints_utils import save_incremental_results, load_existing_results, check_config_match, get_completed_tests
from nixl_implementation import nixl_write_blocks, nixl_read_blocks


# Import C++ extension for high-performance file I/O
try:
    import benchmark_utils
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: C++ extension not available: {e}")
    print("Run 'python setup.py build_ext --inplace' in compare_file_operations/ to build it")
    CPP_AVAILABLE = False

VERIFY = False


def generate_dest_file_names(num_files):
    return [f"/dev/shm/final_{j}" for j in range(num_files)]

def verify_file(original_data, filename):
    """Verifies that the file on disk matches the original CPU buffer."""
    if not os.path.exists(filename):
        return False
    
    with open(filename, "rb") as f:
        file_data = f.read()
    
    return file_data == original_data

def verify_read(block_size, block_indices, view, dest_files):                    
    if VERIFY:
        for i,block_inx in enumerate(block_indices):
            if not verify_file(view[block_inx*block_size: (block_inx+1)*block_size], dest_files[i]):
                print(f"Reading block {block_inx} Failed")
                return

def verify_write(block_size, block_indices, view, dest_files):                    
    if VERIFY:
        for i,block_inx in enumerate(block_indices):
            if not verify_file(view[block_inx*block_size: (block_inx+1)*block_size], dest_files[i]):
                print(f"Writing block {block_inx} Failed")
                return


def verify_file_cpp(original_data, filename):
    """Verifies that the file on disk matches the original CPU buffer."""
    if not os.path.exists(filename):
        print(f"File {filename} does not exist")
        return False
    
    with open(filename, "rb") as f:
        file_data = f.read()
    
    # Convert tensor to bytes for comparison
    original_bytes = original_data.numpy().tobytes()
    
    if len(file_data) != len(original_bytes):
        print(f"Size mismatch: file has {len(file_data)} bytes, expected {len(original_bytes)} bytes")
        return False
    
    return file_data == original_bytes

def write_and_rename_sync(block_size, block_inx, buffer_view, temp_name, final_name):
    start = block_inx * block_size
    end = start + block_size
    with open(temp_name, "wb") as f:
        f.write(buffer_view[start:end])
    os.replace(temp_name, final_name)
    return True

def read_block_from_file_sync(block_size, block_inx, buffer_view, file_name):
    """Synchronously read a single block from file into buffer_view."""
    try:
        start = block_inx * block_size
        mv_slice = buffer_view[start : start + block_size]

        with open(file_name, "rb") as f:
            # read directly into the memoryview slice
            bytes_read = f.readinto(mv_slice)
            if bytes_read != block_size:
                print(f"read {bytes_read} instead of {block_size} from {file_name}")
                return False
        return True
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return False

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
        tasks.append(write_and_rename(block_size, block_inx, buffer_view, f"/dev/shm/temp_block_{block_inx}.bin", dest_files[i]))
    
    results = await asyncio.gather(*tasks)
    end: float = time.perf_counter()
    return (end -start)

def write_block_direct_no_rename(block_size, block_inx, buffer_view, final_name):
    """Ultra-optimized write - direct write without temp file.
    
    Key optimizations:
    - No temp file + rename overhead
    - Single open/write/close sequence
    - Minimal system calls
    """
    try:
        start = block_inx * block_size
        end = start + block_size
        
        # Open final file directly with O_CREAT | O_WRONLY | O_TRUNC
        fd = os.open(final_name, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        try:
            # Single write call
            bytes_written = os.write(fd, buffer_view[start:end])
            if bytes_written != block_size:
                return False
        finally:
            os.close(fd)
        return True
    except:
        return False

# V3: Optimized with temp file + rename pattern
def write_block_direct(block_size, block_inx, buffer_view, temp_name, final_name):
    """Optimized write with temp file + atomic rename.
    
    Key optimizations:
    - Temp file + atomic rename for safety
    - No fsync for better performance
    - Single open/write/close sequence
    - Minimal system calls
    """
    try:
        start = block_inx * block_size
        end = start + block_size
        
        # Write to temp file
        fd = os.open(temp_name, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
        try:
            bytes_written = os.write(fd, buffer_view[start:end])
            if bytes_written != block_size:
                return False
        finally:
            os.close(fd)
        
        # Atomically rename temp to final
        os.replace(temp_name, final_name)
        return True
    except:
        # Clean up temp file on error
        try:
            if os.path.exists(temp_name):
                os.remove(temp_name)
        except:
            pass
        return False

# def write_block_direct_or(block_size: int, block_inx: int, buffer_view: memoryview, final_name: str) -> bool:
#     start = block_inx * block_size
#     end = start + block_size

#     # Direct write with O_EXCL|O_CREAT for atomicity (fails if file exists)
#     fd = os.open(final_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
#     try:
#         os.write(fd, buffer_view[start:end])
#         #os.fsync(fd)  # Durability (optional, but recommended)
#         os.close(fd)
#         return True
#     except:
#         os.close(fd)
#         os.unlink(final_name)  # Cleanup on error
#         raise

async def python_write_blocks(block_size, buffer_view, block_indices, dest_files, num_threads=None):
    """V3: Optimized version with temp file + rename and custom thread pool."""
    loop = asyncio.get_running_loop()
    
    # Create optimized thread pool if thread count specified
    if num_threads:
        executor = ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix='io_write')
    else:
        executor = None
    
    start = time.perf_counter()
   
    tasks = [
        loop.run_in_executor(
            executor,
            write_block_direct,
            block_size, block_inx, buffer_view,
            f"/dev/shm/temp_block_{block_inx}.bin",
            dest_files[i],
        )
        for i, block_inx in enumerate(block_indices)
    ]

    results = await asyncio.gather(*tasks)
    end: float = time.perf_counter()
    
    # Cleanup executor if we created one
    if executor:
        executor.shutdown(wait=False)
    
    return (end -start)

async def python_read_blocks(block_size, buffer_view, block_indices, dest_files, num_threads=None):
    """V3: Ultra-optimized version with direct reads and custom thread pool."""
    loop = asyncio.get_running_loop()
    
    # Create optimized thread pool if thread count specified
    if num_threads:
        executor = ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix='io_read')
    else:
        executor = None
    
    start = time.perf_counter()

    tasks = [
        loop.run_in_executor(
            executor,
            read_block_from_file_sync,
            block_size, block_inx, buffer_view,
            dest_files[i],
        )
        for i, block_inx in enumerate(block_indices)
    ]
    
    results = await asyncio.gather(*tasks)
    end: float = time.perf_counter()
    
    # Cleanup executor if we created one
    if executor:
        executor.shutdown(wait=False)
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
    

# --------- or implementation ------------
def read_block_direct_or(fd, buffer_view_slice):
    try:
        bytes_read = os.readv(fd,buffer_view_slice)
        return bytes_read == len(buffer_view_slice)
    except:
        raise
        return False

async def python_read_blocks_or(block_size, buffer, block_indices, dest_files, num_threads=None):
    view = memoryview(buffer.numpy()).cast('B')
    loop = asyncio.get_running_loop()

    # Create optimized thread pool if thread count specified
    if num_threads:
        executor = ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix='io_read')
    else:
        executor = None

    start = time.perf_counter()

    # Pre-open all files
    fds: list[int] = [os.open(fn, os.O_RDONLY) for fn in dest_files]

    tasks = [
        loop.run_in_executor(
            executor,
            read_block_direct_or,
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

def write_block_direct_or(temp_name, dest_name, buffer_view_slice):
    try:
        fd=os.open(temp_name, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
        bytes_written = os.write(fd, buffer_view_slice)
        os.close(fd)
        os.replace(temp_name, dest_name)
        return bytes_written == len(buffer_view_slice)
    except:
        return False

async def python_write_blocks_or(block_size, buffer, block_indices, dest_files):
    tasks: list[Any] = []
    view = memoryview(buffer.numpy()).cast('B')
    
    loop = asyncio.get_running_loop()
    start = time.perf_counter()

    for file_name in dest_files:
        try:
            if os.path.exists(file_name):
                os.remove(file_name)
        except Exception as e:
            print(f"    Warning: Could not remove {file_name}: {e}")

    temp_names = [f"/dev/shm/temp_block_{i}.bin" for i in block_indices]

    tasks = [
        loop.run_in_executor(
            None,
            write_block_direct_or,
            temp_name, dest_file, view[start:start+block_size]
        )
        for temp_name, dest_file, start in zip(temp_names, dest_files, (i * block_size for i in block_indices))
    ]

    results = await asyncio.gather(*tasks)
    end = time.perf_counter()
    return (end - start)
# ---------------------------

# --------- gemini's version ---------
_os_open = os.open
_os_write = os.write
_os_close = os.close
_os_replace = os.replace
_os_remove = os.remove
# Combine flags once
_FLAGS = os.O_CREAT | os.O_WRONLY | os.O_TRUNC

def write_block_fastest(block_size, block_inx, buffer_view, temp_name, final_name):
    """
    Highly optimized write using raw syscalls and pre-bound globals.
    Assumes buffer_view is a memoryview/mmap for zero-copy slicing.
    """
    try:
        start = block_inx * block_size
        chunk = buffer_view[start : start+ block_size]
        # Open file descriptor (low-level)
        fd = _os_open(temp_name, _FLAGS, 0o644)
        
        try:
            # os.write accepts the memoryview slice directly (Zero-Copy)
            # We assume a successful write of the full block for speed.
            # On local NVMe/SSD, partial writes are extremely rare unless disk is full.
            _os_write(fd, chunk)
        finally:
            _os_close(fd)
        
        # Atomic rename
        _os_replace(temp_name, final_name)
        return True

    except Exception:
        # Minimal cleanup logic, only triggered on failure
        try:
            _os_remove(temp_name)
        except OSError:
            pass
        return False

# ------------------------------------------

async def cpp_write_blocks(block_size, buffer, block_indices, dest_files):
    """C++ implementation wrapper for writing blocks.
    
    The C++ function handles threading internally and releases the GIL,
    so we call it directly. It returns execution time in seconds.
    """
    start = time.perf_counter()
    
    # C++ function releases GIL and uses its own thread pool
    success = benchmark_utils.cpp_write_blocks(
        buffer, block_size, block_indices, dest_files
    )
    if not success:
        print("Writing blocks with c++ Failed")
        return

    end = time.perf_counter()
    if VERIFY:
        bytes_per_element = buffer.element_size()  # 2 for float16
        elements_per_block = block_size // bytes_per_element

        for i, block_inx in enumerate(block_indices):
            start_elem = block_inx * elements_per_block
            end_elem = (block_inx + 1) * elements_per_block
            if not verify_file_cpp(buffer[start_elem:end_elem], dest_files[i]):
                print(f"writing block {block_inx} with c++ Failed")
    return (end-start)

async def cpp_read_blocks(block_size, buffer, block_indices, dest_files):
    """C++ implementation wrapper for reading blocks.
    
    The C++ function handles threading internally and releases the GIL,
    so we call it directly. It returns execution time in seconds.
    """
    start = time.perf_counter()
    # C++ function releases GIL and uses its own thread pool
    success = benchmark_utils.cpp_read_blocks(
        buffer, block_size, block_indices, dest_files
    )
    if not success:
        print("Reading blocks with c++ Failed")
        return
    end = time.perf_counter()

    if VERIFY:
        bytes_per_element = buffer.element_size()  # 2 for float16
        elements_per_block = block_size // bytes_per_element

        for i, block_inx in enumerate(block_indices):
            start_elem = block_inx * elements_per_block
            end_elem = (block_inx + 1) * elements_per_block
            if not verify_file_cpp(buffer[start_elem:end_elem], dest_files[i]):
                print(f"writing block {block_inx} with c++ Failed")
                exit()
    return (end-start)

async def aiofiles_threads(iterations, num_blocks_to_copy, buffer_size):
    start_time = time.perf_counter()
    block_size = 2 *1024 * 1024
    # Store results in a more structured format
    if block_size*num_blocks_to_copy > buffer_size:
        print("Buffer size is too small")
        return
    write_results = {}
    read_results = {}

    num_elements = buffer_size // 2
    print("Allocating buffer... ")
    buffer = torch.randn(num_elements, dtype=torch.float16, device='cpu').pin_memory()
    print("Buffer allocated")
    file_names = generate_dest_file_names(buffer_size)
    threads_nums = [8,16,32,64,128]
    
    for num_of_threads in threads_nums:
        print(f" Running with {num_of_threads} threads")
        # Set up thread pool executor for this iteration
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=num_of_threads)
        loop.set_default_executor(executor)
        
        times_write = []
        times_read = []
        for i in range(iterations):
            blocks_indices: list[int] = random.sample(range(buffer_size // block_size), num_blocks_to_copy)

            time_write = await aiofiles_write_blocks(block_size, buffer,blocks_indices, file_names)
            times_write.append(time_write)
            time_read = await aiofiles_read_blocks(block_size, buffer,blocks_indices, file_names)
            times_read.append(time_read)
        
        # Shutdown executor after each thread count test
        executor.shutdown(wait=True)
        
        # Store results by thread count
        if len(times_write)>0:
            write_results[num_of_threads] = {
                'avg': statistics.mean(times_write),
                'median': statistics.median(times_write),
                'min': min(times_write),
                'max': max(times_write),
                'all_times': times_write
            }
        if len(times_read)>0:
            read_results[num_of_threads] = {
                'avg': statistics.mean(times_read),
                'median': statistics.median(times_read),
                'min': min(times_read),
                'max': max(times_read),
                'all_times': times_read
            }
    
    # Combine results for saving
    all_results = {
        'write': write_results,
        'read': read_results,
        'config': {
            'buffer_size': buffer,
            'num_iterations': iterations,
            'num_blocks_to_copy': num_blocks_to_copy,
            'block_size': block_size,
            'file_system': 'tmpfs (/dev/shm)',
            'implementation': 'Python aiofiles'
        }
    }
    
    # Save results to JSON file
    output_file = f'compare_file_operations_results/threads_comp_bs{block_size}_6.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Print results table
    print("\n" + "="*80)
    print("                 BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Method':<40} | {'Avg (ms)':<10} | {'Median (ms)':<10} | {'Min (ms)':<10} | {'Max (ms)':<10}")
    print("-" * 80)
    
    for num_threads in threads_nums:
        print(f"Write ({num_threads} threads){'':<26} | {write_results[num_threads]['avg']*1000:>10.2f} | {write_results[num_threads]['median']*1000:>10.2f} | {write_results[num_threads]['min']*1000:>10.2f} | {write_results[num_threads]['max']*1000:>10.2f}")
        print(f"Read ({num_threads} threads){'':<27} | {read_results[num_threads]['avg']*1000:>10.2f} | {read_results[num_threads]['median']*1000:>10.2f} | {read_results[num_threads]['min']*1000:>10.2f} | {read_results[num_threads]['max']*1000:>10.2f}")
        print("-" * 80)
    print(f"Script running time is {time.perf_counter()-start_time}s")
    return all_results

async def block_size_comparison_total(total_gb, iterations, buffer_size):
    """Benchmark different block sizes across multiple thread counts.
    
    Args:
        total_mb: Total data size in MB to transfer for each test (default: 2048 MB = 2 GB)
        iterations: Number of iterations per test (default: 5)
    """
    start_time = time.perf_counter()
    
    # Test parameters
    thread_counts = [16, 32, 64]
    block_sizes_mb = [2, 4, 8, 16, 32, 64, 128, 256]
    block_sizes_mb.reverse()

    # Calculate total combinations
    total_combinations = len(block_sizes_mb) * len(thread_counts)
    print(f"Testing {total_combinations} combinations ({len(block_sizes_mb)} block sizes × {len(thread_counts)} threads)")
    print(f"Fixed total data size per test: {total_gb} GB)")
    print(f"Iterations per test: {iterations}\n")
    
    # Prepare output file path
    output_file = f'results/block_size_comparison_{total_gb}gb_zeros.json'
    
    # Check for existing results
    existing_results = load_existing_results(output_file)
    
    # Store results - structure: results[block_size][thread_count] = time
    write_results = {}
    read_results = {}
    
    # Configuration for this benchmark
    config = {
        'buffer_size': buffer_size,
        'num_iterations': iterations,
        'total_data_size_gb': total_gb,
        'thread_counts': thread_counts,
        'block_sizes_mb': block_sizes_mb,
        'file_system': 'tmpfs (/dev/shm)',
        'implementation': 'Python aiofiles'
    }
    
    # Check if we can resume from existing results
    completed_write = set()
    completed_read = set()
    
    if existing_results and 'config' in existing_results:
        if check_config_match(existing_results['config'], config):
            print(f"Found existing results with matching configuration!")
            print(f"Resuming benchmark from previous state...\n")
            write_results = existing_results.get('write', {})
            read_results = existing_results.get('read', {})
            completed_write = get_completed_tests(existing_results, 'write')
            completed_read = get_completed_tests(existing_results, 'read')
            print(f"Already completed: {len(completed_write)} write tests, {len(completed_read)} read tests")
        else:
            print(f"Existing results found but configuration doesn't match.")
            print(f"Starting fresh benchmark...\n")
    else:
        print(f"No existing results found. Starting fresh benchmark...\n")
    
    # Allocate buffer once
    num_elements = buffer_size // 2
    print("Allocating buffer...")
    buffer = torch.zeros(num_elements, dtype=torch.float16, device='cpu', pin_memory=True)
    print("Buffer allocated\n")
    
    combination_count = 0
    
    # OUTER LOOP: Block sizes
    for block_size_mb in block_sizes_mb:
        
        # Calculate number of blocks to achieve target total_mb
        num_blocks_to_copy = int((total_gb * 1024) / block_size_mb)
        
        print(f"\n{'='*80}")
        print(f"Block Size: {block_size_mb:.0f}MB | Blocks to copy: {num_blocks_to_copy} | Total: {total_gb}GB")
        print(f"{'='*80}")
        block_size = block_size_mb * 1024 * 1024
        
        if str(block_size) not in write_results:
            write_results[str(block_size)] = {}
        if str(block_size) not in read_results:
            read_results[str(block_size)] = {}
        
        # INNER LOOP: Thread counts
        for num_threads in thread_counts:
            combination_count += 1
            
            # Check if this combination is already completed
            test_key = (str(block_size), str(num_threads))
            if test_key in completed_write and test_key in completed_read:
                print(f"\n  [{combination_count}/{total_combinations}] Threads: {num_threads} - SKIPPED (already completed)")
                continue
            
            print(f"\n  [{combination_count}/{total_combinations}] Threads: {num_threads}")
            
            # Set up thread pool executor
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=num_threads)
            loop.set_default_executor(executor)
            
            # Generate file names
            file_names = generate_dest_file_names(num_blocks_to_copy)
            
            times_write = []
            times_read = []
            
            for i in range(iterations):
                blocks_indices = random.sample(range(buffer_size // block_size), num_blocks_to_copy)
                
                # Write benchmark
                time_write = await aiofiles_write_blocks(block_size, buffer, blocks_indices, file_names)
                times_write.append(time_write)
                
                # Read benchmark
                time_read = await aiofiles_read_blocks(block_size, buffer, blocks_indices, file_names)
                times_read.append(time_read)
            
            # Cleanup files after this test to free up space
            for file_name in file_names:
                try:
                    if os.path.exists(file_name):
                        os.remove(file_name)
                except Exception as e:
                    print(f"    Warning: Could not remove {file_name}: {e}")
            
            # Shutdown executor
            executor.shutdown(wait=True)
            
            # Calculate average times
            avg_write = statistics.mean(times_write)
            avg_read = statistics.mean(times_read)
            
            # Store results
            write_results[str(block_size)][str(num_threads)] = avg_write
            read_results[str(block_size)][str(num_threads)] = avg_read
            
            # Calculate and print throughput
            write_throughput = total_gb / avg_write
            read_throughput = total_gb / avg_read
            
            print(f"    Write: {avg_write*1000:.2f}ms ({write_throughput:.2f} GB/s)")
            print(f"    Read:  {avg_read*1000:.2f}ms ({read_throughput:.2f} GB/s)")
            
            # Save results incrementally after each test
            current_results = {
                'write': write_results,
                'read': read_results,
                'config': config
            }
            save_incremental_results(output_file, current_results)
            print(f"    Results saved to {output_file}")
    
    # Final save with complete results
    all_results = {
        'write': write_results,
        'read': read_results,
        'config': config
    }
    
    save_incremental_results(output_file, all_results)
    
    total_time = time.perf_counter() - start_time
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Total time: {total_time:.2f}s")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")
    
    return all_results

async def block_size_comparison(num_blocks, iterations, buffer_size, implementation='Python aiofiles'):
    """Benchmark different block sizes across multiple thread counts.
    
    Args:
        num_blocks: number of blocks to transfer in each test
        iterations: Number of iterations per test (default: 5)
    """
    start_time = time.perf_counter()
    if implementation=="C++":
        if CPP_AVAILABLE:
            print("Testing C++ implementation...")
        else:
            print("C++ implementation not available.")
            return
    elif implementation=="Python aiofiles":
        print("Testing Python aiofiles implementation...")
    elif implementation=="python_self_implementation":
        print("Testing Python expiremental implementation...")
    elif implementation=="nixl":
        print("Testing nixl implementation...")
    elif implementation=="python_or":
        print("Testing or's implementation...")
    else:
        print("Invalid implementation specified.")
        return

    # Test parameters
    thread_counts = [16,32,64]
    # block_sizes_mb: list[int] = [2,4]
    block_sizes_mb: list[int] = list(range(4,68,4))
    block_sizes_mb.insert(0,2)
    # Calculate total combinations
    total_combinations = len(block_sizes_mb) * len(thread_counts)
    print(f"Testing {total_combinations} combinations ({len(block_sizes_mb)} block sizes × {len(thread_counts)} threads)")
    print(f"Iterations per test: {iterations}\n")
    
    # Prepare output file path
    output_file = f'results/bsc_12cpu_pok_{implementation}.json'
    
    # Check for existing results
    existing_results: dict[Any, Any] | None = load_existing_results(output_file)
    
    # Store results - structure: results[thread_count][block_size] = time
    write_results = {}
    read_results = {}
    
    # Configuration for this benchmark
    config = {
        'cluster': 'pokprod',
        'buffer_size': buffer_size,
        'num_iterations': iterations,
        'num_blocks': num_blocks,
        'thread_counts': thread_counts,
        'block_sizes_mb': block_sizes_mb,
        'file_system': 'tmpfs (/dev/shm)',
        'implementation': implementation
    }
    
    # Check if we can resume from existing results
    completed_write = set()
    completed_read = set()
    
    if existing_results and 'config' in existing_results:
        if check_config_match(existing_results['config'], config):
            print(f"Found existing results with matching configuration!")
            print(f"Resuming benchmark from previous state...\n")
            write_results = existing_results.get('write', {})
            read_results = existing_results.get('read', {})
            completed_write = get_completed_tests(existing_results, 'write')
            completed_read = get_completed_tests(existing_results, 'read')
            print(f"Already completed: {len(completed_write)} write tests, {len(completed_read)} read tests")
        else:
            print(f"Existing results found but configuration doesn't match.")
            print(f"Starting fresh benchmark...\n")
    else:
        print(f"No existing results found. Starting fresh benchmark...\n")
    
    # Allocate buffer once
    num_elements = buffer_size // 2
    print("Allocating buffer...")
    buffer = torch.zeros(num_elements, dtype=torch.float16, device='cpu', pin_memory=True)
    print("Buffer allocated\n")
    
    # Generate file names
    file_names = generate_dest_file_names(num_blocks)

    combination_count = 0
    
    for num_threads in thread_counts:

        if str(num_threads) not in write_results:
            write_results[str(num_threads)] = {}
        if str(num_threads) not in read_results:
            read_results[str(num_threads)] = {}

        # Set up thread pool executor
        if implementation=="C++":
            benchmark_utils.set_io_thread_count(num_threads)
            
        else:
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=num_threads)
            loop.set_default_executor(executor)

        for block_size_mb in block_sizes_mb:
            block_size = block_size_mb * 1024 * 1024
            combination_count += 1
            
            # Check if this combination is already completed
            test_key = (str(num_threads), str(block_size_mb))
            if test_key in completed_write and test_key in completed_read:
                print(f"\n  [{combination_count}/{total_combinations}] block size: {block_size_mb}MB, threads: {num_threads} - SKIPPED (already completed)")
                continue
            
            print(f"\n  [{combination_count}/{total_combinations}] block size: {block_size_mb}MB, threads: {num_threads}")
            
            times_write = []
            times_read = []
            
            for i in range(iterations):
                blocks_indices_write = random.sample(range(buffer_size // block_size), num_blocks)
                blocks_indices_read = random.sample(range(buffer_size // block_size), num_blocks)
                view = memoryview(buffer.numpy()).cast('B')

                if implementation=="C++":
                    time_write = await cpp_write_blocks(block_size, buffer, blocks_indices_write, file_names)
                    time_read = await cpp_read_blocks(block_size, buffer, blocks_indices_read, file_names)
                elif implementation=="Python aiofiles":
                    time_write = await aiofiles_write_blocks(block_size, view, blocks_indices_write, file_names)
                    verify_write(block_size, blocks_indices_write, view, file_names)

                    time_read = await aiofiles_read_blocks(block_size, view, blocks_indices_read, file_names)
                    verify_read(block_size, view, blocks_indices_read, dest_files=file_names)

                elif implementation=="python_self_implementation":
                    time_write = await python_write_blocks(block_size, view, blocks_indices_write, file_names, num_threads)
                    verify_write(block_size, blocks_indices_write, view, file_names)
                    time_read = await python_read_blocks(block_size, view, blocks_indices_read, file_names, num_threads)
                    verify_read(block_size, view, blocks_indices_read, dest_files=file_names)

                elif implementation=="nixl":
                    time_write = await nixl_write_blocks(block_size, view, blocks_indices_write, file_names, num_threads)
                    verify_write(block_size, blocks_indices_write, view, file_names)
                    time_read = await nixl_read_blocks(block_size, view, blocks_indices_read, file_names, num_threads)
                    verify_read(block_size, view, blocks_indices_read, dest_files=file_names)                
                
                elif implementation=="python_or":
                    time_write = await python_write_blocks_or(block_size, buffer, blocks_indices_write, file_names)
                    verify_write(block_size, blocks_indices_write, view, file_names)
                    time_read = await python_read_blocks_or(block_size, buffer, blocks_indices_read, file_names, num_threads)
                    verify_read(block_size, view, blocks_indices_read, dest_files=file_names)
                else:
                    print("invalid implementation")
                    return
                times_write.append(time_write)
                times_read.append(time_read)
            
                # Cleanup files after this test to free up space
                for file_name in file_names:
                    try:
                        if os.path.exists(file_name):
                            os.remove(file_name)
                    except Exception as e:
                        print(f"    Warning: Could not remove {file_name}: {e}")
            
            # Calculate average times
            avg_write = statistics.mean(times_write)
            avg_read = statistics.mean(times_read)
            
            # Store results
            write_results[str(num_threads)][str(block_size_mb)] = avg_write
            read_results[str(num_threads)][str(block_size_mb)] = avg_read
            
            # Calculate and print throughput
            write_throughput = block_size_mb*num_blocks / (avg_write*1024)
            read_throughput = block_size_mb*num_blocks / (avg_read*1024)
            
            print(f"    Write: {avg_write*1000:.2f}ms ({write_throughput:.2f} GB/s)")
            print(f"    Read:  {avg_read*1000:.2f}ms ({read_throughput:.2f} GB/s)")
            
            # Save results incrementally after each test
            current_results = {
                'write': write_results,
                'read': read_results,
                'config': config
            }
            save_incremental_results(output_file, current_results)
            print(f"    Results saved to {output_file}")
        
        # Shutdown executor
        if implementation=="Python aiofiles":
            executor.shutdown(wait=True)
            
    # Final save with complete results
    all_results = {
        'write': write_results,
        'read': read_results,
        'config': config
    }
    
    save_incremental_results(output_file, all_results)
    
    total_time = time.perf_counter() - start_time
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Total time: {total_time:.2f}s")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    # Run benchmark
    os.system("rm -f /dev/shm/final_* /dev/shm/temp_block_*")
    # asyncio.run(aiofiles_threads(4, 300, buffer_size=10*1024*1024*1024))
    
    # Run block size comparison with 10GB total data size
    # asyncio.run(aiofiles_block_size_comparison_total(total_gb=10, iterations=3, buffer_size=15*1024*1024*1024))

    asyncio.run(block_size_comparison(num_blocks=1000, iterations=5, buffer_size=70*1024*1024*1024, implementation="python_or"))

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

# Import C++ extension for high-performance file I/O
try:
    import benchmark_utils
    CPP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: C++ extension not available: {e}")
    print("Run 'python setup.py build_ext --inplace' in compare_file_operations/ to build it")
    CPP_AVAILABLE = False

VERIFY = True

def generate_dest_file_names(num_files):
    return [f"/dev/shm/final_{j}" for j in range(num_files)]

def verify_file(original_data, filename):
    """Verifies that the file on disk matches the original CPU buffer."""
    if not os.path.exists(filename):
        return False
    
    with open(filename, "rb") as f:
        file_data = f.read()
    
    return file_data == original_data

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
            # buffer_view[block_inx*block_size : (block_inx+1)*block_size] = await f.read(block_size)
            bytes_read = await f.readinto(buffer_view[block_inx*block_size : (block_inx+1)*block_size])
            if bytes_read != block_size:
                print(f"read {bytes_read} instead of {block_size}")
        return True
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return False

async def aiofiles_write_blocks(block_size, buffer, block_indices, dest_files):
    tasks: list[Any] = []
    view = memoryview(buffer.numpy()).cast('B')

    start = time.perf_counter()
    for i,block_inx in enumerate(block_indices):
        tasks.append(write_and_rename(block_size, block_inx, view, f"/dev/shm/temp_block_{block_inx}.bin", dest_files[i]))
    
    results = await asyncio.gather(*tasks)
    end: float = time.perf_counter()
    if VERIFY:
        for i,(result, block_inx) in enumerate(zip(results,block_indices)):
            if not result or not verify_file(view[block_inx*block_size: (block_inx+1)*block_size], dest_files[i]):
                print(f"Writing block {block_inx} with aiofiles Failed")
    return (end -start)


async def aiofiles_read_blocks(block_size, buffer, block_indices, dest_files):
    tasks: list[Any] = []
    view = memoryview(buffer.numpy()).cast('B')

    start = time.perf_counter()
    for i,block_inx in enumerate(block_indices):
        tasks.append(read_block_from_file(block_size, block_inx, view, dest_files[i]))
    
    results = await asyncio.gather(*tasks)
    end: float = time.perf_counter()
    for result in results:
        if not result:
            print(f"Reading blocks with aiofiles Failed")
            return
    if VERIFY:
        for i,block_inx in enumerate(block_indices):
            if not verify_file(view[block_inx*block_size: (block_inx+1)*block_size], dest_files[i]):
                print(f"Reading block {block_inx} with aiofiles Failed")
                exit()
    return (end -start)

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
    
    # Store results - structure: results[block_size][thread_count] = time
    write_results = {}
    read_results = {}
    
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
        write_results[str(block_size)] = {}
        read_results[str(block_size)] = {}
        
        # INNER LOOP: Thread counts
        for num_threads in thread_counts:
            combination_count += 1
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
    
    # Combine results
    all_results = {
        'write': write_results,
        'read': read_results,
        'config': {
            'buffer_size': buffer_size,
            'num_iterations': iterations,
            'total_data_size_gb': total_gb,
            'thread_counts': thread_counts,
            'block_sizes_mb': block_sizes_mb,
            'file_system': 'tmpfs (/dev/shm)',
            'implementation': 'Python aiofiles'
        }
    }
    
    # Save results to JSON
    output_file = f'compare_file_operations_results/block_size_comparison_{total_gb}gb_zeros.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    total_time = time.perf_counter() - start_time
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Total time: {total_time:.2f}s")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")
    
    return all_results

async def block_size_comparison(num_blocks, iterations, buffer_size, cpp=False):
    """Benchmark different block sizes across multiple thread counts.
    
    Args:
        num_blocks: number of blocks to transfer in each test 
        iterations: Number of iterations per test (default: 5)
    """
    start_time = time.perf_counter()
    if cpp:
        implementation = 'C++'
        if CPP_AVAILABLE:
            print("Testing C++ implementation...")
        else:
            print("C++ implementation not available.")
            return
    else:
        print("Testing Python aiofiles implementation...")
        implementation = 'Python aiofiles'

    # Test parameters
    thread_counts = [16, 32, 64]
    block_sizes_mb: list[int] = list(range(4,68,4))

    # Calculate total combinations
    total_combinations = len(block_sizes_mb) * len(thread_counts)
    print(f"Testing {total_combinations} combinations ({len(block_sizes_mb)} block sizes × {len(thread_counts)} threads)")
    print(f"Iterations per test: {iterations}\n")
    
    # Store results - structure: results[thread_count][block_size] = time
    write_results = {}
    read_results = {}
    
    # Allocate buffer once
    num_elements = buffer_size // 2
    print("Allocating buffer...")
    buffer = torch.zeros(num_elements, dtype=torch.float16, device='cpu', pin_memory=True)
    print("Buffer allocated\n")
    
    combination_count = 0
    
    # OUTER LOOP: Block sizes
    for num_threads in thread_counts:

        write_results[str(num_threads)] = {}
        read_results[str(num_threads)] = {}

        # Set up thread pool executor
        if cpp:
            benchmark_utils.set_io_thread_count(num_threads)
        else:
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=num_threads)
            loop.set_default_executor(executor)

        for block_size_mb in block_sizes_mb:
            block_size = block_size_mb * 1024 * 1024
            combination_count += 1
            print(f"\n  [{combination_count}/{total_combinations}] block size: {block_size_mb}")

            # Generate file names
            file_names = generate_dest_file_names(num_blocks)
            
            times_write = []
            times_read = []
            
            for i in range(iterations):
                blocks_indices = random.sample(range(buffer_size // block_size), num_blocks)
                if cpp:
                    # Write benchmark
                    time_write = await cpp_write_blocks(block_size, buffer, blocks_indices, file_names)
                    # Read benchmark
                    time_read = await cpp_read_blocks(block_size, buffer, blocks_indices, file_names)
                else:
                    # Write benchmark
                    time_write = await aiofiles_write_blocks(block_size, buffer, blocks_indices, file_names)
                    # Read benchmark
                    time_read = await aiofiles_read_blocks(block_size, buffer, blocks_indices, file_names)

                times_write.append(time_write)
                times_read.append(time_read)
            
            # Cleanup files after this test to free up space
            for file_name in file_names:
                try:
                    if os.path.exists(file_name):
                        os.remove(file_name)
                except Exception as e:
                    print(f"    Warning: Could not remove {file_name}: {e}")
            
            # Shutdown executor
            if not cpp:
                executor.shutdown(wait=True)
            
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
    
    # Combine results
    all_results = {
        'write': write_results,
        'read': read_results,
        'config': {
            'buffer_size': buffer_size,
            'num_iterations': iterations,
            'num_blocks': num_blocks,
            'thread_counts': thread_counts,
            'block_sizes_mb': block_sizes_mb,
            'file_system': 'tmpfs (/dev/shm)',
            'implementation': implementation
        }
    }
    
    # Save results to JSON
    output_file = f'results/block_size_comparison_{num_blocks}blocks_cpp_{cpp}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
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

    asyncio.run(block_size_comparison(num_blocks=1000, iterations=5, buffer_size=70*1024*1024*1024, cpp=True))
    asyncio.run(block_size_comparison(num_blocks=1000, iterations=5, buffer_size=70*1024*1024*1024, cpp=False))
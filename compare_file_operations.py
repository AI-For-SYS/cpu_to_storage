import asyncio
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import time
import os
import statistics
import torch
import random
import argparse

from checkpoints_utils import save_incremental_results, load_existing_results, check_config_match, get_completed_tests
from backends.nixl_backend import nixl_write_blocks, nixl_read_blocks, nixl_register_buffer, nixl_unregister_buffer, _get_read_agent, _get_write_agent
from backends.aiofiles_backend import aiofiles_write_blocks, aiofiles_read_blocks
from backends.python_self_backend import python_self_write_blocks, python_self_read_blocks
from backends.cpp_backend import cpp_write_blocks, cpp_read_blocks, set_thread_count_cpp, CPP_AVAILABLE

# Storage path configuration - can be overridden by STORAGE_PATH environment variable
STORAGE_PATH = os.environ.get('STORAGE_PATH', '/dev/shm')
CLUSTER = os.environ.get('CLUSTER_NAME', 'unknown')


def generate_dest_file_names(num_files):
    return [f"{STORAGE_PATH}/final_{j}.bin" for j in range(num_files)]

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

async def blocks_benchmark(num_blocks, iterations, buffer_size, implementation, test_name, block_sizes_mb, threads_counts, verify=False):
    """Benchmark different block sizes across multiple thread counts.
    
    Args:
        num_blocks: number of blocks to transfer in each test
        iterations: Number of iterations per test (default: 5)
        buffer_size: Size of buffer in bytes
        implementation: Implementation to test
        test_name: Output file name prefix
        threads_counts: List of thread counts to test (default: [16, 32, 64])
        block_sizes_mb: List of block sizes in MB to test (default: [2, 4, 8, 16, 32])
        verify: Whether to verify file contents after operations (default: False)
    """
    start_time = time.perf_counter()

    python_backends = ["python_aiofiles", "python_self_implementation", "nixl"]
    if implementation=="cpp":
        if not CPP_AVAILABLE:
            print("cpp implementation not available.")
            return
    elif implementation not in python_backends:
        print("Invalid implementation specified.")
        return
    
    diff = block_sizes_mb[-1]*num_blocks - buffer_size/(1024*1024)
    if diff>0:
        print(f"Buffer is not large enugh, missing {diff} mb")
        return

    # Calculate total combinations
    total_combinations = len(block_sizes_mb) * len(threads_counts)
    print(f"Testing {total_combinations} combinations ({len(block_sizes_mb)} block sizes × {len(threads_counts)} threads)")
    
    output_file = f'results/blocks_{test_name}_{implementation}_{num_blocks}blocks.json'
    existing_results: dict[Any, Any] | None = load_existing_results(output_file)
    
    # Store results - structure: results[thread_count][block_size] = time
    write_results = {}
    read_results = {}
    
    # Configuration for this benchmark
    config = {
        'cluster': CLUSTER,
        'buffer_size': buffer_size,
        'num_iterations': iterations,
        'num_blocks': num_blocks,
        'thread_counts': threads_counts,
        'block_sizes_mb': block_sizes_mb,
        'file_system': f'{STORAGE_PATH} ({"tmpfs" if STORAGE_PATH == "/dev/shm" else "persistent storage"})',
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
    view: memoryview[int] = memoryview(buffer.numpy()).cast('B')
 
    file_names = generate_dest_file_names(num_blocks)

    combination_count = 0
    
    for num_threads in threads_counts:

        if str(num_threads) not in write_results:
            write_results[str(num_threads)] = {}
        if str(num_threads) not in read_results:
            read_results[str(num_threads)] = {}

        # Set up thread pool executor
        if implementation=="cpp":
            set_thread_count_cpp(num_threads)
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
                clean_files(file_names)

                blocks_indices_write = random.sample(range(buffer_size // block_size), num_blocks)
                blocks_indices_read = random.sample(range(buffer_size // block_size), num_blocks)

                if implementation=="cpp":
                    time_write = await cpp_write_blocks(block_size, buffer, blocks_indices_write, file_names, view)
                    verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)

                    time_read = await cpp_read_blocks(block_size, buffer, blocks_indices_read, file_names, view)

                elif implementation=="python_aiofiles":
                    time_write = await aiofiles_write_blocks(block_size, view, blocks_indices_write, file_names)
                    verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)

                    time_read = await aiofiles_read_blocks(block_size, view, blocks_indices_read, file_names)

                elif implementation=="nixl":
                    reg_handler_write = nixl_register_buffer(_get_write_agent(), buffer)
                    time_write = nixl_write_blocks(block_size, buffer, blocks_indices_write, file_names)
                    verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)
                    nixl_unregister_buffer(_get_write_agent(), reg_handler_write)

                    reg_handler_read = nixl_register_buffer(_get_read_agent(), buffer)
                    time_read = nixl_read_blocks(block_size, buffer, blocks_indices_read, file_names)
                    nixl_unregister_buffer(_get_read_agent(), reg_handler_read)
                           
                else:
                    time_write = await python_self_write_blocks(block_size, view, blocks_indices_write, file_names)
                    verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)

                    time_read = await python_self_read_blocks(block_size, view, blocks_indices_read, file_names)
       
                verify_op(block_size, blocks_indices_read, view, file_names, "Reading", verify)

                times_write.append(time_write)
                times_read.append(time_read)
            
            avg_write = statistics.mean(times_write)
            avg_read = statistics.mean(times_read)

            write_results[str(num_threads)][str(block_size_mb)] = avg_write
            read_results[str(num_threads)][str(block_size_mb)] = avg_read
            
            write_throughput = block_size_mb*num_blocks / (avg_write*1024)
            read_throughput = block_size_mb*num_blocks / (avg_read*1024)
            
            print(f"    Write: {avg_write*1000:.2f}ms ({write_throughput:.2f} GB/s)")
            print(f"    Read:  {avg_read*1000:.2f}ms ({read_throughput:.2f} GB/s)")
            
            # Save results incrementally after each test
            current_results = {
                'config': config,
                'write': write_results,
                'read': read_results
            }
            save_incremental_results(output_file, current_results)
            print(f"    Results saved to {output_file}")
        
        # Shutdown executor
        if implementation in python_implementations:
            executor.shutdown(wait=True)
            
    # Final save with complete results
    all_results = {
        'config': config,
        'write': write_results,
        'read': read_results
    }
    
    save_incremental_results(output_file, all_results)
    
    total_time = time.perf_counter() - start_time
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Total time: {total_time:.2f}s")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")
    
    return all_results


async def total_data_benchmark(total_gb, iterations, buffer_size, implementation, test_name, block_sizes_mb, threads_counts, verify=False):
    """Benchmark different block sizes across multiple thread counts.
    
    Args:
        total_gb: Total data size in GB to transfer for each test
        iterations: Number of iterations per test
        buffer_size: Size of buffer in bytes
        implementation: Implementation to test
        test_name: Output file name prefix
        thread_counts: List of thread counts to test (default: [16, 32, 64])
        block_sizes_mb: List of block sizes in MB to test (default: [2, 4, 8, 16, 32, 64])
        verify: Whether to verify file contents after operations (default: False)
    """
    start_time = time.perf_counter()

    if implementation=="cpp":
        if not CPP_AVAILABLE:
            print("cpp implementation not available.")
            return

    # Calculate total combinations
    total_combinations = len(block_sizes_mb) * len(threads_counts)
    print(f"Testing {total_combinations} combinations ({len(block_sizes_mb)} block sizes × {len(threads_counts)} threads)")
    print(f"Fixed total data size per test: {total_gb} GB)")
    
    output_file = f'results/data_{test_name}_{total_gb}gb_memory_{implementation}.json'
    existing_results = load_existing_results(output_file)
    
    # Store results - structure: results[block_size][thread_count] = time
    write_results = {}
    read_results = {}
    
    # Configuration for this benchmark
    config = {
        'cluster': CLUSTER,
        'buffer_size': buffer_size,
        'num_iterations': iterations,
        'total_data_size_gb': total_gb,
        'threads_counts': threads_counts,
        'block_sizes_mb': block_sizes_mb,
        'file_system': f'{STORAGE_PATH} ({"tmpfs" if STORAGE_PATH == "/dev/shm" else "persistent storage"})',
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
    view: memoryview[int] = memoryview(buffer.numpy()).cast('B')

    combination_count = 0
    
    for num_threads in threads_counts:
                
        if str(num_threads) not in write_results:
            write_results[str(num_threads)] = {}
        if str(num_threads) not in read_results:
            read_results[str(num_threads)] = {}
        
                # Set up thread pool executor
        if implementation=="cpp":
            set_thread_count_cpp(num_threads)
        else:
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=num_threads)
            loop.set_default_executor(executor)
        
        for block_size_mb in block_sizes_mb:

            combination_count += 1
            num_blocks_to_copy = int((total_gb * 1024) / block_size_mb)
            print(f"\n{'='*80}")
            print(f"Block Size: {block_size_mb:.0f}MB | Blocks to copy: {num_blocks_to_copy} | Total: {total_gb}GB")
            print(f"{'='*80}")
            block_size = block_size_mb * 1024 * 1024
            
            # Check if this combination is already completed
            test_key = (str(block_size), str(num_threads))
            if test_key in completed_write and test_key in completed_read:
                print(f"\n  [{combination_count}/{total_combinations}] Threads: {num_threads} - SKIPPED (already completed)")
                continue
            
            print(f"\n  [{combination_count}/{total_combinations}] Threads: {num_threads}")
            
            # Generate file names
            file_names = generate_dest_file_names(num_blocks_to_copy)
            
            times_write = []
            times_read = []
            
            for i in range(iterations):
                blocks_indices_write = random.sample(range(buffer_size // block_size), num_blocks_to_copy)
                blocks_indices_read = random.sample(range(buffer_size // block_size), num_blocks_to_copy)
                
                if implementation=="cpp":
                    time_write = await cpp_write_blocks(block_size, buffer, blocks_indices_write, file_names, view)
                    verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)

                    time_read = await cpp_read_blocks(block_size, buffer, blocks_indices_read, file_names, view)
                
                elif implementation=="python_aiofiles":
                    time_write = await aiofiles_write_blocks(block_size, view, blocks_indices_write, file_names)
                    verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)

                    time_read = await aiofiles_read_blocks(block_size, view, blocks_indices_read, file_names)
                
                elif implementation=="nixl":
                    reg_handler_write = nixl_register_buffer(_get_write_agent(), buffer)
                    time_write = nixl_write_blocks(block_size, buffer, blocks_indices_write, file_names)
                    verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)
                    nixl_unregister_buffer(_get_write_agent(), reg_handler_write)

                    reg_handler_read = nixl_register_buffer(_get_read_agent(), buffer)
                    time_read = nixl_read_blocks(block_size, buffer, blocks_indices_read, file_names)
                    nixl_unregister_buffer(_get_read_agent(), reg_handler_read)
                else:
                    time_write = await python_self_write_blocks(block_size, view, blocks_indices_write, file_names)
                    verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)

                    time_read = await python_self_read_blocks(block_size, view, blocks_indices_read, file_names)
         
                verify_op(block_size, blocks_indices_read, view, file_names, "Reading", verify)
                
                times_write.append(time_write)
                times_read.append(time_read)
                clean_files(file_names)

            
            avg_write = statistics.mean(times_write)
            avg_read = statistics.mean(times_read)

            write_results[str(num_threads)][str(block_size_mb)] = avg_write
            read_results[str(num_threads)][str(block_size_mb)] = avg_read
            
            write_throughput = total_gb / avg_write
            read_throughput = total_gb / avg_read
            
            print(f"    Write: {avg_write*1000:.2f}ms ({write_throughput:.2f} GB/s)")
            print(f"    Read:  {avg_read*1000:.2f}ms ({read_throughput:.2f} GB/s)")
            
            # Save results incrementally after each test
            current_results = {
                'config': config,
                'write': write_results,
                'read': read_results
            }
            save_incremental_results(output_file, current_results)
            print(f"    Results saved to {output_file}")
        
        if implementation in python_implementations:
            executor.shutdown(wait=True)
    
    # Final save with complete results
    all_results = {
        'config': config,
        'write': write_results,
        'read': read_results
    }
    
    save_incremental_results(output_file, all_results)
    
    total_time = time.perf_counter() - start_time
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Total time: {total_time:.2f}s")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Run I/O benchmark with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['blocks', 'data'],
        default='blocks',
        help='Benchmark mode: "blocks" for fixed number of blocks, "data" for fixed total data size (default: blocks)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['cpp', 'python_aiofiles', 'python_self_implementation', 'nixl'],
        default='python_self_implementation',
        help='I/O backend to benchmark (default: python_self_implementation)'
    )
    parser.add_argument(
        '--buffer-size',
        type=int,
        default=100,
        help='Buffer size in GB (default: 100)'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of iterations per test (default: 5)'
    )
    parser.add_argument(
        '--num-blocks',
        type=int,
        default=1000,
        help='Number of blocks to transfer (for blocks mode, default: 1000)'
    )
    parser.add_argument(
        '--test-name',
        type=str,
        default="",
        help='Output file name prefix'
    )
    parser.add_argument(
        '--total-gb',
        type=int,
        default=100,
        help='Total data size in GB (for total mode, default: 100)'
    )
    parser.add_argument(
        '--block-sizes',
        type=int,
        nargs='+',
        default=[2, 4, 8, 16, 32, 64],
        help='List of block sizes in MB to test. Example: --block-sizes 2 4 8 16'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        default=False,
        help='Verify file contents after write/read operations (default: False)'
    )
    
    args = parser.parse_args()
    
    # Convert buffer size from GB to bytes
    buffer_size = args.buffer_size * 1024 * 1024 * 1024
    threads_counts = [16, 32, 64]
    
    print("="*80)
    print("I/O BENCHMARK CONFIGURATION")
    print("="*80)
    print(f"Mode:            {args.mode}")
    print(f"Backend:         {args.backend}")
    print(f"Buffer Size:     {args.buffer_size} GB")
    print(f"Iterations:      {args.iterations}")
    print(f"Block Sizes:     {args.block_sizes} MB")
    print(f"Storage Path:    {STORAGE_PATH}")
    print(f"Cluster:         {CLUSTER}")
    print(f"Test_name:       {args.test_name}")
    print(f"Verify:          {args.verify}")

    if args.mode == 'blocks':
        print(f"Num Blocks:      {args.num_blocks}")
        print("="*80)
        print()
        
        asyncio.run(blocks_benchmark(
            num_blocks=args.num_blocks,
            iterations=args.iterations,
            buffer_size=buffer_size,
            implementation=args.backend,
            test_name=args.test_name,
            block_sizes_mb=args.block_sizes,
            threads_counts=threads_counts,
            verify=args.verify
        ))
    
    elif args.mode == 'data':
        print(f"Total Data:      {args.total_gb} GB")
        print("="*80)
        print()
        
        asyncio.run(total_data_benchmark(
            total_gb=args.total_gb,
            iterations=args.iterations,
            buffer_size=buffer_size,
            implementation=args.backend,
            test_name=args.test_name,
            block_sizes_mb=args.block_sizes,
            threads_counts=threads_counts,
            verify=args.verify
        ))

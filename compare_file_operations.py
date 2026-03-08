import asyncio
import time
import statistics
import random
import argparse

from utils.config import STORAGE_PATH, CLUSTER, PYTHON_BACKENDS
from utils.file_utils import generate_dest_file_names, clean_files
from utils.cache_clear_utils import setup_cleaning_files, write_cleaning_blocks
from utils.benchmark_core import (
    create_benchmark_config, load_or_create_results, setup_executor, allocate_buffers,
    shutdown_executor, save_results, print_benchmark_summary, run_benchmark_iteration
)
from backends.cpp_backend import CPP_AVAILABLE

async def blocks_benchmark(num_blocks, iterations, buffer_size, implementation, test_name, block_sizes_mb, threads_counts, verify=False):
    start_time = time.perf_counter()

    if implementation=="cpp":
        if not CPP_AVAILABLE:
            print("cpp implementation not available.")
            return

    diff = block_sizes_mb[-1]*num_blocks - buffer_size/(1024*1024)
    if diff>0:
        print(f"Buffer is not large enugh, missing {diff} mb")
        return

    total_combinations = len(block_sizes_mb) * len(threads_counts)
    print(f"Testing {total_combinations} combinations ({len(block_sizes_mb)} block sizes × {len(threads_counts)} threads)")
    
    output_file = f'results/blocks_{test_name}_{implementation}_{num_blocks}blocks.json'
    
    config = create_benchmark_config(
        buffer_size, iterations, threads_counts, block_sizes_mb, implementation,
        num_blocks=num_blocks
    )
    
    write_results, read_results, completed_write, completed_read = load_or_create_results(output_file, config)
    
    buffer, buffer_cleaning, view, view_cleaning = allocate_buffers(buffer_size, verify)
    file_names = generate_dest_file_names("final", num_blocks)
    
    file_names_cleaning, indices_cleaning, block_size_cleaning = setup_cleaning_files()
    
    await write_cleaning_blocks(block_size_cleaning, view_cleaning, indices_cleaning, file_names_cleaning)

    combination_count = 0
    
    for num_threads in threads_counts:

        if str(num_threads) not in write_results:
            write_results[str(num_threads)] = {}
        if str(num_threads) not in read_results:
            read_results[str(num_threads)] = {}

        executor = setup_executor(implementation, num_threads)

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

                time_write, time_read = await run_benchmark_iteration(
                    implementation, block_size, buffer, buffer_cleaning, view, view_cleaning,
                    blocks_indices_write, blocks_indices_read, file_names,
                    file_names_cleaning, indices_cleaning, block_size_cleaning, verify
                )

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
            save_results(output_file, config, write_results, read_results)
            print(f"    Results saved to {output_file}")
        
        shutdown_executor(implementation, executor)
            
    all_results = save_results(output_file, config, write_results, read_results)
    
    total_time = time.perf_counter() - start_time
    print_benchmark_summary(total_time, output_file)
    
    return all_results


async def total_data_benchmark(total_gb, iterations, buffer_size, implementation, test_name, block_sizes_mb, threads_counts, verify=False):
    start_time = time.perf_counter()

    if implementation=="cpp":
        if not CPP_AVAILABLE:
            print("cpp implementation not available.")
            return

    total_combinations = len(block_sizes_mb) * len(threads_counts)
    print(f"Testing {total_combinations} combinations ({len(block_sizes_mb)} block sizes × {len(threads_counts)} threads)")
    print(f"Fixed total data size per test: {total_gb} GB)")
    
    output_file = f'results/data_{test_name}_{total_gb}gb_memory_{implementation}.json'
    
    config = create_benchmark_config(
        buffer_size, iterations, threads_counts, block_sizes_mb, implementation,
        total_data_size_gb=total_gb
    )
    
    write_results, read_results, completed_write, completed_read = load_or_create_results(output_file, config)
    
    buffer, buffer_cleaning, view, view_cleaning = allocate_buffers(buffer_size, verify)
    file_names_cleaning, indices_cleaning, block_size_cleaning = setup_cleaning_files()
    
    await write_cleaning_blocks(block_size_cleaning, view_cleaning, indices_cleaning, file_names_cleaning)

    combination_count = 0
    
    for num_threads in threads_counts:
                
        if str(num_threads) not in write_results:
            write_results[str(num_threads)] = {}
        if str(num_threads) not in read_results:
            read_results[str(num_threads)] = {}
        
        executor = setup_executor(implementation, num_threads)
        
        for block_size_mb in block_sizes_mb:

            combination_count += 1
            num_blocks_to_copy = int((total_gb * 1024) / block_size_mb)
            print(f"\n{'='*80}")
            print(f"Block Size: {block_size_mb:.0f}MB | Blocks to copy: {num_blocks_to_copy} | Total: {total_gb}GB")
            print(f"{'='*80}")
            block_size = block_size_mb * 1024 * 1024
            
            # Check if this combination is already completed
            test_key = (str(num_threads), str(block_size_mb))
            if test_key in completed_write and test_key in completed_read:
                print(f"\n  [{combination_count}/{total_combinations}] Threads: {num_threads} - SKIPPED (already completed)")
                continue
            
            print(f"\n  [{combination_count}/{total_combinations}] Threads: {num_threads}")
            
            # Generate file names
            file_names = generate_dest_file_names("final", num_blocks_to_copy)
            
            times_write = []
            times_read = []
            
            for i in range(iterations):
                blocks_indices_write = random.sample(range(buffer_size // block_size), num_blocks_to_copy)
                blocks_indices_read = random.sample(range(buffer_size // block_size), num_blocks_to_copy)
                
                time_write, time_read = await run_benchmark_iteration(
                    implementation, block_size, buffer, buffer_cleaning, view, view_cleaning,
                    blocks_indices_write, blocks_indices_read, file_names,
                    file_names_cleaning, indices_cleaning, block_size_cleaning, verify
                )
                
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
            save_results(output_file, config, write_results, read_results)
            print(f"    Results saved to {output_file}")
        
        shutdown_executor(implementation, executor)
    
    all_results = save_results(output_file, config, write_results, read_results)
    
    total_time = time.perf_counter() - start_time
    print_benchmark_summary(total_time, output_file)
    
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
        choices=['cpp', 'python_aiofiles', 'python_self_imp', 'nixl'],
        default='python_self_imp',
        help='I/O backend to benchmark (default: python_self_imp)'
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

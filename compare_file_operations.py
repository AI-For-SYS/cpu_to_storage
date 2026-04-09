import asyncio
import time
import statistics
import random
import argparse

from utils.config import STORAGE_PATH, CLUSTER
from utils.file_utils import generate_dest_file_names, clean_files, write_blocks
from utils.benchmark_core import (
    create_benchmark_config, load_or_create_results, setup_executor, allocate_buffers,
    shutdown_executor, save_results, print_benchmark_summary, run_benchmark_iteration,
    run_concurrent_benchmark_iteration, setup_cleaning_files, load_tunable_config
)
from backends.cpp_backend import CPP_AVAILABLE
from backends.threaded_tunable_backend import THREADED_TUNABLE_AVAILABLE
from utils.checkpoints_utils import save_incremental_results

async def blocks_benchmark(num_blocks, iterations, buffer_size, implementation, test_name, block_sizes_mb, threads_counts, verify=False):
    start_time = time.perf_counter()

    if implementation=="cpp":
        if not CPP_AVAILABLE:
            print("cpp implementation not available.")
            return
    if implementation=="threaded_tunable":
        if not THREADED_TUNABLE_AVAILABLE:
            print("threaded_tunable implementation not available.")
            return

    diff = block_sizes_mb[-1]*num_blocks - buffer_size/(1024*1024)
    if diff>0:
        print(f"Buffer is not large enugh, missing {diff} mb")
        return

    total_combinations = len(block_sizes_mb) * len(threads_counts)
    print(f"Testing {total_combinations} combinations ({len(block_sizes_mb)} block sizes × {len(threads_counts)} threads)")
    
    dest = "tmpfs" if STORAGE_PATH== "/dev/shm" else "storage"

    output_file = f'results/blocks_{num_blocks}_{test_name}_{dest}_{implementation}.json'
    
    config = create_benchmark_config(
        buffer_size, iterations, threads_counts, block_sizes_mb, implementation,
        num_blocks=num_blocks
    )
    
    write_results, read_results, completed_write, completed_read = load_or_create_results(output_file, config)
    
    buffer, buffer_cleaning, view, view_cleaning = allocate_buffers(buffer_size, verify)
    file_names = generate_dest_file_names("final", num_blocks)
    
    file_names_cleaning, indices_cleaning, block_size_cleaning = setup_cleaning_files()
    
    await write_blocks(block_size_cleaning, view_cleaning, indices_cleaning, file_names_cleaning)

    combination_count = 0
    block_sizes_mb.reverse()
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
    if implementation=="threaded_tunable":
        if not THREADED_TUNABLE_AVAILABLE:
            print("threaded_tunable implementation not available.")
            return

    total_combinations = len(block_sizes_mb) * len(threads_counts)
    print(f"Testing {total_combinations} combinations ({len(block_sizes_mb)} block sizes × {len(threads_counts)} threads)")
    print(f"Fixed total data size per test: {total_gb} GB)")

    dest = "tmpfs" if STORAGE_PATH== "/dev/shm" else "storage"
    
    output_file = f'results/data_{test_name}_{total_gb}gb_{dest}_{implementation}.json'
    
    config = create_benchmark_config(
        buffer_size, iterations, threads_counts, block_sizes_mb, implementation,
        total_data_size_gb=total_gb
    )
    
    write_results, read_results, completed_write, completed_read = load_or_create_results(output_file, config)
    
    buffer, buffer_cleaning, view, view_cleaning = allocate_buffers(buffer_size, verify)
    file_names_cleaning, indices_cleaning, block_size_cleaning = setup_cleaning_files()
    
    await write_blocks(block_size_cleaning, view_cleaning, indices_cleaning, file_names_cleaning)

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
            block_size = block_size_mb * 1024 * 1024
            
            # Check if this combination is already completed
            test_key = (str(num_threads), str(block_size_mb))
            if test_key in completed_write and test_key in completed_read:
                print(f"\n  [{combination_count}/{total_combinations}] Block Size: {block_size_mb:.0f}MB | Blocks: {num_blocks_to_copy} | Threads: {num_threads} - SKIPPED")
                continue
            
            print(f"\n  [{combination_count}/{total_combinations}] Block Size: {block_size_mb:.0f}MB | Blocks: {num_blocks_to_copy} | Total: {total_gb}GB | Threads: {num_threads}")
            
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
    clean_files(file_names_cleaning)
    total_time = time.perf_counter() - start_time
    print_benchmark_summary(total_time, output_file)
    
    return all_results


async def concurrent_benchmark(total_gb, iterations, buffer_size, implementation, test_name, block_sizes_mb, threads_counts, verify=False):
    """Benchmark concurrent read and write operations.
    
    This benchmark runs read and write operations simultaneously to measure:
    - Combined throughput when both operations compete for I/O resources
    - How well the system handles mixed workloads
    - Individual read/write performance under concurrent load
    
    Note: Uses only half of the buffer for data (the other half remains available).
    """
    start_time = time.perf_counter()

    if implementation=="cpp":
        if not CPP_AVAILABLE:
            print("cpp implementation not available.")
            return
    if implementation=="threaded_tunable":
        if not THREADED_TUNABLE_AVAILABLE:
            print("threaded_tunable implementation not available.")
            return

    total_combinations = len(block_sizes_mb) * len(threads_counts)
    print(f"Testing {total_combinations} combinations ({len(block_sizes_mb)} block sizes × {len(threads_counts)} threads)")
    print(f"Total data size: {total_gb} GB split between read ({total_gb/2} GB) and write ({total_gb/2} GB)")
    
    dest = "tmpfs" if STORAGE_PATH== "/dev/shm" else "storage"
    output_file = f'results/concurrent_{test_name}_{total_gb}gb_{dest}_{implementation}.json'
    
    config = create_benchmark_config(
        buffer_size, iterations, threads_counts, block_sizes_mb, implementation,
        total_data_size_gb=total_gb,
        mode='concurrent'
    )
    
    write_results, read_results, completed_write, completed_read = load_or_create_results(output_file, config)
    
    # Initialize concurrent results dictionary
    concurrent_results = {}
    
    buffer, buffer_cleaning, view, view_cleaning = allocate_buffers(buffer_size, verify)
    file_names_cleaning, indices_cleaning, block_size_cleaning = setup_cleaning_files()
    
    await write_blocks(block_size_cleaning, view_cleaning, indices_cleaning, file_names_cleaning)

    combination_count = 0
    
    for num_threads in threads_counts:
                
        if str(num_threads) not in write_results:
            write_results[str(num_threads)] = {}
        if str(num_threads) not in read_results:
            read_results[str(num_threads)] = {}
        if str(num_threads) not in concurrent_results:
            concurrent_results[str(num_threads)] = {}
        
        executor = setup_executor(implementation, num_threads)
        
        for block_size_mb in block_sizes_mb:

            combination_count += 1
            num_blocks = int((total_gb * 1024 / 2) / block_size_mb)
            block_size = block_size_mb * 1024 * 1024
            
            # Check if this combination is already completed
            test_key = (str(num_threads), str(block_size_mb))
            if test_key in completed_write and test_key in completed_read:
                print(f"\n  [{combination_count}/{total_combinations}] Block Size: {block_size_mb:.0f}MB | Read/Write blocks: {num_blocks} each ({total_gb/2}GB each) | Threads: {num_threads} - SKIPPED")
                continue
            
            print(f"\n  [{combination_count}/{total_combinations}] Block Size: {block_size_mb:.0f}MB | Read/Write blocks: {num_blocks} each ({total_gb/2}GB each) | Threads: {num_threads}")
            
            # Generate separate file names for read and write operations
            file_names_write = generate_dest_file_names("write", num_blocks)
            file_names_read = generate_dest_file_names("read", num_blocks)
            
            # Calculate buffer halves
            total_blocks = buffer_size // block_size
            first_half_blocks = total_blocks // 2
            
            # Pre-populate read files once before all iterations

            await write_blocks(block_size, view, list(range(num_blocks)), file_names_read)
            
            times_write = []
            times_read = []
            times_concurrent = []
            
            for i in range(iterations):
                # Generate random indices for concurrent operations
                # Write uses first half of buffer (0 to first_half_blocks)
                # Read uses second half of buffer (first_half_blocks to total_blocks)
                blocks_indices_write = random.sample(range(0, first_half_blocks), num_blocks)
                blocks_indices_read = random.sample(range(first_half_blocks, total_blocks), num_blocks)
                
                time_write, time_read, time_concurrent = await run_concurrent_benchmark_iteration(
                    implementation, block_size, buffer, buffer_cleaning, view, view_cleaning,
                    blocks_indices_write, blocks_indices_read, file_names_write, file_names_read,
                    file_names_cleaning, indices_cleaning, block_size_cleaning, verify
                )
                
                times_write.append(time_write)
                times_read.append(time_read)
                times_concurrent.append(time_concurrent)
                
                # Clean up write files after each iteration to avoid filling storage
                clean_files(file_names_write)

            
            avg_write = statistics.mean(times_write)
            avg_read = statistics.mean(times_read)
            avg_concurrent = statistics.mean(times_concurrent)

            write_results[str(num_threads)][str(block_size_mb)] = avg_write
            read_results[str(num_threads)][str(block_size_mb)] = avg_read
            concurrent_results[str(num_threads)][str(block_size_mb)] = avg_concurrent
            
            # Calculate throughput for each operation (half the total data each)
            write_throughput = (total_gb / 2) / avg_write
            read_throughput = (total_gb / 2) / avg_read
            combined_throughput = total_gb / avg_concurrent  # Total data transferred
            
            print(f"    Write (concurrent):     {avg_write*1000:.2f}ms ({write_throughput:.2f} GB/s)")
            print(f"    Read (concurrent):      {avg_read*1000:.2f}ms ({read_throughput:.2f} GB/s)")
            print(f"    Total concurrent time:  {avg_concurrent*1000:.2f}ms")
            print(f"    Combined throughput:    {combined_throughput:.2f} GB/s")
            
            # Clean up read files after all iterations for this block size are complete
            clean_files(file_names_read)
            
            # Save results incrementally after each test
            all_results = {
                'config': config,
                'write': write_results,
                'read': read_results,
                'concurrent': concurrent_results
            }
            save_results(output_file, config, write_results, read_results)
            print(f"    Results saved to {output_file}")
        
        shutdown_executor(implementation, executor)
    
    # Final save with all three result types
    all_results = {
        'config': config,
        'write': write_results,
        'read': read_results,
        'concurrent': concurrent_results
    }
    save_incremental_results(output_file, all_results)
    
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
        choices=['blocks', 'data', 'concurrent'],
        default='blocks',
        help='Benchmark mode: "blocks" for fixed number of blocks, "data" for fixed total data size, "concurrent" for simultaneous read/write (default: blocks)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['cpp', 'python_aiofiles', 'python_self_imp', 'nixl', 'threaded_tunable'],
        default='python_self_imp',
        help='I/O backend to benchmark (default: python_self_imp)'
    )
    parser.add_argument(
        '--tunable-config',
        type=str,
        default=None,
        help='Path to JSON config for threaded_tunable backend (e.g., results/best_write_config.json)'
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
    
    # Clean up any leftover files from previous runs
    print("="*80)
    print("CLEANING UP STORAGE")
    print("="*80)
    print(f"Removing all files from {STORAGE_PATH}...")
    import subprocess
    try:
        subprocess.run(f"rm -f {STORAGE_PATH}/*", shell=True, check=False)
        print("Cleanup complete\n")
    except Exception as e:
        print(f"Warning: Cleanup failed: {e}\n")
    
    # Load tunable config if provided
    if args.tunable_config:
        if args.backend != "threaded_tunable":
            print("Warning: --tunable-config is only used with --backend threaded_tunable, ignoring")
        else:
            load_tunable_config(args.tunable_config)

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
    if args.tunable_config and args.backend == "threaded_tunable":
        print(f"Tunable Config:  {args.tunable_config}")

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
    
    elif args.mode == 'concurrent':
        print(f"Total Data:      {args.total_gb} GB per operation (read + write)")
        print(f"Mode:            Concurrent read/write operations")
        print("="*80)
        print()
        
        asyncio.run(concurrent_benchmark(
            total_gb=args.total_gb,
            iterations=args.iterations,
            buffer_size=buffer_size,
            implementation=args.backend,
            test_name=args.test_name,
            block_sizes_mb=args.block_sizes,
            threads_counts=threads_counts,
            verify=args.verify
        ))

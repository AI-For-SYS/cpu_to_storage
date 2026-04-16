import asyncio
import time
import statistics
import random
import argparse
from datetime import datetime

from utils.config import STORAGE_PATH, CLUSTER
from utils.file_utils import generate_dest_file_names, clean_files, write_blocks
from utils.benchmark_core import (
    create_benchmark_config, load_or_create_results, setup_executor, allocate_buffers,
    shutdown_executor, print_benchmark_summary, run_benchmark_iteration,
    run_concurrent_benchmark_iteration, setup_cleaning_files, load_tunable_config,
    get_tunable_config
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results/blocks_{num_blocks}_{test_name}_{dest}_{implementation}_{ts}.json'
    
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

            # Save results incrementally with throughput
            write_tp = {}
            read_tp = {}
            for tc in write_results:
                write_tp[tc] = {bs: int(bs) * num_blocks / (t * 1024)
                                for bs, t in write_results[tc].items()}
            for tc in read_results:
                read_tp[tc] = {bs: int(bs) * num_blocks / (t * 1024)
                               for bs, t in read_results[tc].items()}
            all_results = {
                'config': config,
                'write': write_results, 'write_throughput_gbs': write_tp,
                'read': read_results, 'read_throughput_gbs': read_tp,
            }
            save_incremental_results(output_file, all_results)
            print(f"    Results saved to {output_file}")

        shutdown_executor(implementation, executor)

    clean_files(file_names)
    clean_files(file_names_cleaning)

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
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results/data_{test_name}_{total_gb}gb_{dest}_{implementation}_{ts}.json'
    
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

            # Save results incrementally with throughput
            write_tp = {}
            read_tp = {}
            for tc in write_results:
                write_tp[tc] = {bs: total_gb / t for bs, t in write_results[tc].items()}
            for tc in read_results:
                read_tp[tc] = {bs: total_gb / t for bs, t in read_results[tc].items()}
            all_results = {
                'config': config,
                'write': write_results, 'write_throughput_gbs': write_tp,
                'read': read_results, 'read_throughput_gbs': read_tp,
            }
            save_incremental_results(output_file, all_results)
            print(f"    Results saved to {output_file}")

        shutdown_executor(implementation, executor)

    clean_files(file_names)
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results/concurrent_{test_name}_{total_gb}gb_{dest}_{implementation}_{ts}.json'
    
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
            
            # Save results incrementally with throughput
            half_gb = total_gb / 2
            write_tp = {}
            read_tp = {}
            concurrent_tp = {}
            for tc in write_results:
                write_tp[tc] = {bs: half_gb / t for bs, t in write_results[tc].items()}
            for tc in read_results:
                read_tp[tc] = {bs: half_gb / t for bs, t in read_results[tc].items()}
            for tc in concurrent_results:
                concurrent_tp[tc] = {bs: total_gb / t for bs, t in concurrent_results[tc].items()}
            all_results = {
                'config': config,
                'write': write_results, 'write_throughput_gbs': write_tp,
                'read': read_results, 'read_throughput_gbs': read_tp,
                'concurrent': concurrent_results, 'concurrent_throughput_gbs': concurrent_tp,
            }
            save_incremental_results(output_file, all_results)
            print(f"    Results saved to {output_file}")

        shutdown_executor(implementation, executor)

    clean_files(file_names_cleaning)

    total_time = time.perf_counter() - start_time
    print_benchmark_summary(total_time, output_file)

    return all_results


async def tunable_benchmark(num_blocks, iterations, buffer_size, test_name, total_gb=None, verify=False):
    """Benchmark threaded_tunable with separate best configs for write, read, and concurrent.

    Runs three passes:
    1. Write pass with write config
    2. Read pass with read config
    3. Concurrent pass with concurrent config (simultaneous write + read)

    Args:
        num_blocks: Fixed number of blocks per pass (blocks mode).
                    Ignored if total_gb is set.
        total_gb: Fixed total data in GB (data mode).
                  When set, num_blocks is calculated per pass from total_gb / block_size_mb.
    """
    from utils.benchmark_core import (
        run_tunable_write, run_tunable_read, get_tunable_config,
        threaded_tunable_configure
    )
    from backends.threaded_tunable_backend import (
        threaded_tunable_write_blocks, threaded_tunable_read_blocks
    )

    start_time = time.perf_counter()

    if not THREADED_TUNABLE_AVAILABLE:
        print("threaded_tunable implementation not available.")
        return

    write_cfg = get_tunable_config("write")
    read_cfg = get_tunable_config("read")
    write_block_mb = write_cfg.block_size_mb or 32
    read_block_mb = read_cfg.block_size_mb or 8

    dest = "tmpfs" if STORAGE_PATH == "/dev/shm" else "storage"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if total_gb is not None:
        output_file = f'results/data_{test_name}_{total_gb}gb_{dest}_threaded_tunable_{ts}.json'
    else:
        output_file = f'results/blocks_{num_blocks}_{test_name}_{dest}_threaded_tunable_{ts}.json'

    # Calculate num_blocks per pass
    write_block_size = write_block_mb * 1024 * 1024
    read_block_size = read_block_mb * 1024 * 1024
    max_write_blocks = buffer_size // write_block_size
    max_read_blocks = buffer_size // read_block_size

    if total_gb is not None:
        write_num_blocks = min(int((total_gb * 1024) / write_block_mb), max_write_blocks)
        read_num_blocks = min(int((total_gb * 1024) / read_block_mb), max_read_blocks)
    else:
        write_num_blocks = min(num_blocks, max_write_blocks)
        read_num_blocks = min(num_blocks, max_read_blocks)

    write_total_gb = write_block_mb * write_num_blocks / 1024
    read_total_gb = read_block_mb * read_num_blocks / 1024

    config = create_benchmark_config(
        buffer_size, iterations,
        threads_counts=[write_cfg.thread_count, read_cfg.thread_count],
        block_sizes_mb=[write_block_mb, read_block_mb],
        implementation="threaded_tunable",
        total_data_size_gb=total_gb,
        num_blocks=num_blocks,
        tunable_write_config=write_cfg.to_dict(),
        tunable_read_config=read_cfg.to_dict(),
        tunable_concurrent_config=get_tunable_config("concurrent").to_dict(),
    )

    buffer, buffer_cleaning, view, view_cleaning = allocate_buffers(buffer_size, verify)
    file_names_cleaning, indices_cleaning, block_size_cleaning = setup_cleaning_files()
    await write_blocks(block_size_cleaning, view_cleaning, indices_cleaning, file_names_cleaning)

    write_results = {}
    read_results = {}

    # --- Write pass ---
    if write_num_blocks < (num_blocks or write_num_blocks + 1):
        print(f"  Note: capped write blocks to {write_num_blocks} (buffer={buffer_size//(1024**3)}GB, block={write_block_mb}MB)")
    file_names_write = generate_dest_file_names("final", write_num_blocks)

    print(f"\n  Write pass: block={write_block_mb}MB, threads={write_cfg.thread_count}, blocks={write_num_blocks}, total={write_total_gb:.1f}GB")
    times_write = []
    for i in range(iterations):
        clean_files(file_names_write)
        blocks_indices = random.sample(range(max_write_blocks), write_num_blocks)
        tw = await run_tunable_write(
            write_block_size, buffer, blocks_indices, file_names_write,
            buffer_cleaning, file_names_cleaning, indices_cleaning,
            block_size_cleaning, view, verify
        )
        if tw is None:
            print(f"    Write iteration {i} failed, skipping")
            continue
        times_write.append(tw)

    if not times_write:
        print("    All write iterations failed!")
        return
    avg_write = statistics.mean(times_write)
    write_throughput = write_total_gb / avg_write
    print(f"    Write: {avg_write*1000:.2f}ms ({write_throughput:.2f} GB/s)")

    write_results[str(write_cfg.thread_count)] = {str(write_block_mb): avg_write}

    # Clean write files before starting read pass
    clean_files(file_names_write)

    # --- Read pass ---
    if read_num_blocks < (num_blocks or read_num_blocks + 1):
        print(f"  Note: capped read blocks to {read_num_blocks} (buffer={buffer_size//(1024**3)}GB, block={read_block_mb}MB)")
    file_names_read = generate_dest_file_names("final", read_num_blocks)

    # Write the files that we'll read (using read block size)
    clean_files(file_names_read)
    blocks_indices_read_setup = random.sample(range(max_read_blocks), read_num_blocks)
    await run_tunable_write(
        read_block_size, buffer, blocks_indices_read_setup, file_names_read,
        buffer_cleaning, file_names_cleaning, indices_cleaning,
        block_size_cleaning, view, False
    )

    print(f"\n  Read pass: block={read_block_mb}MB, threads={read_cfg.thread_count}, blocks={read_num_blocks}, total={read_total_gb:.1f}GB")
    times_read = []
    for i in range(iterations):
        blocks_indices = random.sample(range(max_read_blocks), read_num_blocks)
        tr = await run_tunable_read(
            read_block_size, buffer, blocks_indices, file_names_read,
            buffer_cleaning, file_names_cleaning, indices_cleaning,
            block_size_cleaning, view, verify
        )
        if tr is None:
            print(f"    Read iteration {i} failed, skipping")
            continue
        times_read.append(tr)

    if not times_read:
        print("    All read iterations failed!")
        return
    avg_read = statistics.mean(times_read)
    read_throughput = read_total_gb / avg_read
    print(f"    Read:  {avg_read*1000:.2f}ms ({read_throughput:.2f} GB/s)")

    read_results[str(read_cfg.thread_count)] = {str(read_block_mb): avg_read}

    clean_files(file_names_write)
    clean_files(file_names_read)

    # --- Concurrent pass ---
    concurrent_cfg = get_tunable_config("concurrent")
    concurrent_block_mb = concurrent_cfg.block_size_mb or 32
    concurrent_block_size = concurrent_block_mb * 1024 * 1024
    max_concurrent_blocks = buffer_size // concurrent_block_size

    if total_gb is not None:
        concurrent_half_blocks = min(int((total_gb * 1024 / 2) / concurrent_block_mb), max_concurrent_blocks // 2)
    else:
        concurrent_half_blocks = min(num_blocks, max_concurrent_blocks // 2)

    concurrent_total_gb = concurrent_block_mb * concurrent_half_blocks * 2 / 1024
    half_max = max_concurrent_blocks // 2

    file_names_cwrite = generate_dest_file_names("conc_write", concurrent_half_blocks)
    file_names_cread = generate_dest_file_names("conc_read", concurrent_half_blocks)

    # Pre-write read files for concurrent test
    read_indices_setup = random.sample(range(half_max, max_concurrent_blocks), concurrent_half_blocks)
    threaded_tunable_configure(concurrent_cfg)
    await threaded_tunable_write_blocks(
        concurrent_block_size, buffer, read_indices_setup, file_names_cread
    )

    print(f"\n  Concurrent pass: block={concurrent_block_mb}MB, threads={concurrent_cfg.thread_count}, "
          f"blocks={concurrent_half_blocks}×2, total={concurrent_total_gb:.1f}GB")

    concurrent_results = {}
    times_concurrent = []
    for i in range(iterations):
        # Clean cache before concurrent
        await threaded_tunable_read_blocks(
            block_size_cleaning, buffer_cleaning, indices_cleaning, file_names_cleaning
        )
        clean_files(file_names_cwrite)

        write_indices = random.sample(range(0, half_max), concurrent_half_blocks)

        threaded_tunable_configure(concurrent_cfg)
        start_concurrent = time.perf_counter()
        write_task = threaded_tunable_write_blocks(
            concurrent_block_size, buffer, write_indices, file_names_cwrite
        )
        read_task = threaded_tunable_read_blocks(
            concurrent_block_size, buffer, read_indices_setup, file_names_cread
        )
        await asyncio.gather(write_task, read_task)
        elapsed_concurrent = time.perf_counter() - start_concurrent
        times_concurrent.append(elapsed_concurrent)

    clean_files(file_names_cwrite)
    clean_files(file_names_cread)

    if times_concurrent:
        avg_concurrent = statistics.mean(times_concurrent)
        concurrent_throughput = concurrent_total_gb / avg_concurrent
        print(f"    Concurrent: {avg_concurrent*1000:.2f}ms ({concurrent_throughput:.2f} GB/s)")
        concurrent_results[str(concurrent_cfg.thread_count)] = {str(concurrent_block_mb): avg_concurrent}

    # Build throughput sections (GB/s) alongside raw time sections
    write_throughput_results = {}
    for tc, bsizes in write_results.items():
        write_throughput_results[tc] = {bs: write_total_gb / t for bs, t in bsizes.items()}

    read_throughput_results = {}
    for tc, bsizes in read_results.items():
        read_throughput_results[tc] = {bs: read_total_gb / t for bs, t in bsizes.items()}

    concurrent_throughput_results = {}
    for tc, bsizes in concurrent_results.items():
        concurrent_throughput_results[tc] = {bs: concurrent_total_gb / t for bs, t in bsizes.items()}

    all_results = {
        'config': config,
        'write': write_results,
        'write_throughput_gbs': write_throughput_results,
        'read': read_results,
        'read_throughput_gbs': read_throughput_results,
        'concurrent': concurrent_results,
        'concurrent_throughput_gbs': concurrent_throughput_results,
    }
    from utils.checkpoints_utils import save_incremental_results
    save_incremental_results(output_file, all_results)

    clean_files(file_names_cleaning)

    total_time = time.perf_counter() - start_time
    print_benchmark_summary(total_time, output_file)
    return all_results


async def tunable_data_benchmark(total_gb, iterations, buffer_size, test_name, verify=False):
    """Data mode wrapper: calculates num_blocks from total_gb and delegates to tunable_benchmark."""
    return await tunable_benchmark(
        num_blocks=None, iterations=iterations, buffer_size=buffer_size,
        test_name=test_name, total_gb=total_gb, verify=verify
    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Run I/O benchmark with configurable parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['blocks', 'data', 'concurrent'],
        default='data',
        help='Benchmark mode: "data" for fixed total data size (default), "blocks" for fixed number of blocks, "concurrent" for simultaneous read/write'
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
        type=float,
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
    
    # Load tunable config if provided
    tunable_config_loaded = False
    if args.tunable_config:
        if args.backend != "threaded_tunable":
            print("Warning: --tunable-config is only used with --backend threaded_tunable, ignoring")
        else:
            load_tunable_config(args.tunable_config)
            tunable_config_loaded = True

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
    if tunable_config_loaded:
        print(f"Tunable Config:  {args.tunable_config}")
        write_cfg = get_tunable_config("write")
        read_cfg = get_tunable_config("read")
        print(f"  Write: threads={write_cfg.thread_count}, block={write_cfg.block_size_mb}MB")
        print(f"  Read:  threads={read_cfg.thread_count}, block={read_cfg.block_size_mb}MB")

    # Route tunable with loaded config to dedicated benchmark function
    if tunable_config_loaded and args.mode == 'data':
        print(f"Total Data:      {args.total_gb} GB")
        print("="*80)
        print()

        asyncio.run(tunable_data_benchmark(
            total_gb=args.total_gb,
            iterations=args.iterations,
            buffer_size=buffer_size,
            test_name=args.test_name,
            verify=args.verify
        ))

    elif tunable_config_loaded and args.mode == 'blocks':
        print(f"Num Blocks:      {args.num_blocks}")
        print("="*80)
        print()

        asyncio.run(tunable_benchmark(
            num_blocks=args.num_blocks,
            iterations=args.iterations,
            buffer_size=buffer_size,
            test_name=args.test_name,
            verify=args.verify
        ))

    elif args.mode == 'blocks':
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

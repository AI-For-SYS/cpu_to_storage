"""Core benchmark utilities shared between different benchmark modes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from checkpoints_utils import save_incremental_results, load_existing_results, check_config_match, get_completed_tests
from backends.cpp_backend import set_thread_count_cpp
from utils.config import PYTHON_BACKENDS, CLUSTER, STORAGE_PATH
import torch

def create_benchmark_config(buffer_size, iterations, threads_counts, block_sizes_mb, implementation, **kwargs):
    """Create configuration dictionary for benchmark."""
    config = {
        'cluster': CLUSTER,
        'buffer_size': buffer_size,
        'num_iterations': iterations,
        'threads_counts': threads_counts,
        'block_sizes_mb': block_sizes_mb,
        'file_system': f'{STORAGE_PATH} ({"tmpfs" if STORAGE_PATH == "/dev/shm" else "persistent storage"})',
        'implementation': implementation
    }
    config.update(kwargs)
    return config


def load_or_create_results(output_file, config):
    """Load existing results or create new result structure."""
    existing_results = load_existing_results(output_file)
    
    write_results = {}
    read_results = {}
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
    
    return write_results, read_results, completed_write, completed_read


def setup_executor(implementation, num_threads):
    """Setup thread pool executor for the given implementation."""
    if implementation == "cpp":
        set_thread_count_cpp(num_threads)
        return None
    else:
        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor(max_workers=num_threads)
        loop.set_default_executor(executor)
        return executor


def shutdown_executor(implementation, executor):
    """Shutdown executor if needed."""
    if implementation in PYTHON_BACKENDS and executor is not None:
        executor.shutdown(wait=True)


def save_results(output_file, config, write_results, read_results):
    """Save benchmark results to file."""
    all_results = {
        'config': config,
        'write': write_results,
        'read': read_results
    }
    save_incremental_results(output_file, all_results)
    return all_results


def print_benchmark_summary(total_time, output_file):
    """Print benchmark completion summary."""
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Total time: {total_time:.2f}s")
    print(f"Results saved to {output_file}")
    print(f"{'='*80}")


async def run_benchmark_iteration(
    implementation, block_size, buffer, buffer_cleaning, view, view_cleaning,
    blocks_indices_write, blocks_indices_read, file_names,
    file_names_cleaning, indices_cleaning, block_size_cleaning, verify
):
    """Run a single benchmark iteration for write and read operations."""
    from backends.cpp_backend import cpp_write_blocks, cpp_read_blocks
    from backends.aiofiles_backend import aiofiles_write_blocks, aiofiles_read_blocks
    from backends.python_self_backend import python_self_write_blocks, python_self_read_blocks
    from backends.nixl_backend import nixl_write_blocks, nixl_read_blocks, nixl_register_buffer, nixl_unregister_buffer, _get_read_agent, _get_write_agent
    from utils.file_utils import verify_op
    
    if implementation == "cpp":
        time_write = await cpp_write_blocks(block_size, buffer, blocks_indices_write, file_names)
        verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)
        
        # reading 100GB of other blocks to clean the cache
        await cpp_read_blocks(block_size_cleaning, buffer_cleaning, indices_cleaning, file_names_cleaning)

        time_read = await cpp_read_blocks(block_size, buffer, blocks_indices_read, file_names)
        verify_op(block_size, blocks_indices_read, view, file_names, "Reading", verify)
        
        # reading 100GB of other blocks to clean the cache
        await cpp_read_blocks(block_size_cleaning, buffer_cleaning, indices_cleaning, file_names_cleaning)

    elif implementation == "python_aiofiles":
        time_write = await aiofiles_write_blocks(block_size, view, blocks_indices_write, file_names)
        verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)
        
        # reading 100GB of other blocks to clean the cache
        await python_self_read_blocks(block_size_cleaning, view_cleaning, indices_cleaning, file_names_cleaning)

        time_read = await aiofiles_read_blocks(block_size, view, blocks_indices_read, file_names)
        verify_op(block_size, blocks_indices_read, view, file_names, "Reading", verify)
        
        # reading 100GB of other blocks to clean the cache
        await python_self_read_blocks(block_size_cleaning, view_cleaning, indices_cleaning, file_names_cleaning)

    elif implementation == "nixl":
        reg_handler_write = nixl_register_buffer(_get_write_agent(), buffer)
        time_write = nixl_write_blocks(block_size, buffer, blocks_indices_write, file_names)
        verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)
        nixl_unregister_buffer(_get_write_agent(), reg_handler_write)
        
        # reading 100GB of other blocks to clean the cache
        reg_handler_cleaning = nixl_register_buffer(_get_read_agent(), buffer_cleaning)
        nixl_read_blocks(block_size_cleaning, buffer_cleaning, indices_cleaning, file_names_cleaning)
        nixl_unregister_buffer(_get_read_agent(), reg_handler_cleaning)
        
        reg_handler_read = nixl_register_buffer(_get_read_agent(), buffer)
        time_read = nixl_read_blocks(block_size, buffer, blocks_indices_read, file_names)
        nixl_unregister_buffer(_get_read_agent(), reg_handler_read)
        verify_op(block_size, blocks_indices_read, view, file_names, "Reading", verify)
        
        # reading 100GB of other blocks to clean the cache
        reg_handler_cleaning = nixl_register_buffer(_get_read_agent(), buffer_cleaning)
        nixl_read_blocks(block_size_cleaning, buffer_cleaning, indices_cleaning, file_names_cleaning)
        nixl_unregister_buffer(_get_read_agent(), reg_handler_cleaning)

    else:  # python_self_imp
        time_write = await python_self_write_blocks(block_size, view, blocks_indices_write, file_names)
        verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)
        
        # reading 100GB of other blocks to clean the cache
        await python_self_read_blocks(block_size_cleaning, view_cleaning, indices_cleaning, file_names_cleaning)

        time_read = await python_self_read_blocks(block_size, view, blocks_indices_read, file_names)
        verify_op(block_size, blocks_indices_read, view, file_names, "Reading", verify)
        
        # reading 100GB of other blocks to clean the cache
        await python_self_read_blocks(block_size_cleaning, view_cleaning, indices_cleaning, file_names_cleaning)
    
    return time_write, time_read


def allocate_buffers(buffer_size, verify=False):
    """Allocate main buffer and cleaning buffer for benchmarks."""
    num_elements = buffer_size // 2
    print("Allocating buffers...")
    
    if verify:
        buffer = torch.randn(num_elements, dtype=torch.float16, device='cpu', pin_memory=True)
    else:
        buffer = torch.zeros(num_elements, dtype=torch.float16, device='cpu', pin_memory=True)
    
    buffer_cleaning = torch.zeros(50*1024*1024*1024, dtype=torch.float16, device='cpu', pin_memory=True)
    
    print("Buffers allocated\n")
    
    view = memoryview(buffer.numpy()).cast('B')
    view_cleaning = memoryview(buffer_cleaning.numpy()).cast('B')
    
    return buffer, buffer_cleaning, view, view_cleaning


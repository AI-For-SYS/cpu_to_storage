from typing import Any
import torch
from nixl._api import nixl_agent, nixl_agent_config
import os
import numpy as np
import time

# Storage path configuration - can be overridden by STORAGE_PATH environment variable
STORAGE_PATH = os.environ.get('STORAGE_PATH', '/dev/shm')


# Global persistent agents - created once and reused
_write_agent = None
_read_agent = None

def _get_write_agent():
    global _write_agent
    
    # Recreate agent if num_threads changed
    if _write_agent is None:
        _write_agent = None  # Release old agent
        conf = nixl_agent_config(
            enable_prog_thread=True,
            backends=["POSIX"]
        )
        _write_agent = nixl_agent(agent_name="NIXL_Writer", nixl_conf=conf, instantiate_all=False)
    return _write_agent


def _get_read_agent():
    global _read_agent
    
    # Recreate agent if num_threads changed
    if _read_agent is None:
        _read_agent = None  # Release old agent
        conf = nixl_agent_config(
            enable_prog_thread=True,
            backends=["POSIX"]
        )
        _read_agent = nixl_agent(agent_name="NIXL_Reader", nixl_conf=conf, instantiate_all=False)
    return _read_agent

def nixl_register_buffer(agent: nixl_agent, buffer: torch.Tensor):
    """
    Registers the entire CPU buffer with the NIXL agent once.
    Call this during your application's initialization phase.
    """
    # NIXL requires the tensor to be contiguous for native extraction
    if not buffer.is_contiguous():
        buffer = buffer.contiguous()
    # NIXL automatically extracts base_addr, region_len, and gpu_id from the Tensor
    reg_dl = agent.get_reg_descs(buffer)
    # Register the memory region
    reg_handle = agent.register_memory(reg_dl)
    
    # Return the handle so it can be kept alive and deregistered at shutdown
    return reg_handle

def nixl_unregister_buffer(agent: nixl_agent, reg_handle):
    """
    Unregisters the memory region associated with the given handle.
    Call this during your application's shutdown phase.
    """
    agent.deregister_memory(reg_handle)

def nixl_write_blocks(block_size, buffer, blocks_indices, file_names):
    """
    Writes discrete blocks from a CPU buffer to files as fast as possible.
    Uses a persistent agent for better performance across multiple calls.
    """
    start = time.perf_counter()
    
    # Get the persistent agent
    agent = _get_write_agent()
    buffer_addr = buffer.data_ptr()
    
    # 2. Build Xfer Descriptor Lists
    local_descs_data = [] # CPU side
    remote_descs_data = [] # File side
    open_fds = []
    temp_files = []  # Track (temp_file, final_file) pairs for renaming

    try:
        for i, (idx, fname) in enumerate(zip(blocks_indices, file_names)):
            # Write to temporary file first
            temp_fname = f"{STORAGE_PATH}/temp_block_{idx}.bin"
            temp_files.append((temp_fname, fname))
            fd = os.open(temp_fname, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            open_fds.append(fd)

            # Local: DRAM block. Format: (addr, len, devID, metadata_handle)
            block_addr = buffer_addr + (idx * block_size)
            local_descs_data.append((block_addr, block_size, 0))

            # Remote: File target. Format: (offset, len, fd, metadata)
            remote_descs_data.append((0, block_size, fd, ""))

        # Convert raw lists to NIXL-internal descriptor lists
        local_xfer_dl = agent.get_xfer_descs(local_descs_data, mem_type="DRAM")

        file_handle = agent.register_memory(remote_descs_data, mem_type="FILE", backends=["POSIX"])
        file_desc = file_handle.trim()
    
        # 3. Initialize and Start Transfer
        xfer_handle = agent.initialize_xfer(
            operation="WRITE",
            local_descs=local_xfer_dl,
            remote_descs=file_desc,
            remote_agent="NIXL_Writer",
            backends=["POSIX"]
        )
        # Start the asynchronous data movement
        agent.transfer(xfer_handle)

        # 4. Busy Polling (Fastest completion detection)
        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                raise RuntimeError("NIXL Transfer Failed")

        # 5. Close file descriptors before renaming
        for fd in open_fds:
            os.close(fd)
        open_fds = []  # Clear the list so finally block doesn't try to close again

        # 6. Rename temporary files to final names atomically
        for temp_fname, final_fname in temp_files:
            os.rename(temp_fname, final_fname)
    except:
        for fd in open_fds:
            os.close(fd)
    finally:
        end = time.perf_counter()

        if 'xfer_handle' in locals():
            agent.release_xfer_handle(xfer_handle)
        if 'file_handle' in locals():
            agent.deregister_memory(file_handle)

    return (end-start)


def nixl_read_blocks(block_size, buffer, block_indices: list, file_names: list):
    """
    Reads discrete blocks from files into a CPU buffer as fast as possible.
    Uses a persistent agent for better performance across multiple calls.
    """
    start = time.perf_counter()
    
    # Get the persistent agent
    agent = _get_read_agent()

    # 1. Memory Registration
    buffer_addr = buffer.data_ptr()

    # 2. Build Xfer Descriptor Lists
    local_descs_data = []
    remote_descs_data = []
    open_fds = []

    try:
        for idx, fname in zip(block_indices, file_names):
            fd = os.open(fname, os.O_RDONLY )
            open_fds.append(fd)

            # Local: Destination block in CPU buffer
            block_addr = buffer_addr + (idx * block_size)
            local_descs_data.append((block_addr, block_size, 0))

            # Remote: Source file
            remote_descs_data.append((0, block_size, fd, ""))

        # Convert raw lists to NIXL-internal descriptor lists
        local_xfer_dl = agent.get_xfer_descs(local_descs_data, mem_type="DRAM")

        file_handle = agent.register_memory(remote_descs_data, mem_type="FILE", backends=["POSIX"])
        file_desc = file_handle.trim()
        
        # 3. Initialize and Start Transfer
        xfer_handle = agent.initialize_xfer(
            operation="READ",
            local_descs=local_xfer_dl,
            remote_descs=file_desc,
            remote_agent="NIXL_Reader",
            backends=["POSIX"]
        )
        
        agent.transfer(xfer_handle)

        # 4. Busy Polling
        while True:
            state = agent.check_xfer_state(xfer_handle)
            if state == "DONE":
                break
            elif state == "ERR":
                raise RuntimeError("NIXL Transfer Failed")

        for fd in open_fds:
            os.close(fd)
    except:
        for fd in open_fds:
            os.close(fd)
    finally:
        end = time.perf_counter()
        if 'xfer_handle' in locals():
            agent.release_xfer_handle(xfer_handle)
        if 'file_handle' in locals():
            agent.deregister_memory(file_handle)
    
    return (end-start)


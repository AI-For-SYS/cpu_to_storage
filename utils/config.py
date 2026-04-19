"""Configuration constants and environment variables for benchmarking."""

import os
import torch

# Storage path configuration - can be overridden by STORAGE_PATH environment variable
STORAGE_PATH = os.environ.get('STORAGE_PATH', '/dev/shm')

# Cluster name for tracking benchmark results
CLUSTER = os.environ.get('CLUSTER_NAME', 'unknown')

# List of Python-based backends (vs C++ backends)
PYTHON_BACKENDS = ["python_aiofiles", "python_self_imp", "nixl"]

# Use pinned memory only if CUDA is available (required by torch.Tensor pin_memory=True)
PIN_MEMORY = torch.cuda.is_available()
if not PIN_MEMORY:
    print("[WARN] CUDA not available — buffers will use pageable memory (pin_memory=False)")

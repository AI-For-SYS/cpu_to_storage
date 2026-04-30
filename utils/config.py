"""Configuration constants and environment variables for benchmarking."""

import os
import torch
from dotenv import load_dotenv

# Load .env from project root (or cwd walking up). Does not override existing env vars,
# so shell exports / run_benchmark_on_lsf.sh's `source .env` / pod env still win.
load_dotenv()

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

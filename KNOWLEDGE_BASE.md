# CPU-to-Storage I/O Benchmark - Knowledge Base

## Project Purpose

A high-performance benchmarking suite that measures and compares file I/O throughput (read/write) across multiple implementation backends. Designed for evaluating parallel I/O strategies relevant to ML model checkpointing and large-scale data pipelines. The goal is to find the fastest way to move data between CPU memory and storage.

---

## Architecture Overview

```
compare_file_operations.py    Entry point. Parses CLI args, dispatches to benchmark mode.
        |
        v
utils/benchmark_core.py       Core engine. Allocates buffers, runs iterations, dispatches to backends.
        |
        +---> backends/cpp_backend.py          -> cpp_ext (C++ PyBind11 module)
        +---> backends/python_self_backend.py  -> asyncio + os.readv/os.write
        +---> backends/aiofiles_backend.py     -> aiofiles library
        +---> backends/nixl_backend.py         -> NVIDIA NIXL library
        |
        v
utils/file_utils.py            File naming, verification, cleanup
utils/checkpoints_utils.py     Incremental save/resume of results
utils/config.py                Environment variables (STORAGE_PATH, CLUSTER_NAME)
plotter.py                     Matplotlib visualization of results
```

---

## Directory Structure

```
cpu_to_storage/
├── compare_file_operations.py     # Main CLI entry point
├── setup.py                       # Builds C++ extension via PyTorch CppExtension
├── plotter.py                     # Result visualization (multiple plot types)
├── copy_to_pod.sh                 # kubectl copy helper for K8s deployment
├── README.md                      # User-facing documentation
├── CLAUDE.md                      # Claude Code session instructions
├── KNOWLEDGE_BASE.md              # This file
│
├── backends/
│   ├── cpp_backend.py             # Thin async wrapper around cpp_ext module
│   ├── python_self_backend.py     # Pure Python: asyncio + ThreadPoolExecutor + os.readv/write
│   ├── aiofiles_backend.py        # aiofiles library-based async I/O
│   ├── nixl_backend.py            # NVIDIA NIXL with POSIX backend, memory registration
│   └── benchmark_cpp_utils/
│       ├── cpp_utils.cpp          # Core C++ I/O: pread/pwrite + thread pool dispatch
│       ├── simple_thread_pool.cpp # Thread pool implementation (condition_variable based)
│       └── simple_thread_pool.hpp # Thread pool header with template enqueue()
│
├── utils/
│   ├── benchmark_core.py          # Buffer allocation, benchmark iteration logic, executor setup
│   ├── config.py                  # STORAGE_PATH, CLUSTER_NAME, PYTHON_BACKENDS list
│   ├── file_utils.py              # generate_dest_file_names, verify_op, clean_files, write_blocks
│   └── checkpoints_utils.py       # save_incremental_results, load_existing_results, resume logic
│
├── deployment/
│   ├── io_bench_pod.yaml          # K8s pod: 450Gi RAM, 12-128 CPU, 1 GPU, pytorch image
│   └── io_bench_pvc.yaml          # K8s PVC: 300Gi ReadWriteMany
│
├── results/                       # Benchmark output JSON files
└── plots/                         # Generated matplotlib plots
```

---

## Benchmark Modes

### 1. `blocks` mode
- Writes/reads a **fixed number of blocks** for each block size.
- Total data varies with block size (more data for larger blocks).
- Output: `results/blocks_{num_blocks}_{test_name}_{dest}_{backend}.json`

### 2. `data` mode
- Writes/reads a **fixed total data size** (e.g., 100GB).
- Number of blocks varies inversely with block size.
- Output: `results/data_{test_name}_{total_gb}gb_{dest}_{backend}.json`

### 3. `concurrent` mode
- Runs read and write **simultaneously** using `asyncio.gather()`.
- Splits total data: half for reads, half for writes.
- Measures individual and combined throughput.
- Output: `results/concurrent_{test_name}_{total_gb}gb_{dest}_{backend}.json`

---

## Backend Details

### C++ Backend (`cpp_backend.py` + `benchmark_cpp_utils/`)

- **Platform**: Linux only (uses `pread`, `pwrite`, `fcntl.h`, `unistd.h`)
- **Build**: `python setup.py build_ext --inplace` (requires PyTorch, C++17, GCC/Clang)
- **Compiler flags**: `-std=c++17 -O3 -march=native -fPIC`
- **Thread pool**: Custom `SimpleThreadPool` using `std::condition_variable` + task queue
- **Read strategy**: Workers open their own FDs + `pread()` in parallel (no directory contention for existing files)
- **Write strategy**: Main thread pre-opens all temp files sequentially (serializes `O_CREAT` to avoid directory inode contention), then workers do `pwrite()` + `close()` + `rename()` in parallel
- **GIL**: Released via `py::gil_scoped_release` during all I/O operations
- **Thread count**: Configurable at runtime via `set_thread_count()`, recreates the pool
- **Exposed functions**: `cpp_read_blocks()`, `cpp_write_blocks()`, `set_thread_count()`, `get_io_thread_count()`

### Python Self-Implementation (`python_self_backend.py`)

- **Platform**: Linux only (uses `os.readv()` which is POSIX-only)
- **Concurrency**: `asyncio.get_running_loop().run_in_executor(None, ...)` with `ThreadPoolExecutor`
- **Read**: Pre-opens all FDs, dispatches `os.readv()` calls via executor, closes FDs after gather
- **Write**: Pre-opens temp files with `os.open(O_CREAT|O_WRONLY|O_TRUNC)`, workers do `os.write()` loop + `os.close()` + `os.replace()`
- **Thread count**: Controlled by the executor set in `benchmark_core.setup_executor()`

### aiofiles Backend (`aiofiles_backend.py`)

- **Platform**: Cross-platform (aiofiles is pure Python)
- **Concurrency**: Native `asyncio` coroutines with `asyncio.gather()`
- **Read**: `aiofiles.open(rb)` + `f.readinto(memoryview_slice)`
- **Write**: `aiofiles.open(wb)` + `f.write()`, then `aiofiles.os.replace()` for atomic rename
- **Thread count**: Controlled by the executor set in `benchmark_core.setup_executor()`

### NIXL Backend (`nixl_backend.py`)

- **Platform**: Linux with NVIDIA NIXL library
- **Architecture**: Two persistent global agents (NIXL_Writer, NIXL_Reader) with POSIX backend
- **Memory model**: Explicit `register_memory()` / `deregister_memory()` calls around each benchmark iteration
- **Transfer**: Builds descriptor lists (local DRAM + remote FILE), calls `initialize_xfer()` + `transfer()`, then busy-polls `check_xfer_state()` until DONE
- **Write**: Opens temp files, transfers, closes FDs, renames atomically
- **Read**: Opens source files, transfers into registered buffer, closes FDs
- **Cleanup**: `release_xfer_handle()` + `deregister_memory()` in finally blocks

---

## Core Engine Details (`utils/benchmark_core.py`)

### Buffer Allocation (`allocate_buffers()`)
- **Main buffer**: `torch.zeros/randn(N, dtype=float16, pin_memory=True)` where N = buffer_size_bytes / 2
- **Cleaning buffer**: Fixed 50GB (`torch.zeros(50*1024*1024*1024, dtype=float16, pin_memory=True)`)
- Both buffers are exposed as `memoryview(...numpy()).cast('B')` for byte-level access
- **Limitation**: The 50GB cleaning buffer is hardcoded, making local testing with limited RAM difficult

### Cache Cleaning Strategy
- Between write and read phases, reads 100GB of unrelated data (3200 files x 32MB) from `/dev/shm`
- Uses `python_self_read_blocks` for Python backends and `cpp_read_blocks` for C++ backend
- Purpose: Evict the written data from OS page cache to measure true storage throughput

### Benchmark Iteration Flow (`run_benchmark_iteration()`)
1. Write blocks to storage
2. Optionally verify written data
3. Clean cache (read 100GB unrelated data)
4. Read blocks back from storage
5. Optionally verify read data
6. Clean cache again
7. Return (write_time, read_time)

### Backend Dispatch
- Uses `if/elif` chains keyed on implementation name string
- Each branch has ~15 lines of similar but not identical logic
- NIXL branch is more complex due to memory registration/deregistration
- **Design limitation**: No backend interface/base class; adding a new backend requires modifying `run_benchmark_iteration()`, `run_concurrent_benchmark_iteration()`, `setup_executor()`, `shutdown_executor()`, plus CLI argument choices

---

## Configuration System (`utils/config.py`)

| Variable | Env Var | Default | Purpose |
|----------|---------|---------|---------|
| `STORAGE_PATH` | `STORAGE_PATH` | `/dev/shm` | Where benchmark files are written |
| `CLUSTER` | `CLUSTER_NAME` | `unknown` | Identifier stored in results metadata |
| `PYTHON_BACKENDS` | - | `["python_aiofiles", "python_self_imp", "nixl"]` | List of non-C++ backends (used for executor management) |

---

## CLI Arguments (`compare_file_operations.py`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `blocks` | `blocks`, `data`, or `concurrent` |
| `--backend` | `python_self_imp` | `cpp`, `python_aiofiles`, `python_self_imp`, `nixl` |
| `--buffer-size` | `100` | Buffer size in GB |
| `--iterations` | `5` | Iterations per (thread_count, block_size) combination |
| `--num-blocks` | `1000` | Blocks to transfer (blocks mode only) |
| `--total-gb` | `100` | Total data in GB (data/concurrent modes) |
| `--block-sizes` | `2 4 8 16 32 64` | Block sizes in MB to test |
| `--test-name` | `""` | Prefix for result filename |
| `--verify` | `False` | Enable data integrity verification |

Hardcoded: `threads_counts = [16, 32, 64]`

---

## Checkpoint/Resume System (`utils/checkpoints_utils.py`)

- Results saved incrementally after each (thread_count, block_size) combination
- Atomic writes via temp file + `os.replace()`
- On restart with same config: detects completed (thread_count, block_size) pairs and skips them
- Config matching: checks `buffer_size`, `num_iterations`, `block_sizes_mb`, `thread_counts`, `num_blocks`, `total_data_size_gb`, `implementation`

---

## Plotter (`plotter.py`)

Five plot functions, dispatched by mode via `main()`:

| Function | Mode | Layout | Description |
|----------|------|--------|-------------|
| `plot_results_threads_comparison()` | (legacy) | 1x2 | Throughput vs thread count for a single block size |
| `plot_throughput_tables()` | (legacy) | 2x1 tables | Formatted tables + heatmaps |
| `plot_block_size_heatmaps()` | (legacy) | 1x2 heatmaps | Block size vs threads heatmap |
| `plot_blocks_throughput_by_threads()` | `blocks` | 2x3 (ops x threads) | Throughput vs block size, multi-backend comparison |
| `plot_total_data_throughput_by_threads()` | `data` | 2x3 (ops x threads) | Same but for fixed total data; secondary axis shows # blocks |
| `plot_concurrent_throughput_by_threads()` | `concurrent` | 3x3 (metrics x threads) | Write/Read/Combined throughput under concurrent load |

Usage: `python plotter.py <mode> <file1> [file2] ...`

---

## Kubernetes Deployment

- **Pod** (`io_bench_pod.yaml`): 450Gi RAM, 12-128 CPU, 1 GPU, pytorch:latest image, installs build-essential + ninja + aiofiles + nixl
- **PVC** (`io_bench_pvc.yaml`): 300Gi ReadWriteMany
- **Volumes**: `/dev/shm` (450Gi tmpfs emptyDir) + `/mnt/persistent-storage` (PVC)
- **Namespace**: `rotem`
- **Deploy helper**: `copy_to_pod.sh` (copies source to pod, `--get-results` copies results back)

---

## Key Design Decisions & Patterns

1. **Atomic writes everywhere**: All backends write to temp file then rename. Prevents partial/corrupt files.
2. **Serialized file creation in C++**: Main thread opens all temp files sequentially before dispatching workers. Avoids directory inode contention from concurrent `O_CREAT`.
3. **Parallel FD opens for reads in C++**: Since files already exist, no `O_CREAT` contention, so opening in workers is faster.
4. **Cache invalidation between phases**: Reads 100GB unrelated data to flush OS page cache. Critical for measuring true I/O throughput, not cached reads.
5. **Pinned memory**: `pin_memory=True` prevents OS from swapping out the buffer during benchmarks.
6. **float16 buffers**: Halves memory usage vs float32, allowing larger benchmarks with same RAM.

---

## Known Limitations & Improvement Opportunities

### Platform
- C++ backend and python_self_backend are **Linux-only** (POSIX APIs: `pread`, `pwrite`, `os.readv`)
- Would need `ReadFile`/`WriteFile` (Win32) or cross-platform wrappers for Windows

### Architecture
- **No backend interface**: Adding a new backend requires editing 5+ locations across multiple files. An abstract base class or protocol would make this a single-file operation.
- **Hardcoded thread counts**: `[16, 32, 64]` is hardcoded in `compare_file_operations.py`, not a CLI arg.
- **Hardcoded cleaning buffer**: 50GB fixed in `allocate_buffers()`, not configurable.
- **Duplicated STORAGE_PATH**: Read from env in `config.py`, `python_self_backend.py`, `aiofiles_backend.py`, and `nixl_backend.py` independently (should use single source from config).

### Performance Optimization Opportunities
- C++ thread pool: consider io_uring for Linux 5.1+ (eliminates syscall overhead)
- C++ writes: experiment with `O_DIRECT` to bypass page cache entirely
- Python backends: could benefit from `os.preadv`/`os.pwritev` for scatter-gather I/O
- NIXL: currently busy-polls; could experiment with event-driven completion
- Buffer allocation: could use `mmap` with `MAP_HUGETLB` for huge pages
- File creation contention: could pre-create empty files or use per-thread subdirectories

---

## Result Format

```json
{
  "config": {
    "cluster": "...",
    "buffer_size": 107374182400,
    "num_iterations": 5,
    "threads_counts": [16, 32, 64],
    "block_sizes_mb": [2, 4, 8, 16, 32, 64],
    "file_system": "/dev/shm (tmpfs)",
    "implementation": "cpp",
    "total_data_size_gb": 100
  },
  "write": {
    "16": { "2": 0.045, "4": 0.048, ... },
    "32": { "2": 0.042, ... },
    "64": { ... }
  },
  "read": {
    "16": { "2": 0.052, ... },
    ...
  },
  "concurrent": {  // only in concurrent mode
    "16": { "2": 0.090, ... },
    ...
  }
}
```

Values are **mean elapsed time in seconds** across all iterations for that (thread_count, block_size_mb) combination.

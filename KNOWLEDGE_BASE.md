# CPU-to-Storage I/O Benchmark - Knowledge Base

## Project Purpose

A high-performance benchmarking suite for **KV cache offload**: moving KV cache layers between CPU memory (pinned `torch.Tensor`, float16) and storage, and reloading them back. This is not a general-purpose I/O benchmark — it specifically models the ML inference pattern where:

- **Write (offload)**: KV cache layers are evicted from CPU memory to storage in parallel (multiple layers/shards concurrently). Data originates from live tensors, not from prior disk reads.
- **Read (reload)**: KV cache layers are loaded back from storage into CPU tensors. Data is read once per reload — no random re-reads, no caching benefit.
- **Parallelism**: Multiple threads handle different files (layers/shards) concurrently. No parameter or I/O strategy may restrict parallel access.
- **Durability**: Optional. If the system crashes, KV cache can be recomputed from the model. Sync-to-disk is only needed for durable checkpointing, not for temporary offload.

The suite compares file I/O throughput (read/write) across multiple implementation backends to find the fastest offload/reload strategy. Also relevant to ML model checkpointing and large-scale data pipelines.

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
├── setup.py                       # Builds C++ extension (cpp_ext) via PyTorch CppExtension
├── setup_threaded_tunable.py      # Builds threaded_tunable extension (threaded_tunable_ext)
├── setup_iouring.py               # Builds io_uring extension (iouring_ext), links liburing
├── plotter.py                     # Result visualization (multiple plot types)
├── copy_to_pod.sh                 # kubectl copy helper for K8s deployment
├── setup_env.sh                   # Idempotent venv + deps + extensions setup (safe to re-run)
├── run_benchmark_on_lsf.sh        # LSF job submission wrapper
├── benchmark_job.sh               # Benchmark job script (called by run_benchmark_on_lsf.sh)
├── .gitattributes                 # Forces LF line endings for all source files
├── .gitignore                     # Excludes sftp config, build artifacts
├── README.md                      # User-facing documentation
├── CLAUDE.md                      # Claude Code session instructions
├── KNOWLEDGE_BASE.md              # This file
│
├── backends/
│   ├── cpp_backend.py             # Thin async wrapper around cpp_ext module
│   ├── threaded_tunable_backend.py # Tunable wrapper + ThreadedTunableConfig dataclass + save/load
│   ├── iouring_backend.py         # Thin async wrapper around iouring_ext module
│   ├── python_self_backend.py     # Pure Python: asyncio + ThreadPoolExecutor + os.readv/write
│   ├── aiofiles_backend.py        # aiofiles library-based async I/O
│   ├── nixl_backend.py            # NVIDIA NIXL with POSIX backend, memory registration
│   ├── benchmark_cpp_utils/
│   │   ├── cpp_utils.cpp              # Core C++ I/O: pread/pwrite + thread pool dispatch
│   │   ├── threaded_tunable_utils.cpp # Tunable C++ I/O: same core + IOConfig parameters
│   │   ├── simple_thread_pool.cpp     # Thread pool implementation (shared by cpp and tunable)
│   │   └── simple_thread_pool.hpp     # Thread pool header with template enqueue()
│   └── benchmark_iouring_utils/
│       └── iouring_utils.cpp      # Core io_uring I/O: batched SQ/CQ submission
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
├── tests/
│   ├── test_iouring.py            # Quick smoke test for iouring_ext
│   └── test_threaded_tunable.py   # Smoke test: build, config roundtrip, each knob, baseline comparison
├── docs/
│   ├── optuna_autotuning_plan.md  # Design plan for Optuna + threaded_tunable
│   └── ...                        # Presentation files
├── results/                       # Benchmark output JSON files + tunable config JSONs
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

### Threaded Tunable Backend (`threaded_tunable_backend.py` + `benchmark_cpp_utils/`)

- **Platform**: Linux only (same POSIX APIs as cpp backend)
- **Build**: `python setup_threaded_tunable.py build_ext --inplace`
- **Purpose**: Same pread/pwrite thread pool as baseline `cpp`, with tunable I/O parameters for Optuna auto-optimization
- **Core I/O**: Identical to `cpp` — pread/pwrite in thread pool, sequential temp-file creation for writes, parallel opens for reads, atomic rename
- **Configuration**: `IOConfig` struct with 9 tunable parameters, `configure_all(dict)` bulk setter, `get_config()` getter
- **Tunable parameters**: `O_NOATIME`, `O_DIRECT`, `posix_fadvise` hints, `io_chunk_size`, `prefetch_depth` (readahead), `fallocate` pre-allocation, `sync_strategy` (none/fdatasync/sync_file_range), `cpu_affinity`
- **Python config**: `ThreadedTunableConfig` dataclass with `FadviseHint`/`SyncStrategy` enums, `save()`/`load()` for JSON persistence to `results/`
- **Config loading**: `compare_file_operations.py --tunable-config results/best_write_config.json` loads saved params before benchmarking
- **Default behavior**: With all parameters at defaults, behaves identically to baseline `cpp`
- **GIL**: Released via `py::gil_scoped_release` during all I/O operations
- **Exposed functions**: `threaded_tunable_read_blocks()`, `threaded_tunable_write_blocks()`, `configure_all()`, `get_config()`, individual setters
- **Smoke test**: `python -m tests.test_threaded_tunable` — verifies build, config roundtrip, data integrity, each parameter individually
- **Design doc**: `docs/optuna_autotuning_plan.md`

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

### io_uring Backend (`iouring_backend.py` + `benchmark_iouring_utils/`) — IN PROGRESS

- **Platform**: Linux only, requires kernel 5.6+ with io_uring enabled (`kernel.io_uring_disabled = 0`)
- **Build**: `python setup_iouring.py build_ext --inplace` (requires liburing at `$HOME/.local` or system-wide)
- **Compiler flags**: `-std=c++17 -O3 -march=native -fPIC -luring`
- **Concurrency model**: No thread pool — uses io_uring's submission/completion queue rings. Kernel handles I/O parallelism.
- **Read strategy**: Opens FDs sequentially, batches pread SQEs into the submission queue, reaps completions from CQ
- **Write strategy**: Serialized temp file creation (same O_CREAT pattern as cpp_ext), batches pwrite SQEs, then renames atomically
- **Batch overflow**: When ops > queue_depth, submits in batches with interleaved completion reaping
- **GIL**: Released via `py::gil_scoped_release` during all I/O operations
- **Tunable parameters**: `queue_depth` (ring size), `batch_size` (SQEs per submit). Future: SQPOLL, O_DIRECT, registered files/buffers.
- **Exposed functions**: `iouring_read_blocks()`, `iouring_write_blocks()`, `set_queue_depth()`, `set_batch_size()`
- **Availability detection**: Runtime probe (`iouring_probe()`) at import time detects both missing extension (ImportError) and kernel-blocked io_uring (RuntimeError). Sets `IOURING_AVAILABLE = False` with warning.
- **Status**: Code written, builds, and probe works. Not yet integrated into `benchmark_core.py` or `compare_file_operations.py` (step 5 in plan). Blocked on cluster io_uring access (`kernel.io_uring_disabled=2` on current LSF cluster). Needs a machine with sudo + kernel 5.6+ to enable io_uring.
- **Design**: Modular strategy functions (`open_files_for_read/write`, `submit_io_ops`, `reap_completions`, `rename_temp_files`) structured for future Bayesian auto-tuning and OpenEvolve code evolution.
- **Full plan**: See `docs/iouring_implementation_plan.md`

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
| `--backend` | `python_self_imp` | `cpp`, `threaded_tunable`, `python_aiofiles`, `python_self_imp`, `nixl` |
| `--tunable-config` | `None` | Path to JSON config for `threaded_tunable` backend |
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

## LSF Remote Infrastructure

### Setup
- `setup_env.sh` — idempotent setup: venv creation (skipped if exists), dependency install, C++ extension build, threaded_tunable extension build, liburing build (skipped if exists), io_uring extension build. Safe to re-run.
- Venv at `.venv/`, activate with `source .venv/bin/activate`
- `nixl` backend is optional — import gracefully falls back with a warning if not installed
- `iouring` backend is optional — requires liburing and kernel with `io_uring_disabled=0`. Probes at import time and falls back with a warning if blocked.
- Requires: Python 3.9+, GCC 11+, PyTorch, ninja
- liburing built from source into `$HOME/.local` (shared across LSF nodes via shared home filesystem)

### Running Benchmarks
1. Create `.env` file in project root with your paths (gitignored):
   ```
   PROJ_DIR=/path/to/cpu_to_storage
   STORAGE_PATH=/path/to/benchmark_tmp
   CLUSTER_NAME=your_cluster
   ```
2. Create logs directory: `mkdir -p logs`
3. Run:
   - `./run_benchmark_on_lsf.sh short` — sanity check: 2 block sizes (4,8 MB), 3 thread counts, 1 iteration, 10 blocks, cpp backend. ~2 min.
   - `./run_benchmark_on_lsf.sh full` — full benchmark: 6 block sizes (2-64 MB), 3 thread counts, 5 iterations, 1000 blocks, cpp backend. ~15-20 min.
   - `./run_benchmark_on_lsf.sh short threaded_tunable` — run single backend.
   - `./run_benchmark_on_lsf.sh compare-short` — run all backends on same node for fair comparison (short).
   - `./run_benchmark_on_lsf.sh compare-short "cpp threaded_tunable"` — compare selected backends only.
   - `./run_benchmark_on_lsf.sh compare-full "cpp threaded_tunable" results/best_write_config.json` — full comparison with tunable config.
4. Monitor: `bjobs`, `bpeek <job_id>`
5. Results: `logs/benchmark_<job_id>.out` and `results/` directory

### LSF Quick Reference
| Command | Purpose |
|---------|---------|
| `bjobs` | Check job status |
| `bjobs -l <id>` | Detailed status / pending reason |
| `bpeek <id>` | Live output of running job |
| `bkill <id>` | Kill job (`bkill 0` kills all) |

### Notes
- `rusage[mem]` may cause scheduling failures on clusters with non-standard memory reporting — omit if jobs stay in PEND
- The 50GB hardcoded cleaning buffer in `allocate_buffers()` requires machines with sufficient RAM (~160GB+ for a full 100GB buffer run)

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

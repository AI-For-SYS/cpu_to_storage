# Bayesian Auto-Tuning with Optuna — Design Plan

## Goal

Use Optuna to find the optimal combination of C++ backend parameters that maximizes I/O throughput (GB/s) for read and write operations on a given storage system.

The original `cpp` backend (`cpp_utils.cpp`) remains untouched as the baseline. A new `threaded_tunable` backend is created with all the tunable parameters, so results can be compared directly: baseline `cpp` vs. Optuna-tuned `threaded_tunable`. The naming convention supports future tunable backends (e.g., `iouring_tunable`).

---

## 1. New Backend: `threaded_tunable`

### 1.1 File Layout

```
cpu_to_storage/
├── compare_file_operations.py        # UNTOUCHED — existing benchmark entry point
├── optuna_tuner.py                   # NEW — Optuna auto-tuning entry point
├── setup.py                          # UNTOUCHED — builds cpp_ext
├── setup_threaded_tunable.py         # NEW — builds threaded_tunable_ext
├── scripts/
│   ├── run_benchmark_on_lsf.sh       # EXISTING — benchmark LSF wrapper (moved here)
│   ├── benchmark_job.sh              # EXISTING — benchmark job script (moved here)
│   ├── run_optuna_on_lsf.sh          # NEW — LSF submission wrapper for Optuna
│   └── optuna_job.sh                 # NEW — Optuna job script (venv, run tuner)
│
├── tests/
│   ├── test_iouring.py               # EXISTING
│   └── test_threaded_tunable.py      # NEW — smoke test: build, defaults, tunable parameters, compare vs cpp
│
├── backends/
│   ├── benchmark_cpp_utils/
│   │   ├── cpp_utils.cpp             # UNTOUCHED — baseline backend
│   │   ├── threaded_tunable_utils.cpp # NEW — thread-based tunable backend
│   │   ├── simple_thread_pool.cpp    # UNTOUCHED — shared by both backends
│   │   └── simple_thread_pool.hpp    # UNTOUCHED — shared
│   ├── cpp_backend.py                # UNTOUCHED — baseline wrapper
│   └── threaded_tunable_backend.py   # NEW — Python wrapper + ThreadedTunableConfig
```

- `setup_threaded_tunable.py` — builds `threaded_tunable_ext` module, links against same PyTorch/pybind11
- The tunable backend reuses `simple_thread_pool.hpp/.cpp` (shared source, compiled into both extensions)

### 1.2 How It Differs from Baseline `cpp`

The core I/O pattern is identical: pread/pwrite in a thread pool, sequential temp-file creation for writes, parallel opens for reads, atomic rename. The differences are:

| Aspect | Baseline `cpp` | `threaded_tunable` |
|--------|----------------|---------------|
| `open()` flags | `O_RDONLY` / `O_WRONLY\|O_CREAT\|O_TRUNC` | + optional `O_NOATIME`, `O_DIRECT` |
| After open | nothing | optional `posix_fadvise()`, optional `fallocate()` |
| pread/pwrite loop | reads/writes full remaining block per syscall | configurable chunk size per syscall |
| Before I/O | nothing | optional `readahead()` on upcoming files |
| After write | close immediately | optional `fdatasync()` / `sync_file_range()` before close |
| Thread pool | fixed, recreated only on count change | + optional CPU affinity per worker |
| Configuration | `set_thread_count()` only | `ThreadedTunableConfig` dataclass → `configure_all(dict)` at PyBind11 boundary |
| State management | global thread count | all params reset-then-apply on each `configure_all()` call |

With all tunable parameters at defaults, `threaded_tunable` behaves identically to baseline `cpp`.

---

## 2. Tunable Parameters

### 2.0 KV Cache Offload Compatibility

All parameters must be compatible with the real use case: parallel offload/reload of KV cache layers between CPU memory and storage. This means:

- **No parameter may restrict parallel access** — multiple threads read/write different files concurrently. All parameters below operate on per-FD or per-process scope.
- **Data is used once** — KV cache is offloaded to storage, then reloaded into a tensor later. No random re-reads of the same file. This favors `NOREUSE`/`DONTNEED` fadvise hints and makes `O_DIRECT` viable (no benefit from page cache).
- **Durability is optional** — if the system crashes, KV cache can be recomputed. `sync_strategy=none` is expected to win for offload workloads. `fdatasync` only matters for durable checkpointing.
- **CPU affinity must not conflict with ML inference threads** — I/O worker threads should not be pinned to cores that are running the model's attention/prefill computation. In production, this would be coordinated with the ML framework's thread pool.

### 2.1 Full Search Space

Optuna parameter types:
- **int**: continuous integer range — Optuna can try *any* integer in [low, high]. With `log=True`, samples on log scale (spreads wide ranges evenly).
- **categorical**: fixed list of discrete choices — no ordering assumed. Optuna models each choice independently.
- **bool**: categorical with two choices `[True, False]`.

```
thread_count:        int,         [4, 8, 16, 32, 64, 128]
block_size_mb:       categorical, [2, 4, 8, 16, 32, 64, 128]
o_noatime:           bool         (reads only)
o_direct:            bool         (requires aligned buffers)
fadvise_hint:        categorical, [NORMAL, SEQUENTIAL, RANDOM, WILLNEED, NOREUSE]
io_chunk_kb:         int,         [256, 512, 1024, 2048, 4096, 0]  (0 = full block)
prefetch_depth:      int,         [0, 1, 2, 4, 8]                  (reads only)
fallocate_prealloc:  bool         (writes only)
sync_strategy:       categorical, [none, fdatasync, sync_file_range]  (writes only)
cpu_affinity:        bool
```

10 parameters, ~30k+ possible combinations — exactly the kind of space where Bayesian optimization shines vs. grid search.

### 2.2 Parameter Details

#### File Descriptor Flags (set at `open()` time)

| Parameter | Values | Effect | Applies To |
|-----------|--------|--------|------------|
| **O_NOATIME** | on/off | Skips access-time metadata update on reads. Reduces metadata I/O. Requires file ownership or `CAP_DAC_READ_SEARCH`. | Reads |
| **O_DIRECT** | on/off | Bypasses page cache entirely. Data goes straight from userspace buffer to disk (or vice versa). Requires buffer aligned to filesystem block size (typically 512B or 4096B). Eliminates double-copy through page cache. Can be faster on NVMe/parallel FS; can be slower on tmpfs or when data fits in cache. | Both |

#### posix_fadvise Hints (set after `open()`, before I/O)

| Parameter | Values | Effect | Applies To |
|-----------|--------|--------|------------|
| **fadvise_read** | NORMAL / SEQUENTIAL / RANDOM / WILLNEED / DONTNEED / NOREUSE | Hints to kernel about access pattern. `SEQUENTIAL` doubles readahead window. `RANDOM` disables readahead. `WILLNEED` triggers async prefetch. `NOREUSE` hints data won't be reused (good for streaming). | Reads |
| **fadvise_write** | NORMAL / SEQUENTIAL / NOREUSE / DONTNEED | For writes: `SEQUENTIAL` helps writeback ordering. `DONTNEED` after write evicts from page cache (reduces memory pressure). | Writes |

#### I/O Chunking

| Parameter | Range | Effect | Applies To |
|-----------|-------|--------|------------|
| **io_chunk_size** | [256KB, 512KB, 1MB, 2MB, 4MB, block_size] | Size of individual `pread()`/`pwrite()` calls within a block. Currently each call tries to read/write the full remaining block. Smaller chunks can improve pipeline parallelism with the storage controller, allow the OS to interleave I/O from multiple threads, and reduce per-syscall latency. Too small = syscall overhead dominates. | Both |

#### Prefetching

| Parameter | Range | Effect | Applies To |
|-----------|-------|--------|------------|
| **prefetch_depth** | [0, 1, 2, 4, 8] files | Number of upcoming files to `readahead()` or `posix_fadvise(WILLNEED)` before starting their I/O. Worker threads prefetch N files ahead of the current one. 0 = disabled. | Reads |

#### Write-Specific

| Parameter | Values | Effect |
|-----------|--------|--------|
| **fallocate_prealloc** | on/off | Call `fallocate()` to pre-allocate file space before `pwrite()`. Avoids extent allocation during writes, reducing filesystem metadata overhead and fragmentation. Especially useful on ext4/XFS with large files. |
| **sync_strategy** | none / fdatasync / sync_file_range | Controls writeback to stable storage. `none` = OS decides when to flush (current behavior). `fdatasync` = flush data after write (no metadata). `sync_file_range` = async writeback of specific ranges — can overlap with next write. |

#### Thread/CPU Affinity

| Parameter | Values | Effect | Applies To |
|-----------|--------|--------|------------|
| **cpu_affinity** | on/off | Pin each worker thread to a specific CPU core via `sched_setaffinity()`. Reduces cache line bouncing and context switch overhead. Can hurt if storage interrupts are handled on pinned cores. | Both |

### 2.3 Parameters Intentionally Excluded

| Parameter | Why Excluded |
|-----------|-------------|
| **mmap I/O path** | Different I/O mechanism entirely, not a tuning knob — would be a separate backend |
| **O_SYNC / O_DSYNC** | Redundant with `sync_strategy`; these open flags are less flexible than post-write sync calls |
| **ioprio_set** | Requires root or `CAP_SYS_NICE`; unlikely to be available on shared LSF cluster |
| **Per-thread subdirectories** | Changes filesystem layout, not a transparent tuning parameter |
| **Buffer alignment size** | Determined by O_DIRECT requirement (query via `statvfs`), not a free parameter |

---

## 3. C++ Implementation: `threaded_tunable_utils.cpp`

### 3.0 API Compatibility with Baseline `cpp`

The tunable backend must match the baseline's function signatures exactly so it plugs into `benchmark_core.py` without special casing:

```cpp
// Same signature as cpp_utils.cpp — receives torch::Tensor, not raw pointers
bool threaded_tunable_read_blocks(torch::Tensor buffer, int64_t block_size,
                                   std::vector<int64_t> block_indices,
                                   std::vector<std::string> source_files);

bool threaded_tunable_write_blocks(torch::Tensor buffer, int64_t block_size,
                                    std::vector<int64_t> block_indices,
                                    std::vector<std::string> dest_files);
```

Key requirements carried over from baseline:
- Buffer must be CPU, contiguous `torch::Tensor` (float16, `pin_memory=True`)
- Bounds-check `block_indices[i] * block_size` against `buffer.numel() * buffer.element_size()`
- `py::gil_scoped_release` during all I/O
- Extract `uint8_t*` via `static_cast<uint8_t*>(buffer.data_ptr())`
- Return `bool` (success/failure), Python wrapper times the call

### 3.1 Configuration

Two separate scopes — thread pool globals (same pattern as baseline `cpp_utils.cpp`) and a new I/O config struct for the tunable parameters:

```cpp
// --- Thread pool (same pattern as baseline cpp_utils.cpp) ---
static std::mutex g_pool_mutex;
static std::unique_ptr<SimpleThreadPool> g_thread_pool;
static size_t g_thread_count = 0;      // 0 = auto-detect

// Track previous thread config to avoid unnecessary pool rebuilds
static size_t g_prev_thread_count  = 0;
static bool   g_prev_cpu_affinity  = false;

// --- I/O configuration (new, grouped in a struct) ---
struct IOConfig {
    // File descriptor flags
    bool     o_noatime          = false;
    bool     o_direct           = false;

    // posix_fadvise hints
    int      fadvise_read       = POSIX_FADV_NORMAL;
    int      fadvise_write      = POSIX_FADV_NORMAL;

    // I/O chunking
    size_t   io_chunk_size      = 0;      // 0 = full block

    // Prefetching (reads only)
    int      prefetch_depth     = 0;

    // Write-specific
    bool     fallocate_prealloc = false;
    int      sync_strategy      = 0;      // 0=none, 1=fdatasync, 2=sync_file_range

    // Thread/CPU affinity
    bool     cpu_affinity       = false;

    void reset() {
        *this = IOConfig{};  // reset all fields to defaults
    }
};

static std::mutex g_io_config_mutex;
static IOConfig   g_io_config;
```

Thread pool globals are modified only when `thread_count` or `cpu_affinity` changes. `IOConfig` is reset-then-applied on every `configure_all()` call — cheap, no side effects.

### 3.2 Bulk Setter (Optuna-Friendly)

```cpp
void configure_all(py::dict config) {
    // 1. Reset I/O config to defaults (no state leaks between trials)
    {
        std::lock_guard<std::mutex> lock(g_io_config_mutex);
        g_io_config.reset();

        if (config.contains("o_noatime"))      g_io_config.o_noatime = config["o_noatime"].cast<bool>();
        if (config.contains("o_direct"))       g_io_config.o_direct = config["o_direct"].cast<bool>();
        if (config.contains("fadvise_read"))   g_io_config.fadvise_read = config["fadvise_read"].cast<int>();
        if (config.contains("fadvise_write"))  g_io_config.fadvise_write = config["fadvise_write"].cast<int>();
        if (config.contains("io_chunk_size"))  g_io_config.io_chunk_size = config["io_chunk_size"].cast<size_t>();
        if (config.contains("prefetch_depth")) g_io_config.prefetch_depth = config["prefetch_depth"].cast<int>();
        if (config.contains("fallocate"))      g_io_config.fallocate_prealloc = config["fallocate"].cast<bool>();
        if (config.contains("sync_strategy"))  g_io_config.sync_strategy = config["sync_strategy"].cast<int>();
        if (config.contains("cpu_affinity"))   g_io_config.cpu_affinity = config["cpu_affinity"].cast<bool>();
    }

    // 2. Rebuild thread pool only if thread_count or cpu_affinity changed
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        if (config.contains("thread_count"))
            g_thread_count = config["thread_count"].cast<size_t>();

        bool need_rebuild = (g_thread_count != g_prev_thread_count)
                         || (g_io_config.cpu_affinity != g_prev_cpu_affinity);
        if (need_rebuild) {
            g_thread_pool.reset();
            g_thread_pool = std::make_unique<SimpleThreadPool>(g_thread_count, g_io_config.cpu_affinity);
            g_prev_thread_count = g_thread_count;
            g_prev_cpu_affinity = g_io_config.cpu_affinity;
        }
    }
}
```

### 3.3 get_config() for Logging

```cpp
py::dict get_config() {
    py::dict d;
    // Thread pool
    d["thread_count"]       = g_thread_count;
    // I/O config
    d["o_noatime"]          = g_io_config.o_noatime;
    d["o_direct"]           = g_io_config.o_direct;
    d["fadvise_read"]       = g_io_config.fadvise_read;
    d["fadvise_write"]      = g_io_config.fadvise_write;
    d["io_chunk_size"]      = g_io_config.io_chunk_size;
    d["prefetch_depth"]     = g_io_config.prefetch_depth;
    d["fallocate_prealloc"] = g_io_config.fallocate_prealloc;
    d["sync_strategy"]      = g_io_config.sync_strategy;
    d["cpu_affinity"]       = g_io_config.cpu_affinity;
    return d;
}
```

### 3.4 Modified I/O Functions

**Read path (`threaded_tunable_pread_file()`):**
```cpp
static bool threaded_tunable_pread_file(const std::string& path, uint8_t* buffer_ptr,
                                         size_t block_size, const IOConfig& cfg) {
    int flags = O_RDONLY;
    if (cfg.o_noatime) flags |= O_NOATIME;
    if (cfg.o_direct)  flags |= O_DIRECT;

    int fd = open(path.c_str(), flags);
    if (fd < 0) { /* error handling */ }

    // fadvise hint
    if (cfg.fadvise_read != POSIX_FADV_NORMAL) {
        posix_fadvise(fd, 0, block_size, cfg.fadvise_read);
    }

    // Chunked pread loop
    size_t chunk = (cfg.io_chunk_size > 0) ? cfg.io_chunk_size : block_size;
    size_t total_read = 0;
    while (total_read < block_size) {
        size_t to_read = std::min(chunk, block_size - total_read);
        ssize_t n = pread(fd, buffer_ptr + total_read, to_read, total_read);
        // ... same error handling as baseline ...
        total_read += n;
    }

    close(fd);
    return true;
}
```

I/O functions receive `const IOConfig&` — a snapshot taken at the start of each read/write call. This avoids holding the config mutex during I/O and prevents mid-operation config changes.

**Write path changes (inside worker lambda):**
```cpp
// After open, before pwrite loop:
if (cfg.fallocate_prealloc) {
    fallocate(fd, 0, 0, block_size);  // pre-allocate, ignore EOPNOTSUPP
}
if (cfg.fadvise_write != POSIX_FADV_NORMAL) {
    posix_fadvise(fd, 0, block_size, cfg.fadvise_write);
}

// Chunked pwrite loop (same pattern as read)
size_t chunk = (cfg.io_chunk_size > 0) ? cfg.io_chunk_size : block_size;

// After write completes, before close:
if (cfg.sync_strategy == 1) {
    fdatasync(fd);
} else if (cfg.sync_strategy == 2) {
    sync_file_range(fd, 0, block_size, SYNC_FILE_RANGE_WRITE);
}
```

**Prefetch (in `threaded_tunable_read_blocks()`, before dispatching workers):**
```cpp
// Snapshot config, then prefetch upcoming files via readahead()
const IOConfig cfg = []{ std::lock_guard<std::mutex> l(g_io_config_mutex); return g_io_config; }();

if (cfg.prefetch_depth > 0) {
    for (size_t i = 0; i < std::min((size_t)cfg.prefetch_depth, n); i++) {
        int pfd = open(source_files[i].c_str(), O_RDONLY);
        if (pfd >= 0) {
            readahead(pfd, 0, block_size);
            close(pfd);
        }
    }
}
```

### 3.5 O_DIRECT Buffer Alignment

When `O_DIRECT` is enabled, buffers must be aligned to the filesystem block size. PyTorch's pinned memory allocator (`cudaHostAlloc`) typically returns page-aligned (4096B) memory, which satisfies O_DIRECT requirements. Add a runtime check:

```cpp
if (cfg.o_direct) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(buffer_ptr);
    if (addr % 4096 != 0) {
        throw std::runtime_error("O_DIRECT requires 4096-byte aligned buffer");
    }
}
```

### 3.6 PyBind11 Exports

```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("threaded_tunable_read_blocks",  &threaded_tunable_read_blocks);
    m.def("threaded_tunable_write_blocks", &threaded_tunable_write_blocks);
    m.def("configure_all",            &configure_all);
    m.def("get_config",               &get_config);
    m.def("set_thread_count",         &set_thread_count);
    m.def("get_io_thread_count",      &get_io_thread_count);
    // Individual setters also exposed for manual use
    m.def("set_o_noatime",            &set_o_noatime);
    m.def("set_o_direct",             &set_o_direct);
    // ... etc
}
```

---

## 4. Python Wrapper: `threaded_tunable_backend.py`

Same pattern as `cpp_backend.py` — thin async wrapper, plus a typed dataclass for configuration.

### 4.1 Configuration Dataclass

```python
from dataclasses import dataclass, asdict
from enum import Enum

class FadviseHint(str, Enum):
    NORMAL = "normal"
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    WILLNEED = "willneed"
    NOREUSE = "noreuse"

class SyncStrategy(str, Enum):
    NONE = "none"
    FDATASYNC = "fdatasync"
    SYNC_FILE_RANGE = "sync_file_range"

@dataclass
class ThreadedTunableConfig:
    thread_count: int = 0
    o_noatime: bool = False
    o_direct: bool = False
    fadvise_hint: FadviseHint = FadviseHint.NORMAL
    io_chunk_kb: int = 0            # 0 = full block
    prefetch_depth: int = 0         # 0 = disabled
    fallocate: bool = False
    sync_strategy: SyncStrategy = SyncStrategy.NONE
    cpu_affinity: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fadvise_hint"] = self.fadvise_hint.value
        d["sync_strategy"] = self.sync_strategy.value
        return d
```

`FadviseHint` and `SyncStrategy` are `str, Enum` — type-safe with autocomplete, but serialize as plain strings via `to_dict()` so the C++ side receives the same string keys it expects.

### 4.2 Backend Functions

```python
import threaded_tunable_ext

async def threaded_tunable_read_blocks(block_size, buffer, block_indices, dest_files):
    start = time.perf_counter()
    threaded_tunable_ext.threaded_tunable_read_blocks(buffer, block_size, block_indices, dest_files)
    return time.perf_counter() - start

async def threaded_tunable_write_blocks(block_size, buffer, block_indices, dest_files):
    start = time.perf_counter()
    threaded_tunable_ext.threaded_tunable_write_blocks(buffer, block_size, block_indices, dest_files)
    return time.perf_counter() - start

def configure(config: ThreadedTunableConfig):
    """Apply typed config to C++ backend. Single entry point for all parameters including thread count."""
    threaded_tunable_ext.configure_all(config.to_dict())

def get_config() -> ThreadedTunableConfig:
    """Read current C++ config back as a dataclass (with enum values)."""
    raw = threaded_tunable_ext.get_config()
    # Converts raw int/string values back to enums
    return ThreadedTunableConfig(...)
```

---

## 5. Build: `setup_tunable.py`

```python
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='threaded_tunable_ext',
    ext_modules=[CppExtension(
        'threaded_tunable_ext',
        sources=[
            'backends/benchmark_cpp_utils/threaded_tunable_utils.cpp',
            'backends/benchmark_cpp_utils/simple_thread_pool.cpp',  # reuse
        ],
        extra_compile_args=['-std=c++17', '-O3', '-march=native', '-fPIC'],
    )],
    cmdclass={'build_ext': BuildExtension},
)
```

---

## 6. Optuna Integration: `optuna_tuner.py`

### 6.1 Architecture

```
optuna_tuner.py                  New file. Entry point for auto-tuning.
    |
    +---> Defines Optuna study (sampler, pruner, search space)
    +---> Objective function: runs a short benchmark, returns throughput
    +---> Builds ThreadedTunableConfig per trial, calls configure() to apply
    +---> Uses existing benchmark_core.py for buffer alloc and iteration
    |
    v
Results: best params, parameter importance, optimization history
```

### 6.2 Objective Function

```python
def objective(trial: optuna.Trial, mode: str = "write") -> float:
    """Single Optuna trial. Returns throughput in GB/s (maximize)."""

    # 1. Sample parameters into typed config
    config = ThreadedTunableConfig(
        thread_count  = trial.suggest_int("thread_count", 4, 128, log=True),
        o_direct      = trial.suggest_categorical("o_direct", [True, False]),
        fadvise_hint  = FadviseHint(trial.suggest_categorical("fadvise_hint", [e.value for e in FadviseHint])),
        io_chunk_kb   = trial.suggest_categorical("io_chunk_kb", [0, 256, 512, 1024, 2048, 4096]),
        cpu_affinity  = trial.suggest_categorical("cpu_affinity", [True, False]),
    )
    block_size_mb = trial.suggest_categorical("block_size_mb", [2, 4, 8, 16, 32, 64, 128])

    # Conditional: read-only params
    if mode in ("read", "concurrent"):
        config.o_noatime      = trial.suggest_categorical("o_noatime", [True, False])
        config.prefetch_depth = trial.suggest_int("prefetch_depth", 0, 8)

    # Conditional: write-only params
    if mode in ("write", "concurrent"):
        config.fallocate      = trial.suggest_categorical("fallocate", [True, False])
        config.sync_strategy  = SyncStrategy(trial.suggest_categorical("sync_strategy", [e.value for e in SyncStrategy]))

    # 2. Apply typed config to C++ backend
    configure(config)

    # 3. Run short benchmark (reduced data for speed)
    elapsed = run_trial_benchmark(block_size_mb, mode)

    # 4. Return throughput
    total_gb = TRIAL_DATA_SIZE_GB
    return total_gb / elapsed
```

### 6.3 Study Configuration

```python
study = optuna.create_study(
    study_name=f"threaded_tunable_{mode}_tuning",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(
        n_startup_trials=20,       # random exploration first
        multivariate=True,         # model parameter interactions
    ),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=0,
    ),
    storage="sqlite:///results/optuna_study.db",  # persistent, resume across LSF jobs
)

study.optimize(objective, n_trials=200, timeout=3600)
```

### 6.4 Separate Studies

| Study | Objective | Read-Only Params | Write-Only Params |
|-------|-----------|-----------------|-------------------|
| `threaded_tunable_write_tuning` | Max write throughput | excluded | fallocate, sync_strategy |
| `threaded_tunable_read_tuning` | Max read throughput | o_noatime, prefetch_depth | excluded |
| `threaded_tunable_concurrent_tuning` | Max combined throughput | all | all |

### 6.5 Trial Benchmark Design

Each trial must produce realistic numbers for the KV cache offload use case (CPU memory ↔ storage). This means:

- **Buffers**: `torch.Tensor` with `dtype=float16`, `pin_memory=True` — same as production checkpoint buffers. Pre-allocate once at startup, reuse across all trials. The tunable backend receives `torch::Tensor` (not raw pointers), matching the baseline `cpp` API exactly.
- **Cleaning buffer**: Separate 50GB `torch.Tensor` (float16, pinned), 3200 × 32MB files on `/dev/shm` — same as existing `benchmark_core.allocate_buffers()`.
- **Total data per trial**: 10GB (enough for stable measurement, fast enough for iteration)
- **Iterations**: 2 (take mean, discard first as warmup)
- **File cleanup**: Clean benchmark files between trials to prevent disk filling

**Cache cleaning per mode** (mandatory for realistic results):

| Mode | Cache cleaning | Why |
|------|---------------|-----|
| **Write** | Not needed | Data originates from CPU memory (hot tensor), written to cold storage. Page cache is irrelevant — no prior read to cache. |
| **Read** | **Mandatory** — read 100GB (3200 × 32MB) from `/dev/shm` using `threaded_tunable_read_blocks` after every write phase, before every read measurement | Without this, reads hit page cache instead of storage. Optimizing for cache hits produces useless params for real KV cache reload. |
| **Concurrent** | **Mandatory** — clean before each concurrent iteration | Same reason: read portion would hit warm cache otherwise. |

Cache cleaning uses the tunable backend's own read function (not baseline `cpp_read_blocks`) to avoid loading a second extension and to match the existing pattern where each backend cleans cache with its own implementation.

**Estimated trial times:**

| Mode | Per trial | 200 trials |
|------|-----------|------------|
| Write | ~5s | ~17 min |
| Read | ~25s (includes 100GB cache clean) | ~80 min |
| Concurrent | ~30s | ~100 min |

### 6.6 Constraint Handling

| Invalid Combination | Handling |
|---------------------|----------|
| `O_DIRECT` + misaligned buffer | Runtime check, skip trial with `TrialPruned` |
| `O_DIRECT` on tmpfs (`/dev/shm`) | Detect filesystem, skip trial |
| `io_chunk_size` > `block_size` | Clamp to block_size in C++ |
| `O_NOATIME` without file ownership | Silently ignored by kernel (EPERM → flag dropped) |
| `fallocate` on NFS | Catch `EOPNOTSUPP`, ignore (write still works) |
| `sync_file_range` unsupported | Catch error, fall back to `none` |

---

## 7. CLI Interface

### 7.1 `optuna_tuner.py` — Tuning Entry Point

```
python optuna_tuner.py \
    --mode write|read|concurrent \
    --n-trials 200 \
    --timeout 3600 \
    --data-gb 10 \
    --backend threaded_tunable \
    --storage-path /mnt/benchmark_tmp \
    --study-db results/optuna_study.db \
    --export-config results/best_write_config.json
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `write` | Which I/O direction to optimize |
| `--n-trials` | `200` | Max Optuna trials |
| `--timeout` | `3600` | Max seconds for the study |
| `--data-gb` | `10` | Total data per trial (kept small for speed) |
| `--backend` | `threaded_tunable` | Which tunable backend to optimize (future: `iouring_tunable`) |
| `--storage-path` | from `.env` | Where benchmark files are written |
| `--study-db` | `results/optuna_study.db` | SQLite path for checkpoint/resume |
| `--export-config` | `results/best_{mode}_config.json` | Where to save best params |

### 7.2 `compare_file_operations.py` — Benchmark with All Backends

The benchmark and optimization stages are **separate**:

1. **Optimize** (run once per system): `optuna_tuner.py` → saves `best_{mode}_config.json`
2. **Benchmark** (run anytime): `compare_file_operations.py` → runs all backends side-by-side, `threaded_tunable` uses the saved config

Add `threaded_tunable` as a backend choice and `--tunable-config` to load saved Optuna results:

```bash
# Run a single backend
python compare_file_operations.py \
    --backend threaded_tunable \
    --tunable-config results/best_write_config.json \
    --mode data --total-gb 100 --iterations 5

# Run ALL backends in one job (apples-to-apples comparison)
python compare_file_operations.py \
    --backend all \
    --tunable-config results/best_write_config.json \
    --mode data --total-gb 100 --iterations 5
```

When `--backend all` is used:
- `cpp`, `python_self_imp`, `python_aiofiles` run with their defaults
- `threaded_tunable` loads params from `--tunable-config`
- All backends run under the same conditions (same machine, same buffer, same cache cleaning)
- Results are saved per-backend as usual, directly comparable in plotter

### 7.3 Running Modes

Both stages run on any Linux machine. LSF scripts are optional wrappers.

**Direct (any Linux machine):**
```bash
# Stage 1: Optimize (once per system)
python optuna_tuner.py --mode write --n-trials 200

# Stage 2: Benchmark (all backends, including tuned threaded_tunable)
python compare_file_operations.py --backend all \
    --tunable-config results/best_write_config.json \
    --mode data --total-gb 100 --iterations 5
```

**LSF:**
```bash
# Stage 1
./scripts/run_optuna_on_lsf.sh write 200

# Stage 2
./scripts/run_benchmark_on_lsf.sh full all results/best_write_config.json
```

---

## 8. Output

### 8.1 Tuning Output (stdout/log during `optuna_tuner.py`)

After each trial, print a one-liner:

```
Trial  17 | 8.42 GB/s | threads=48 block=32MB chunk=2MB fadvise=sequential o_direct=off fallocate=on
Trial  18 | 7.91 GB/s | threads=32 block=16MB chunk=1MB fadvise=random     o_direct=off fallocate=off
...
```

At study completion, print a comparison summary:

```
=== Optuna Tuning Complete: write mode ===

Best trial (#142):
  Throughput:      12.4 GB/s
  thread_count:    48
  block_size_mb:   32
  o_direct:        False
  fadvise_hint:    sequential
  io_chunk_kb:     2048
  fallocate:       True
  sync_strategy:   none
  cpu_affinity:    False

Top-5 parameter importance (fANOVA):
  1. thread_count     62.3%
  2. block_size_mb    18.7%
  3. io_chunk_kb       9.1%
  4. fadvise_hint      5.4%
  5. fallocate         2.8%

Config exported to: results/best_write_config.json
Study saved to:     results/optuna_study.db (142 trials, resumable)
```

### 8.2 Benchmark Comparison Output (after `--backend all`)

When running with `--backend all`, the benchmark logs each backend's results as usual. At the end, print a summary comparing all backends:

```
=== Benchmark Complete: data mode, 100GB, 5 iterations ===

threaded_tunable config loaded from: results/best_write_config.json

Write throughput (GB/s) by (threads, block_size):
                     16 threads    32 threads    64 threads
  cpp                 8.2           9.1          8.8
  threaded_tunable   10.4          12.4         11.9         ← tuned (thread_count=48 from config used for all)
  python_self_imp     6.1           6.8          6.5

Read throughput (GB/s) by (threads, block_size):
  cpp                14.2          15.1         14.8
  threaded_tunable   16.7          18.7         17.9
  python_self_imp    11.3          12.1         11.8

Results saved per-backend: results/data_*_cpp.json, results/data_*_threaded_tunable.json, ...
Plot: python plotter.py data results/*.json
```

### 8.3 Exported Best Config

```json
{
  "thread_count": 48,
  "block_size_mb": 32,
  "o_direct": false,
  "fadvise_hint": "sequential",
  "io_chunk_kb": 2048,
  "prefetch_depth": 4,
  "fallocate": true,
  "sync_strategy": "none",
  "cpu_affinity": false,
  "measured_write_throughput_gbs": 12.4,
  "measured_read_throughput_gbs": 18.7,
  "storage_path": "/mnt/storage",
  "study_name": "threaded_tunable_write_tuning",
  "trial_number": 142,
  "total_trials": 200
}
```

### 8.4 Workflow Examples

**Full workflow (first time on a system):**
```
1. Optimize:    python optuna_tuner.py --mode write --n-trials 200
               → saves results/best_write_config.json

2. Benchmark:   python compare_file_operations.py --backend all \
                    --tunable-config results/best_write_config.json \
                    --mode data --total-gb 100 --iterations 5
               → runs cpp, threaded_tunable, python_self_imp, etc. side-by-side

3. Compare:     python plotter.py data results/*.json
               → overlays all backends on same chart
```

**Just run threaded_tunable (other baselines already exist):**
```
python compare_file_operations.py --backend threaded_tunable \
    --tunable-config results/best_write_config.json \
    --mode data --total-gb 100 --iterations 5
```

**Resume a partial Optuna study (e.g., LSF job timed out):**
```
python optuna_tuner.py --mode write --n-trials 200 --study-db results/optuna_study.db
# Picks up from where it left off (SQLite has all completed trials)
```

---

## 9. Smoke Test: `tests/test_threaded_tunable.py`

A pre-Optuna validation script that verifies the new backend builds, runs, and produces comparable results to the baseline `cpp` backend. Run this after Phase 1 to catch issues before investing in Optuna integration.

```
python tests/test_threaded_tunable.py [--storage-path /path] [--tolerance 0.15]
```

### 9.1 What It Tests

| Test | What | Pass Criteria |
|------|------|---------------|
| **Import** | `import threaded_tunable_ext` succeeds | No ImportError |
| **configure_all / get_config roundtrip** | Set params via dataclass, read back, verify match | All fields equal |
| **Defaults match baseline** | Write+read with all defaults, compare throughput vs `cpp` | Within tolerance (default 15%) |
| **Each knob individually** | Toggle one knob at a time (O_NOATIME, fadvise=sequential, chunk=1MB, etc.), run write+read | No crash, data integrity verified |
| **Combined tunable parameters** | Multiple tunable parameters enabled together | No crash, data integrity verified |
| **Invalid combos graceful** | O_DIRECT on tmpfs (if applicable) | Fails gracefully, no segfault |

### 9.2 Test Parameters

Small and fast — just verifying correctness, not measuring performance:

- **Data**: 1GB buffer, 10 blocks of 8MB
- **Threads**: 16 (match baseline default)
- **Iterations**: 1

### 9.3 Output

```
=== threaded_tunable backend smoke test ===
[PASS] Import threaded_tunable_ext
[PASS] configure_all / get_config roundtrip
[PASS] Defaults write: 0.045s (cpp baseline: 0.043s, diff: +4.7%)
[PASS] Defaults read:  0.038s (cpp baseline: 0.036s, diff: +5.6%)
[PASS] Knob: o_noatime=True         — read OK, data verified
[PASS] Knob: fadvise=sequential     — write OK, read OK, data verified
[PASS] Knob: io_chunk_kb=1024       — write OK, read OK, data verified
[PASS] Knob: prefetch_depth=4       — read OK, data verified
[PASS] Knob: fallocate=True         — write OK, data verified
[PASS] Knob: sync_strategy=fdatasync — write OK, data verified
[PASS] Knob: cpu_affinity=True      — write OK, read OK, data verified
[PASS] Combined tunable parameters               — write OK, read OK, data verified
[SKIP] O_DIRECT on tmpfs            — not supported, skipped gracefully

11/12 passed, 1 skipped, 0 failed
Baseline comparison: write +4.7%, read +5.6% (within 15% tolerance)
```

---

## 10. Implementation Order

### Phase 1: Tunable C++ Backend
1. Create `backends/benchmark_cpp_utils/threaded_tunable_utils.cpp`
   - Copy from `cpp_utils.cpp`, add global config, `configure_all()`, `get_config()`
   - Modify read path: `O_NOATIME`, `O_DIRECT`, `fadvise`, chunked I/O, prefetch
   - Modify write path: `O_DIRECT`, `fadvise`, `fallocate`, chunked I/O, sync strategy
   - Add CPU affinity support to thread pool
2. Create `setup_threaded_tunable.py`
3. Create `backends/threaded_tunable_backend.py` (Python wrapper + `ThreadedTunableConfig` dataclass)
4. Create `tests/test_threaded_tunable.py`
5. Build on remote, run smoke test — all tests must pass before proceeding

### Phase 2: Backend Integration
1. Add `threaded_tunable` backend choice to `compare_file_operations.py`
2. Add dispatch branch in `benchmark_core.py`
3. Add `--tunable-config` CLI flag for loading saved configs
4. Update `setup_env.sh` to build `threaded_tunable_ext`
5. Run baseline comparison: `cpp` vs `threaded_tunable` with defaults (should be identical)

### Phase 3: Optuna Tuner
1. Add `optuna` to dependencies in `setup_env.sh`
2. Create `optuna_tuner.py` (project root) with objective function, study setup, CLI
3. Wire up `ThreadedTunableConfig` + `configure()` from objective function
4. Implement separate read/write/concurrent studies with conditional params
5. Add SQLite storage for checkpoint/resume across LSF jobs
6. Create `scripts/run_optuna_on_lsf.sh` — LSF submission wrapper (resource requests, job name, log path)
7. Create `scripts/optuna_job.sh` — job script (source .env, activate venv, run `optuna_tuner.py`)
8. Test with small study (20 trials) on remote

### Phase 4: Analysis and Validation
1. Run full tuning study (200+ trials)
2. Extract parameter importance (fANOVA)
3. Export best config to JSON
4. Validate: full benchmark with best config (100GB, 5 iterations)
5. Compare baseline `cpp` vs tuned `threaded_tunable` via plotter
6. Add Optuna visualization (optional)

---

## 11. Risks and Considerations

| Risk | Mitigation |
|------|-----------|
| O_DIRECT not supported on tmpfs (`/dev/shm`) | Detect at trial start; skip or mark as failed trial. Works on real filesystems (NFS, ext4, GPFS). |
| `fallocate()` not supported on NFS | Catch `EOPNOTSUPP`, ignore gracefully (write still proceeds without pre-allocation) |
| Noisy measurements on shared cluster | Use multiple iterations per trial, run during off-peak, consider using dedicated nodes via LSF resource requests |
| Trial overhead (buffer alloc, file cleanup) | Allocate buffer once, reuse across trials. Clean up files between trials but don't re-create the buffer. |
| Parameter interactions | TPESampler with `multivariate=True` models correlations. 20 startup random trials ensure exploration. |
| Overfitting to specific storage | Run tuning on target storage system. Results may differ between /dev/shm, NFS, GPFS, NVMe. Store results per-storage. |
| Tunable defaults ≠ baseline | Validate explicitly: run both with same thread count/block size, verify identical throughput. |

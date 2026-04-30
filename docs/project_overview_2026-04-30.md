# CPU-to-Storage I/O Benchmark — Project Overview (2026-04-30)

A short orientation for teammates new to the project. For deep details, follow the links inline or the "Where to look next" section at the bottom.

---

## 1. Purpose

We are optimizing the **CPU ↔ Storage I/O path** for **KV cache offload** in llm-d. LLM inference reuses KV cache to skip prefill, but at scale the cache no longer fits in memory (e.g., Llama-3.1-70B needs ~305 GB for 1M tokens). llm-d adds storage as a third cache tier (GPU → CPU → Storage), and the bottleneck is how fast we can read/write KV blocks to that tier — and whether the I/O strategy can be tuned per storage type.

This project is **forked from the LLMD team's KV cache offload benchmark**, which evaluated several I/O strategies over fixed grid sweeps of thread count and block size. We use it as the baseline harness and extend it in two directions: (a) a new `iouring` backend, and (b) **automatic parameter tuning with [Optuna](https://optuna.org)** — an open-source Bayesian hyperparameter optimization framework. Optuna's TPE (Tree-structured Parzen Estimator) sampler learns from previous trials and focuses on promising regions of the search space, which is far more efficient than grid or random search; we use it to find the best I/O parameters per storage type instead of trial-and-error.

### Methods we benchmark

- **C++ pread/pwrite + thread pool** ([backends/cpp_backend.py](../backends/cpp_backend.py), C++ core in [benchmark_cpp_utils/cpp_utils.cpp](../backends/benchmark_cpp_utils/cpp_utils.cpp)) — baseline. One `pread`/`pwrite` syscall per block, dispatched in parallel by a custom `SimpleThreadPool`. GIL released during I/O.
- **Threaded tunable** ([backends/threaded_tunable_backend.py](../backends/threaded_tunable_backend.py), C++ core in [benchmark_cpp_utils/threaded_tunable_utils.cpp](../backends/benchmark_cpp_utils/threaded_tunable_utils.cpp)) — same I/O core as the baseline, but exposes 9 tunable knobs (`O_DIRECT`, `O_NOATIME`, `posix_fadvise` hints, IO chunk size, prefetch depth, `fallocate`, sync strategy, CPU affinity, thread count) that Optuna explores.
- **io_uring** ([backends/iouring_backend.py](../backends/iouring_backend.py), C++ core in [benchmark_iouring_utils/iouring_utils.cpp](../backends/benchmark_iouring_utils/iouring_utils.cpp)) — Linux kernel asynchronous I/O (5.1+). Submission/completion rings shared with the kernel let us batch hundreds of ops in one syscall, eliminating most context-switch overhead. Tunable: `queue_depth`, `batch_size`, `block_size`.
- **Python asyncio + threadpool** ([backends/python_self_backend.py](../backends/python_self_backend.py)) — pure-Python reference using `asyncio.run_in_executor` over `os.readv`/`os.write` in a `ThreadPoolExecutor`.
- **aiofiles** ([backends/aiofiles_backend.py](../backends/aiofiles_backend.py)) — community [aiofiles](https://github.com/Tinche/aiofiles) library; native async coroutines with `f.readinto(memoryview)` and `aiofiles.os.replace()` for atomic rename.
- **NIXL** ([backends/nixl_backend.py](../backends/nixl_backend.py)) — NVIDIA's [NIXL](https://github.com/ai-dynamo/nixl) data-movement library with POSIX backend. Persistent reader/writer agents, explicit memory registration, descriptor-list transfers, busy-poll completion.

---

## 2. Requirements to run

- **OS**: Linux for full functionality. The C++ baseline (`cpp`), `python_self_imp`, and `aiofiles` use only POSIX APIs and *should* work on macOS too (untested). The `iouring` backend is Linux-only (io_uring is a Linux kernel interface). The `threaded_tunable` knobs (`posix_fadvise`, `O_DIRECT`, `O_NOATIME`, `sync_file_range`, `fallocate`) are Linux-specific and would need porting for macOS. NIXL requires CUDA. Windows is unsupported.
- **Kernel** (for `iouring` only): 5.6+ with `kernel.io_uring_disabled=0`. Without this the `iouring` backend gracefully falls back at import; other backends still work.
- **Toolchain**: Python 3.9+, GCC 11+, PyTorch, ninja. `liburing` is built from source by [setup_env.sh](../setup_env.sh) into `$HOME/.local`.
- **GPU**: Recommended. Pinned-memory CPU buffers (`torch.zeros(..., pin_memory=True)`) work best with CUDA available; we run on LSF/K8s pods with 1 GPU. Not strictly required but the runtime path was developed assuming it.
- **RAM**: ≥160 GB recommended. The cleaning-buffer logic allocates a hardcoded 50 GB pinned tensor and a 100 GB main buffer by default. Smaller machines need `--buffer-size` lowered.
- **Storage**: Anything POSIX-mountable. Tested on `/dev/shm` (tmpfs), GPFS (CCC cluster), XFS (lsf-gpu4 local NVMe). `STORAGE_PATH` env var picks the mount.

Canonical hardware spec is in [deployment/io_bench_pod.yaml](../deployment/io_bench_pod.yaml) (450 GiB RAM, 12-128 CPU, 1 GPU).

---

## 3. What's in the repo today

Beyond the 6 backends listed in section 1:

- **3 benchmark modes** — write-only / read-only / concurrent (read+write via `asyncio.gather`). Dispatched from [utils/benchmark_core.py](../utils/benchmark_core.py); CLI in [compare_file_operations.py](../compare_file_operations.py).
- **Optuna auto-tuners** — [optuna_tuner_threads.py](../optuna_tuner_threads.py) (threaded_tunable) and [optuna_tuner_iouring.py](../optuna_tuner_iouring.py) (iouring). Each runs three studies (write / read / concurrent) and exports a per-mode best config. fANOVA analysis in [scripts/analyze_optuna_threads.py](../scripts/analyze_optuna_threads.py).
- **Plotting** — [plotter.py](../plotter.py) (per-backend grids) and [plot_comparison.py](../plot_comparison.py) (cpp vs tunable bar charts).
- **Checkpointing** — incremental JSON results with resume support in [utils/checkpoints_utils.py](../utils/checkpoints_utils.py).

---

## 4. Latest results

Two environments, very different I/O characteristics. **CCC** uses GPFS — a shared HPC parallel filesystem with high run-to-run variance because other cluster tenants compete for the same I/O path, making it hard to isolate the effect of parameter changes. **lsf-gpu4** is a single-node machine with a local-NVMe XFS mount and 660 GB RAM, so I/O is exclusive and reproducible — but the 30 GB workload fits in page cache, so numbers reflect memcpy bandwidth more than real NVMe throughput.

### CCC (GPFS, shared, 60 trials/mode, 10 GB/trial, 100 GB buffer)

| Mode | cpp baseline | threaded_tunable (Optuna) |
|---|---|---|
| Read | ~50 GB/s | ~49 GB/s (comparable) |
| Write | — | worse than cpp |
| Concurrent | — | inconclusive (high variance) |

Pipeline works end-to-end, but GPFS noise prevents a meaningful winner. POSIX hints (fadvise, O_DIRECT, chunking) showed no benefit here.

### lsf-gpu4 (XFS local NVMe, exclusive, 30 GB total / 5 iter)

| Mode | cpp (best) | threaded_tunable (Optuna) | iouring (Optuna) |
|---|---|---|---|
| Write | 24.0 GB/s | 21.6 (−10 %) | 19.6 (−18 %) |
| Read | 23.3 GB/s | 19.5 (−16 %) | 18.7 (−20 %) |
| **Concurrent (combined)** | 20.6 GB/s | **24.0 (+17 %)** | **20.9 (+5 %)** |

Auto-tuning wins concurrent on both backends; pure write/read still favor cpp's grid max (structural — cpp reports max over 18 cells, tuned backend picks one). For iouring, `block_size_mb` dominates (96 % of fANOVA variance). Caveat: all buffered I/O — next step is `O_DIRECT` to bypass page cache.

Full chart and discussion: [docs/team_presentation_2026-04-13.md](team_presentation_2026-04-13.md).

---

## 5. How to run / recreate

### One-time setup
```bash
./setup_env.sh        # idempotent: venv + deps + 3 C++ extensions + liburing
source .venv/bin/activate
```

Create `.env` in project root (gitignored):
```
PROJ_DIR=/path/to/cpu_to_storage
STORAGE_PATH=/path/to/benchmark_tmp
CLUSTER_NAME=your_cluster
```

### Single benchmark
```bash
python compare_file_operations.py --backend cpp --mode data --total-gb 30
python compare_file_operations.py --backend threaded_tunable --threads-config results/best_config.json --mode data --total-gb 30
python compare_file_operations.py --backend iouring --iouring-config results/best_iouring_config.json --mode data --total-gb 30
```

### Compare all backends on LSF
```bash
./run_benchmark_on_lsf.sh compare-short                                   # quick wiring check
./run_benchmark_on_lsf.sh compare-full "cpp threaded_tunable iouring"     # full
```
See [run_benchmark_on_lsf.sh](../run_benchmark_on_lsf.sh).

### Auto-tune (Optuna)
```bash
# threaded_tunable
python optuna_tuner_threads.py --preset short    # ~6 min, sanity
python optuna_tuner_threads.py                   # full, ~5 hours
./scripts/run_optuna_threads_on_lsf.sh           # via LSF

# iouring (requires kernel.io_uring_disabled=0)
./scripts/run_optuna_iouring.sh short            # quick
./scripts/run_optuna_iouring.sh                  # full
```
Output: `results/best_{config,iouring_config}_{timestamp}.json`. Wrappers: [scripts/run_optuna_threads_on_lsf.sh](../scripts/run_optuna_threads_on_lsf.sh), [scripts/run_optuna_iouring.sh](../scripts/run_optuna_iouring.sh).

### Plot
```bash
python plot_comparison.py results/cpp_data.json results/tunable_data.json [results/concurrent.json]
```

### Smoke tests
```bash
python -m tests.test_threaded_tunable    # build + config roundtrip + each knob
python -m tests.test_iouring             # iouring extension sanity
```
Sources: [tests/test_threaded_tunable.py](../tests/test_threaded_tunable.py), [tests/test_iouring.py](../tests/test_iouring.py).

---

## 6. Where to look next

- [KNOWLEDGE_BASE.md](../KNOWLEDGE_BASE.md) — full reference: architecture, all CLI args, result format, every tunable
- [docs/team_presentation_2026-04-13.md](team_presentation_2026-04-13.md) — latest slide deck with charts and findings
- [docs/iouring_implementation_plan.md](iouring_implementation_plan.md) — io_uring design + future work (SQPOLL, registered files/buffers, O_DIRECT)
- [docs/optuna_autotuning_plan.md](optuna_autotuning_plan.md) — Optuna design + parameter rationale + storage-type matrix
- [README.md](../README.md) — original user-facing docs

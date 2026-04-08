# io_uring Backend Implementation Plan

## Goal

Add a new `iouring` backend to the I/O benchmark project using Linux io_uring for async I/O submission. Built as a separate C++ extension (`iouring_ext`) to allow direct comparison against the existing `cpp_ext` (pread/pwrite + thread pool) baseline.

---

## Architecture

```
backends/iouring_backend.py            Thin async Python wrapper (same pattern as cpp_backend.py)
    |
    v
iouring_ext (PyBind11 module)          New C++ extension
    |
    +-- backends/benchmark_iouring_utils/iouring_utils.cpp    Core io_uring I/O logic
    +-- (links against liburing)
```

The existing `cpp_ext` stays untouched as the baseline.

---

## File Changes

### New files
| File | Purpose |
|------|---------|
| `backends/iouring_backend.py` | Python wrapper: `iouring_read_blocks()`, `iouring_write_blocks()`, `set_thread_count()` |
| `backends/benchmark_iouring_utils/iouring_utils.cpp` | Core C++ implementation using liburing |
| `setup_iouring.py` | Build script for `iouring_ext` (separate from `setup.py`) |

### Modified files
| File | Change |
|------|--------|
| `compare_file_operations.py` | Add `iouring` to `--backend` choices, add availability check |
| `utils/benchmark_core.py` | Add `iouring` branches in `run_benchmark_iteration()`, `run_concurrent_benchmark_iteration()`, `setup_executor()`, `shutdown_executor()` |
| `utils/config.py` | No change needed — `iouring` will use its own thread/queue management like `cpp` |

---

## C++ Implementation Design (`iouring_utils.cpp`)

### Design principle

Separate meaningful strategy functions from orchestration. Each strategy function encapsulates
logic that could change independently — for Bayesian tuning of parameters and OpenEvolve
mutation of strategies. Don't extract trivial operations (close FDs, init ring) into their own
functions — keep those inline.

### Strategy functions

**`open_files_for_read(paths) -> vector<int>`**
Opens all source file FDs. For reads, files already exist so no O_CREAT contention — can
open concurrently or sequentially. This is an evolvable strategy (batch size, ordering).
~15-20 lines.

**`open_files_for_write(dest_files) -> (vector<int> fds, vector<string> tmp_paths)`**
Serialized O_CREAT pattern: opens temp files one-by-one in the caller to avoid directory
inode contention. Creates parent dirs if needed. Evolvable: could experiment with batched
creation, per-thread subdirs, or pipelined open+write.
~20-30 lines.

**`submit_io_ops(ring, fds, buffer, block_indices, block_size, is_write) -> int`**
Fills the SQ with pread or pwrite SQEs in batches of `queue_depth`. Handles ring overflow
by submitting and reaping between batches. Sets user_data per SQE for completion tracking.
Returns number of ops submitted. Evolvable: submission ordering, batch size, interleaved
submit/reap ratio.
~30-50 lines.

**`reap_completions(ring, expected_count) -> bool`**
Drains the CQ until `expected_count` completions received. Checks each CQE `res` for errors
(negative = errno). Returns false on any failure. Evolvable: wait strategy (block vs. peek),
partial completion handling, error tolerance.
~20-30 lines.

**`rename_temp_files(tmp_paths, dest_files) -> bool`**
Renames all temp files to final names atomically. Sequential — rename is a fast metadata op.
~10-15 lines.

**`validate_buffer_args(buffer, block_size, block_indices, files)`**
Shared input validation: CPU check, contiguous check, size match, bounds check. Shared
between read and write paths. Deduplicates ~6 identical checks.
~15 lines.

### Orchestration functions (PyBind11-exposed)

**`iouring_read_blocks(buffer, block_size, block_indices, source_files) -> bool`**
1. `validate_buffer_args()`
2. Release GIL
3. `open_files_for_read()` — get FDs
4. Init `io_uring` ring with configured queue depth
5. `submit_io_ops()` — fill SQ with pread SQEs
6. `reap_completions()` — drain CQ
7. Close FDs, cleanup ring
8. Return success/failure

**`iouring_write_blocks(buffer, block_size, block_indices, dest_files) -> bool`**
1. `validate_buffer_args()`
2. Release GIL
3. `open_files_for_write()` — get FDs + tmp paths (serialized O_CREAT)
4. Init `io_uring` ring
5. `submit_io_ops()` — fill SQ with pwrite SQEs
6. `reap_completions()` — drain CQ
7. Close FDs, cleanup ring
8. `rename_temp_files()` — atomic commit
9. Return success/failure

### Error handling

- Each CQE `res` field checked for errors (negative = errno)
- On any failure: drain remaining completions, close FDs, clean up temp files, return false
- EINTR handling: io_uring handles this internally (unlike raw pread/pwrite)

---

## Tunable Parameters

Exposed via Python-callable functions (like `set_thread_count()` in cpp_ext).
Each parameter exposed as an individual setter/getter (not bundled), so Optuna can define its
search space per-parameter independently. Stored as module-level globals (same pattern as
`g_thread_count` in cpp_ext).

```cpp
void set_queue_depth(size_t depth);
void set_use_sqpoll(bool enable);
void set_use_direct(bool enable);
void set_use_registered_files(bool enable);
void set_use_registered_buffers(bool enable);
void set_batch_size(size_t size);
```

### Parameter Details

**`queue_depth`** (default: 256)
The size of io_uring's submission queue (SQ) and completion queue (CQ). This controls how many
I/O operations can be "in flight" at the same time — submitted to the kernel but not yet
completed. A higher queue depth lets the kernel see more pending work, which allows it to
optimize scheduling (merge adjacent reads, reorder for disk layout, keep the device queue full).
Too low and the storage device sits idle between batches. Too high and you waste memory on ring
buffers and may hit diminishing returns. For NVMe drives that support 64K internal queue entries,
higher values (256-1024) help saturate the device. For network filesystems, the optimal value
depends on how many concurrent RPCs the server handles well. Typical range to search: 32-1024.

**`use_sqpoll`** (default: false)
When enabled, the kernel spawns a dedicated polling thread that continuously monitors the
submission queue for new entries. Without SQPOLL, every time we submit I/O ops we must call
`io_uring_submit()`, which is a syscall — a context switch from user mode to kernel mode (~1-2us
each). With SQPOLL, we just write SQEs into shared memory and the kernel thread picks them up
automatically — zero syscalls for submission. The tradeoff: the kernel polling thread consumes
one full CPU core spinning, even when idle. Worth it for sustained high-throughput workloads
(like our benchmarks), but wasteful for sporadic I/O. May require elevated privileges
(`CAP_SYS_ADMIN`) on some kernels — needs testing on our cluster. On kernel 5.14, SQPOLL is
supported but the permission requirement varies by sysctl config.

**`use_direct`** (default: false)
Opens files with the `O_DIRECT` flag, which bypasses the kernel's page cache entirely. Normally,
when you read a file, data travels: storage device → kernel page cache (kernel memory) → your
buffer (user memory) — two memory copies. With O_DIRECT, data goes straight from the storage
device into your buffer via DMA — one copy. For writes, it goes directly from your buffer to the
device without being staged in the page cache first. This eliminates ~50MB of unnecessary
memcpy per block. The tradeoffs: (1) your buffer must be aligned to the filesystem's block size
(typically 4KB or 512 bytes) — `posix_memalign()` or aligned PyTorch allocations needed.
(2) You lose the page cache, so repeated reads of the same file become slower. For our workload
(write once, read once) there's no caching benefit, so O_DIRECT is a pure win — on real storage.
**O_DIRECT has no effect on tmpfs** (`/dev/shm`) because tmpfs is already RAM — there's no
device-to-cache copy to skip. This parameter only matters when benchmarking on real storage
(NVMe, Spectrum Scale, etc.).

**`use_registered_files`** (default: false)
Normally, each SQE references a file descriptor (FD) number, and the kernel must look up the
FD in the process's file descriptor table for every I/O operation. With registered files, we
pre-register a batch of FDs with the io_uring instance via `io_uring_register_files()`. The
kernel maps them once upfront, and subsequent SQEs reference them by index into the registered
array — skipping the per-op FD lookup. This saves a small amount of overhead per operation
(~100-200ns), which adds up when submitting thousands of ops. The setup cost is a single
registration call. Most useful when reusing the same FDs across many ops (e.g., reading from
the same set of files repeatedly). For our benchmark where each file is opened and read once,
the benefit is smaller but still worth testing.

**`use_registered_buffers`** (default: false)
Similar to registered files but for memory buffers. Normally, for each I/O op, the kernel must
call `get_user_pages()` to pin and map the user-space buffer into kernel address space — this
involves page table walks and TLB operations. With `io_uring_register_buffers()`, we
pre-register our buffer regions once. The kernel pins and maps them upfront, and subsequent ops
using `io_uring_prep_read_fixed()` / `io_uring_prep_write_fixed()` skip this per-op overhead.
Since our PyTorch buffers are already pinned (`pin_memory=True`), the kernel still needs to map
them, but the mapping is done once instead of per-op. Most impactful with many small I/O ops.
Note: on kernel 5.14 this works but 5.19+ has improvements for large buffer registration.

**`batch_size`** (default: equals queue_depth)
Controls how many SQEs we fill before calling `io_uring_submit()`. When we have more I/O ops
than the queue_depth (e.g., 3200 blocks but queue_depth=256), we can't submit them all at once.
The batch_size determines the chunking strategy: fill `batch_size` SQEs, submit, reap available
completions, repeat. A smaller batch_size means more frequent submit/reap cycles (more
responsive, lower latency per op) but more syscalls. A larger batch_size means fewer cycles but
the kernel gets bigger batches to optimize. Setting it equal to queue_depth is the simplest
strategy — fill the ring completely, submit, drain, repeat. An interesting alternative is to
submit partial batches and interleave submission with reaping, keeping the ring always partially
full so the device never goes idle. This is one of the strategies OpenEvolve could explore.

---

## Python Wrapper Design (`iouring_backend.py`)

The backend detects two failure modes at import time:
- Extension not built (`ImportError`) — `iouring_ext.so` missing
- Kernel blocks io_uring (`RuntimeError`) — `kernel.io_uring_disabled != 0`

Both set `IOURING_AVAILABLE = False` with a warning message. Detection uses a lightweight
`iouring_probe()` C++ function that tries `io_uring_queue_init()` + `io_uring_queue_exit()`.

```python
import time

IOURING_AVAILABLE = False
try:
    import iouring_ext
    iouring_ext.iouring_probe()
    IOURING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: io_uring extension not available: {e}")
except RuntimeError as e:
    print(f"Warning: io_uring blocked by kernel: {e}")

# Individual parameter setters — one per tunable, for Optuna search space compatibility
def set_queue_depth(depth):
    iouring_ext.set_queue_depth(depth)

def set_use_sqpoll(enable):
    iouring_ext.set_use_sqpoll(enable)

def set_use_direct(enable):
    iouring_ext.set_use_direct(enable)

async def iouring_write_blocks(block_size, buffer, block_indices, dest_files):
    start = time.perf_counter()
    success = iouring_ext.iouring_write_blocks(buffer, block_size, block_indices, dest_files)
    end = time.perf_counter()
    if not success:
        print("Writing blocks with io_uring failed")
        return
    return end - start

async def iouring_read_blocks(block_size, buffer, block_indices, dest_files):
    start = time.perf_counter()
    success = iouring_ext.iouring_read_blocks(buffer, block_size, block_indices, dest_files)
    if not success:
        print("Reading blocks with io_uring failed")
        return
    end = time.perf_counter()
    return end - start
```

---

## Build Setup (`setup_iouring.py`)

- Separate build script: `python setup_iouring.py build_ext --inplace`
- Uses `CppExtension` from `torch.utils.cpp_extension` (same as existing setup.py)
- Links against `-luring` (liburing must be installed on the build machine)
- Same compiler flags: `-std=c++17 -O3 -march=native -fPIC`
- Source: `backends/benchmark_iouring_utils/iouring_utils.cpp`

### liburing dependency

On the remote LSF cluster:
- Check: `dpkg -l | grep liburing` or `rpm -qa | grep liburing`
- Install if missing: `sudo apt install liburing-dev` or build from source (https://github.com/axboe/liburing)
- Header: `<liburing.h>`, link flag: `-luring`

---

## Integration into Benchmark Core

### `benchmark_core.py` changes

Add `iouring` branch in `run_benchmark_iteration()` — identical structure to the `cpp` branch:

```python
elif implementation == "iouring":
    time_write = await iouring_write_blocks(block_size, buffer, blocks_indices_write, file_names)
    verify_op(block_size, blocks_indices_write, view, file_names, "Writing", verify)
    await iouring_read_blocks(block_size_cleaning, buffer_cleaning, indices_cleaning, file_names_cleaning)
    time_read = await iouring_read_blocks(block_size, buffer, blocks_indices_read, file_names)
    verify_op(block_size, blocks_indices_read, view, file_names, "Reading", verify)
    await iouring_read_blocks(block_size_cleaning, buffer_cleaning, indices_cleaning, file_names_cleaning)
```

Same pattern for `run_concurrent_benchmark_iteration()`.

### `setup_executor()` / `shutdown_executor()`

io_uring manages its own concurrency (kernel-side), so like `cpp`, it returns `None` for executor. But we still need a "thread count equivalent" — this maps to `queue_depth` for io_uring. The `set_iouring_params()` call replaces `set_thread_count_cpp()`.

### `compare_file_operations.py`

- Add `'iouring'` to `choices` list in `--backend` argument
- Add availability check block (same pattern as cpp)

---

## Implementation Order & Status

1. ~~**Scaffold**: Create directory structure, empty files, build script~~ **DONE**
2. ~~**Read path**: Implement `iouring_read_blocks()` in C++~~ **DONE**
3. ~~**Build & smoke test**: Build on remote~~ **DONE** (builds OK, but io_uring blocked on cluster — see Blockers)
4. ~~**Write path**: Implement `iouring_write_blocks()` with atomic temp-rename pattern~~ **DONE**
5. **Integration**: Wire into `benchmark_core.py` and `compare_file_operations.py` — **NEXT**
6. **Baseline comparison**: Run `iouring` vs `cpp` on `/dev/shm` with same parameters
7. **Tuning knobs**: Add SQPOLL, O_DIRECT, registered files/buffers as toggleable options
8. **Real storage test**: Run on actual filesystem (Spectrum Scale / NFS / whatever the cluster has)

### Blockers

- **io_uring disabled on LSF cluster**: `kernel.io_uring_disabled = 2` (fully disabled for all users).
  Login nodes and compute nodes both block `io_uring_queue_init()` with `EPERM`.
  Need a machine with sudo access to set `sysctl kernel.io_uring_disabled=0`.
- **Machine requirements**: Linux kernel 5.6+, any NVIDIA GPU (for pin_memory), sudo, 64GB+ RAM, ideally local NVMe.

### What's Ready to Resume

- All C++ code written and compiles: `iouring_utils.cpp` with modular strategy functions
- Python wrapper ready: `iouring_backend.py`
- Build script ready: `setup_iouring.py` (auto-detects liburing at `$HOME/.local`)
- liburing built and installed on LSF cluster at `$HOME/.local`
- Smoke test script: `test_iouring.py` — run on a machine with io_uring enabled
- Next step after successful smoke test: integrate into `benchmark_core.py` and `compare_file_operations.py`

---

## Linux Kernel Requirements

- io_uring: Linux 5.1+ (basic), 5.6+ (SQPOLL, registered files), 5.19+ (registered buffers improvements)
- **Cluster kernel: 5.14** — confirmed. Supports all features except 5.19 registered buffer improvements.
- **io_uring enabled**: confirmed via `cat /proc/kallsyms | grep io_uring_setup`

---

## Resolved Questions

- **Kernel version**: 5.14 — SQPOLL, registered files all supported.
- **liburing**: Not in system packages. Built from source into `$HOME/.local` (shared across LSF nodes via shared home filesystem). Build added to `setup_env.sh` with idempotency guard.

## Open Questions

- **O_DIRECT alignment**: PyTorch pinned memory may already be page-aligned, but need to verify. If not, we'll need `posix_memalign()` for a separate aligned buffer and memcpy in/out.
- **SQPOLL permissions**: SQPOLL may require `CAP_SYS_ADMIN` or `IORING_SETUP_SQPOLL` with elevated privileges. Need to test on the cluster.

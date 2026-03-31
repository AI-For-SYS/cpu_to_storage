# I/O Optimization Presentation Plan

## Slide 1 — Why Offload KV Cache to Storage?

### Content
- KV cache stores pre-computed attention keys/values — recalculating requires a full prefill pass (expensive)
- GPU memory is limited and shared with model weights + activations — KV cache is the first thing to evict
- **Three-tier memory hierarchy for KV cache:**

```
GPU memory  →  CPU memory  →  Storage
(fastest,       (fast,          (slowest,
 smallest)       larger)         largest, persistent)
```

  - **GPU → CPU**: already happens in LLMD — standard first-level eviction
  - **CPU → Storage**: second-level eviction / persistence — **this is what we're optimizing**

- **GPU → Storage directly (GDS/GPUDirect Storage) — preferred when available, but often isn't:**
  - GDS is the fastest path — bypasses CPU entirely via DMA/RDMA
  - But: requires specific NVIDIA hardware + compatible NIC + storage drivers
  - Works best with local NVMe; on shared/network filesystems (Spectrum Scale, Lustre) often unsupported or falls back to CPU path
  - Ties up GPU resources during transfer — GPU can't do inference while doing I/O
  - Not available in all deployment environments (cloud, heterogeneous clusters)
  - **When GDS isn't available → the CPU path is the fallback, and it needs to be fast**

- Why the CPU ↔ Storage leg matters:
  - CPU memory is finite — when it fills up, overflow must go to storage
  - Persistence — CPU memory doesn't survive process restarts, migration, or failures
  - CPU handles I/O in the background while GPU continues inference — better overall utilization
- Key scenarios where the CPU ↔ Storage path is needed:
  1. **New decode instance joins** — cache is empty, loading from storage is faster than recalculating
  2. **Cache evicted from all GPUs** — save to storage before eviction, reload later instead of recalculating
  3. **Preemption / migration** — request paused or moved to another GPU; persist cache for seamless resume
- Read path (loading cache) is more latency-sensitive than write path (offload can be backgrounded)

### Speaker Notes
- The core tradeoff: recalculating KV cache requires a full forward pass through the model (prefill), which is expensive in both compute and latency. If we can reload from storage faster than recalculating, we save GPU cycles and reduce time-to-first-token.
- **The three-tier hierarchy:** GPU → CPU eviction already exists in LLMD and is fast (PCIe/NVLink bandwidth). Our project focuses on the second leg: CPU ↔ Storage. This leg is triggered when: (a) CPU memory itself is under pressure and must overflow, or (b) the cache needs to persist beyond process lifetime (restarts, migrations, failures).
- **CPU memory as hot cache:** In many cases, KV blocks evicted from GPU live in CPU memory and are reloaded to GPU without ever touching storage. Storage is the cold tier — only needed when CPU memory is full or when persistence is required. This means the CPU→Storage write path can often be backgrounded (not urgent), but the Storage→CPU read path is latency-critical when it does happen (cache miss in CPU memory).
- **GPU → Storage directly (GDS) — preferred when available, but often isn't:**
  GPUDirect Storage (GDS) is the fastest path — bypasses CPU entirely via DMA/RDMA. When available, it's preferred. But it has significant availability limitations:
  - **Hardware requirements:** GDS requires a specific NVIDIA GPU + Mellanox NIC + compatible storage controller + MLNX_OFED drivers. Many production deployments don't have the full stack.
  - **Filesystem compatibility:** GDS works best with local NVMe. On network/shared filesystems (Spectrum Scale, Lustre, GPFS), GDS is often unsupported or silently falls back to the CPU-bounce path anyway — giving you worse performance than a well-optimized CPU path because of the fallback overhead.
  - **GPU utilization tradeoff:** While GDS transfers data, the GPU's DMA engine is busy. For inference serving, every microsecond of GPU time matters. The CPU path lets the GPU keep doing inference while CPU and storage handle I/O in the background — better end-to-end throughput.
  - **Portability:** Different clouds, on-prem clusters, and edge deployments have different hardware. The CPU path works everywhere with zero special requirements — just Linux + any filesystem.
  - **In practice, both paths will coexist.** GDS for specific cases where hardware is available and latency is critical (e.g., local NVMe on a DGX). CPU path as the general-purpose layer that works across all deployments. Optimizing the CPU path has broad impact because it applies everywhere, not just on specific hardware configurations.
- Additional scenarios (not on slide but can mention if asked): long-context conversations (KV cache grows beyond GPU memory, offload older tokens to storage), batch size scaling (more concurrent requests = more KV caches competing for GPU memory, overflow to storage).
- In all scenarios, the read path (storage → CPU → GPU) is on the critical path — it directly impacts latency the user sees. Write path can be async/backgrounded.

---

## Slide 2 — Benchmark Project

### Content
- Benchmark project by [team name] — framework for evaluating I/O strategies for KV cache offload
- **4 backends tested:**
  - **C++** — pread/pwrite + custom thread pool, GIL released, serialized file creation
  - **Python self** — asyncio + ThreadPoolExecutor + os.readv/write
  - **aiofiles** — pure Python async I/O library
  - **NIXL** — NVIDIA library with POSIX backend, memory registration, descriptor-based transfers
- **Test matrix:**
  - Block sizes: 2, 4, 8, 16, 32, 64 MB (KV cache blocks are ~5-50MB depending on model)
  - Thread counts: 16, 32, 64
  - Modes: sequential write/read, fixed total data (100GB), concurrent read+write
- **Environment:** K8s pods, 450Gi RAM, /dev/shm (tmpfs) + persistent storage
- All backends use atomic writes (temp file + rename) and cache cleaning between phases

### Speaker Notes
- Credit the other team's work — solid foundation, well-structured benchmarks
- Block size depends on model: LLaMA-70B ~52MB, LLaMA-13B ~13MB, quantized ~3-7MB. The tested range (2-64MB) covers realistic KV cache block sizes.
- Cache cleaning between phases: reads 100GB of unrelated data to evict written data from OS page cache. This ensures we measure true storage throughput, not cached reads. Critical for meaningful results.
- Atomic write pattern (temp file + rename): prevents partial/corrupt files if process crashes mid-write. All 4 backends implement this.
- The benchmark framework is reusable — we'll add new backends and parameters to it for our optimization work
- **NIXL** (NVIDIA Interconnect eXchange Library) — an NVIDIA library designed for high-performance data transfers, primarily GPU-to-GPU and RDMA across nodes. It supports a POSIX file backend, which is what's tested here. The library uses explicit memory registration (pinning regions for DMA) and a descriptor-based transfer model. File I/O isn't its primary use case — it's optimized for network/GPU transfers. The busy-polling completion model also wastes CPU cycles compared to blocking reads. We included it because it's being evaluated within the LLMD ecosystem for other transfer paths.
- **aiofiles** — a pure Python async I/O library. Cross-platform, easy to use, but adds abstraction overhead on top of the OS. Included as a baseline to show the cost of high-level abstractions vs native implementations. Not expected to compete with C++ or even the Python self-implementation.

---

## Slide 3 — Current Results (baseline)

### Content
- The 3-way comparison plot (cpp vs python_self vs nixl, 100GB data mode, /dev/shm)
- Key numbers table:

| Backend | Peak Read | Peak Write | Notes |
|---|---|---|---|
| C++ | ~55-60 GB/s | ~40-45 GB/s | Best overall |
| Python self | ~45-55 GB/s | ~25-35 GB/s | GIL limits write |
| NIXL | ~15-20 GB/s | ~10-15 GB/s | Busy-poll overhead |

- Key observations:
  - Throughput plateaus at 16-32MB block size
  - Diminishing returns beyond 32 threads
  - C++ wins because: GIL released, pread/pwrite with custom thread pool, serialized file creation avoids directory contention
- **This is our baseline — the C++ backend using standard POSIX pread/pwrite**

### Speaker Notes
- tmpfs (/dev/shm) — a filesystem that lives entirely in RAM, not on any physical storage device. Reads and writes go directly to/from memory, no disk involved. This gives us the theoretical upper bound for I/O throughput — any real storage (NVMe, Spectrum Scale) will be slower. We benchmark on tmpfs first to measure software overhead in isolation, without storage hardware being the bottleneck. On real storage, the same software optimizations will have proportionally larger impact since there's more overhead to eliminate.
- Explain why C++ beats Python: GIL release during I/O, native thread pool vs Python's ThreadPoolExecutor overhead
- NIXL (NVIDIA Interconnect eXchange Library) — an NVIDIA library designed for high-performance data transfers, primarily GPU-to-GPU and RDMA across nodes. It supports a POSIX file backend, which is what's tested here. The library uses explicit memory registration (pinning regions for DMA) and a descriptor-based transfer model. It underperforms in this benchmark because file I/O isn't its primary use case — it's optimized for network/GPU transfers. The busy-polling completion model also wastes CPU cycles compared to blocking reads. We included it because it's being evaluated within the LLMD ecosystem for other transfer paths.
- The plateau at 16-32MB suggests we're hitting syscall overhead and page cache copy limits, not device bandwidth — room to improve
- Mention these are averages over 5 iterations with 100GB cache cleaning between write and read phases

---

## Slide 4 — How Linux I/O Works & Where We Can Improve

### Content
- Diagram: data path for a pread call

```
User Buffer → syscall (context switch) → Page Cache (kernel copy) → Block Layer → Device Driver → Storage
```

- Current approach (pread/pwrite) hits **every layer** on this path
- Three places to optimize:
  1. **Syscall overhead** — each pread = one context switch. With 1000+ blocks = 1000+ syscalls
  2. **Page cache copy** — kernel copies data from device into page cache, then from page cache into user buffer. Double copy.
  3. **Scheduling** — kernel doesn't know our access pattern, can't optimize I/O order
- What each optimization targets:
  - **io_uring** → eliminates #1. Shared ring buffer between user and kernel — submit batches of I/O ops without syscalls. Used by high-performance databases (ScyllaDB, TigerBeetle). Available since Linux 5.1.
  - **O_DIRECT** → eliminates #2. Data goes straight from storage device to user buffer, skipping page cache entirely. Requires 4K-aligned buffers.
  - **posix_fadvise** → helps #3. Hints to kernel: SEQUENTIAL enables readahead, WILLNEED triggers prefetch. Nearly free to add.
  - **Combined (io_uring + O_DIRECT)** → eliminates all three. The theoretical ceiling for Linux I/O performance.

### Speaker Notes
- **User Buffer** — this is your application's memory where the KV cache data lives. In our case, a pinned PyTorch tensor. "Pinned" means we asked the OS to never swap it to disk — it stays in physical RAM. This is where data starts (for writes) or ends up (for reads).

- **Syscall (system call)** — when your program calls `pread()`, it can't talk to the storage device directly. It has to ask the kernel to do it. This involves a "context switch" — the CPU saves your program's state, switches to kernel mode, does the work, then switches back. Think of it like going through a security checkpoint every time you want to access the warehouse. Each checkpoint takes ~1-2 microseconds. Sounds tiny, but with 1000 blocks that's 1-2 milliseconds of pure overhead just for "permission to enter."

- **Page Cache** — the kernel doesn't read from disk directly into your buffer. It reads into its own memory area first (the page cache), then copies to your buffer. Why? Because if another program (or you) reads the same file again, the kernel can serve it from cache instantly. It's like a library that photocopies every book you request — you get the photocopy, the original stays on the shelf for the next person. Great for general use, but for our workload we write each KV block once and read it once. We're paying for a photocopy we'll never reuse.

- **Block Layer** — the kernel's I/O scheduler. It receives I/O requests and decides what order to send them to the device. It can merge adjacent requests, reorder for optimal seek patterns, or batch submissions. The problem: it doesn't know our access pattern unless we tell it.

- **Device Driver → Storage** — the actual hardware communication. NVMe drives have their own internal queue (up to 64K outstanding commands). The faster we can fill this queue, the more the drive can parallelize internally. With pread + threads, each thread submits one request at a time. With io_uring, we can flood the device queue with hundreds of requests in one batch.

- **io_uring explained** — traditional I/O requires a syscall per operation: your program calls pread(), CPU switches to kernel mode, kernel does the I/O, switches back. io_uring replaces this with two shared memory ring buffers between your program and the kernel: a submission queue (SQ) where you drop I/O requests, and a completion queue (CQ) where the kernel drops results. You can fill the SQ with hundreds of read requests and notify the kernel once — or with SQPOLL mode, a dedicated kernel thread watches the SQ continuously, so you don't even need that one notification. The kernel then processes all requests with optimal ordering and fills the CQ as they complete. Your program polls the CQ for results. No context switches per operation, no per-request syscall overhead. This is why databases like ScyllaDB and TigerBeetle use it — it was designed specifically for high-throughput I/O workloads. Available since Linux 5.1 (2019), mature and well-tested.

- **O_DIRECT explained** — normally when you read a file, data travels: storage → kernel page cache → your buffer (two copies). O_DIRECT tells the kernel "skip your cache, transfer directly into my buffer." This eliminates one full memcpy — for a 50MB block, that's 50MB of unnecessary copying saved. The tradeoff: your buffer must be aligned to 4K boundaries (the storage sector size), and you lose the benefit of the page cache for repeated reads. For our workload (read once into KV cache) there's no benefit to caching, so O_DIRECT is pure win. Important caveat: O_DIRECT doesn't apply on tmpfs since tmpfs is already RAM — this optimization only matters on real storage devices (NVMe, Spectrum Scale).

- **posix_fadvise explained** — a single syscall after opening a file that tells the kernel how you plan to access it. `POSIX_FADV_SEQUENTIAL` says "I'm reading front to back" — the kernel doubles or quadruples its readahead window, pre-loading data before you ask for it. `POSIX_FADV_WILLNEED` says "I'll need this data soon, start loading it now" — the kernel begins background I/O immediately. For our workload, we know exactly which blocks we'll need, so these hints let the storage device start working before our pread() call happens. The kernel may ignore hints under memory pressure, but they cost essentially nothing to provide. Currently our code gives zero hints — the kernel is guessing our pattern.

- **Why combined is the ceiling** — io_uring + O_DIRECT together means: no syscall overhead (ring buffer), no page cache copy (direct transfer), and the kernel optimally schedules all requests. The only remaining bottleneck is the physical storage device itself. Everything above the device in the software stack is either eliminated or minimized.

- **Analogy for the full picture:** Current approach (pread): You go to a warehouse. For each box you need, you walk to the security desk (syscall), they photocopy your request (page cache), then fetch the box. 1000 boxes = 1000 trips to the security desk. io_uring + O_DIRECT: You hand the warehouse a list of all 1000 boxes at once (io_uring). They deliver directly to your truck without photocopying (O_DIRECT). One trip, no unnecessary copies.

---

## Slide 5 — Optimization Approaches

### Content

**A. Optimize Current Approach (existing C++ backend)**

| Parameter | What it does | Effort |
|---|---|---|
| O_NOATIME | Skip access time metadata update on every read | 1 line |
| posix_fadvise | Tell kernel our access pattern (sequential / prefetch) | 2-3 lines |
| I/O chunk size | Tune pread/pwrite size within a block | Small change |
| Prefetch depth | fadvise(WILLNEED) on next N blocks while reading current one | Small change |
| Thread count range | Currently hardcoded [16,32,64] — open up to test 8-256 | 1 line |

Expected gain: ~5-15% with right combination. Minimal effort, no new code architecture.

**B. New I/O Mechanisms**

| Approach | What it does | Expected read gain | Effort |
|---|---|---|---|
| io_uring | Batch syscalls, kernel-side scheduling | ~10-20% | New backend |
| O_DIRECT | Bypass page cache | ~5-15% | Flag + alignment |
| io_uring + O_DIRECT | All combined | ~15-30% | New backend |

### Speaker Notes

**Section A — parameter explanations:**

- **O_NOATIME** — every time Linux opens a file for reading, it updates the file's "last accessed" timestamp in the filesystem metadata. This is a small write operation for every read — completely useless for our workload. `O_NOATIME` flag tells the OS to skip this update. It's one character added to the `open()` call, zero downside.

- **posix_fadvise** — after opening a file, we can tell the kernel how we plan to read it. `POSIX_FADV_SEQUENTIAL` tells it "I'm reading front to back" — the kernel enables aggressive readahead, pre-loading the next chunks before we ask. `POSIX_FADV_WILLNEED` says "I need this data soon, start loading it into page cache now." For our workload where we know which blocks we'll need, this lets the storage device start working before our `pread()` call even happens. Currently our code gives the kernel zero hints — it's guessing.

- **I/O chunk size** — our current code calls `pread(fd, buffer, 50MB, offset)` — asking the kernel to read the entire block in one syscall. But the kernel may internally break this into smaller transfers. Different storage systems have different optimal transfer sizes — a GPFS filesystem with 4MB block size might perform better with 4MB-aligned reads than one giant 50MB read. A tunable chunk size (e.g., 256KB, 1MB, 4MB) within the pread loop lets us find what the storage controller handles most efficiently.

- **Prefetch depth** — while we're reading block N, we call `posix_fadvise(WILLNEED)` on blocks N+1, N+2, ..., N+K. This tells the kernel to start loading those blocks from storage into memory while we're still processing the current one. The depth K controls how far ahead we look. Too small: the next block isn't ready when we need it. Too large: we waste memory loading blocks we won't need for a while. The optimal value depends on how fast the storage is relative to our processing — faster storage needs less prefetch.

- **Thread count range** — currently hardcoded to test [16, 32, 64]. But the optimal thread count depends on the storage. A local NVMe might peak at 32 threads, while a parallel filesystem like Spectrum Scale (where data is striped across many servers) could benefit from 128 or 256 threads to keep all servers busy. Conversely, on a slow device, too many threads cause contention and hurt performance — 8 might be better than 64. We need to open up the search range to find the real optimum.

**Section B — new mechanisms:**

- io_uring is what high-performance databases use (ScyllaDB, TigerBeetle). New backend but reuses existing benchmark framework for direct comparison. It also introduces its own tunable parameters (queue depth, SQPOLL, IOPOLL, registered buffers/files) — roughly doubling our parameter space, which further motivates auto-tuning.
- O_DIRECT caveat: needs 4K-aligned buffers, doesn't work on tmpfs. Must test on actual target storage.

---

## Slide 6 — Auto-Tuning & Roadmap

### Content

**The tuning problem:**
- Across existing + new backends: ~15 tunable parameters
- Thread count, block size, I/O chunk size, queue depth, O_DIRECT, O_NOATIME, fadvise hint, prefetch depth, io_uring SQ size, SQPOLL, IOPOLL, registered buffers/files, submission batch size
- Optimal values depend on target hardware and interact non-trivially
- Manual tuning doesn't scale across deployment targets

**Tuning methods:**

| Method | What it tunes | Best for |
|---|---|---|
| Bayesian (Optuna) | All parameters | Per-machine optimal config |
| Evolutionary (OpenEvolve) | Parameters + code strategies | Discovering new I/O patterns |

**Recommended roadmap:**
1. Add tunable parameters + POSIX hints to existing C++ backend *(quick win)*
2. Build io_uring + O_DIRECT backend *(main investment)*
3. Bayesian auto-tuning per deployment target *(optimal config)*
4. Evolutionary search for code-level strategies *(exploration)*

### Speaker Notes
- Key insight: we don't need to understand every hardware detail. Build the parameter space, let the search find what works.
- Example of non-obvious interaction: O_DIRECT might help at 64 threads but hurt at 16. Or chunk size 1MB wins with SEQUENTIAL hint but 4MB wins with WILLNEED. You'd never find this manually.
- **Bayesian optimization** is a method for finding the best values of parameters when each evaluation is expensive (like running a benchmark). After each trial, it builds a statistical model (a "surrogate") that predicts which parameter combinations are likely to give good results. It then picks the most promising untried combination — balancing between exploring new regions and exploiting what already looks good. This is much more efficient than random or grid search because it learns from every trial. **Optuna** is a popular Python library that implements this. You define your parameter ranges and a function that runs the benchmark and returns throughput. Optuna handles the rest — choosing what to try next, tracking results, and providing visualizations of which parameters mattered most.
- **Evolutionary search (OpenEvolve)** — the key question is: why do we need code evolution here, not just parameter tuning? Because some of the most impactful optimizations aren't expressible as numbers. For example: should each thread read one file, or should one thread read multiple files sequentially to benefit from filesystem readahead? Should we sort blocks by physical location before issuing reads, or keep the original order? Should file creation be fully serialized (current approach), pipelined in batches, or interleaved with writes? These are branching code paths, not slider values. Bayesian search can't explore them — it needs a fixed code structure with numeric/categorical knobs. Evolutionary search can generate and test structurally different implementations, combine successful patterns from different variants, and discover strategies we wouldn't think to try manually. Given that the current C++ backend was hand-optimized and is already quite good, incremental parameter tuning may hit a ceiling. Code-level evolution is how we might break past it.
- Roadmap step 1 is low effort — enables all later work. Step 2 is the core engineering. Step 3 makes it production-ready for any hardware. Step 4 is research/exploration.
- End with: "build once, auto-tune everywhere" — same framework works whether target is local NVMe, Spectrum Scale, or future storage

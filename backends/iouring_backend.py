import json
import time
from dataclasses import dataclass, asdict

IOURING_AVAILABLE = False
try:
    import iouring_ext
    iouring_ext.iouring_probe()
    IOURING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: io_uring extension not available: {e}")
    print("Run 'python setup_iouring.py build_ext --inplace' to build it")
except RuntimeError as e:
    print(f"Warning: io_uring blocked by kernel: {e}")


# Individual parameter setters for Optuna search space compatibility
def set_queue_depth(depth):
    iouring_ext.set_queue_depth(depth)

def set_batch_size(size):
    iouring_ext.set_batch_size(size)

def set_iowq_max_workers(bounded, unbounded):
    """Override io_uring iowq worker-pool caps. Pass 0 to keep kernel default.
    Raising `bounded` is the main lever for buffered file I/O throughput on
    older kernels (the default is often a handful of workers, which serializes
    our parallel writes)."""
    iouring_ext.set_iowq_max_workers(int(bounded), int(unbounded))

def set_force_async(enable):
    """When True, every SQE gets IOSQE_ASYNC — forces iowq dispatch instead of
    inline completion. May unlock parallelism for buffered I/O where the
    default 'inline-first' path serializes ops inside a single submit syscall."""
    iouring_ext.set_force_async(bool(enable))


async def iouring_write_blocks(block_size, buffer, block_indices, dest_files):
    """io_uring implementation wrapper for writing blocks.

    Opens all temp-file FDs sequentially (serialized O_CREAT),
    then batches pwrite ops through the io_uring submission queue.
    """
    start = time.perf_counter()
    success = iouring_ext.iouring_write_blocks(
        buffer, block_size, block_indices, dest_files
    )
    end = time.perf_counter()

    if not success:
        print("Writing blocks with io_uring failed")
        return
    return (end - start)


async def iouring_read_blocks(block_size, buffer, block_indices, dest_files):
    """io_uring implementation wrapper for reading blocks.

    Batches pread ops through the io_uring submission queue.
    """
    start = time.perf_counter()
    success = iouring_ext.iouring_read_blocks(
        buffer, block_size, block_indices, dest_files
    )
    if not success:
        print("Reading blocks with io_uring failed")
        return
    end = time.perf_counter()
    return (end - start)


@dataclass
class IouringTunableConfig:
    queue_depth: int = 256
    batch_size: int = 256
    block_size_mb: int = 0                  # 0 = use benchmark default
    # iowq worker-pool caps: 0 = leave kernel default. Raising `bounded` is
    # the primary lever for buffered file I/O throughput on older kernels.
    # Note: kernel must be >= 5.15 for this register to take effect.
    iowq_bounded:   int = 0
    iowq_unbounded: int = 0
    # Force IOSQE_ASYNC on every SQE — routes ops through iowq workers instead
    # of attempting inline completion. For buffered I/O this unlocks parallelism
    # (measured ~6x improvement). Default ON; set False only for experimentation.
    force_async: bool = True
    # Frozen at defaults for the first Optuna pass — C++ setters still pending.
    # See docs/iouring_implementation_plan.md (Tunable Parameters section).
    use_sqpoll: bool = False
    use_direct: bool = False
    use_registered_files: bool = False
    use_registered_buffers: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IouringTunableConfig":
        return cls(
            queue_depth=d.get("queue_depth", 256),
            batch_size=d.get("batch_size", 256),
            block_size_mb=d.get("block_size_mb", 0),
            iowq_bounded=d.get("iowq_bounded", 0),
            iowq_unbounded=d.get("iowq_unbounded", 0),
            force_async=d.get("force_async", True),
            use_sqpoll=d.get("use_sqpoll", False),
            use_direct=d.get("use_direct", False),
            use_registered_files=d.get("use_registered_files", False),
            use_registered_buffers=d.get("use_registered_buffers", False),
        )


def configure(config: IouringTunableConfig):
    """Apply typed config to the iouring C++ backend.

    Forwards queue_depth, batch_size, iowq_max_workers, and force_async. The
    remaining flags live on the Python config for Optuna search-space symmetry,
    but their C++ setters are not yet implemented. Unfreezing any of them
    requires adding the corresponding C++ setter first.
    """
    set_queue_depth(config.queue_depth)
    set_batch_size(config.batch_size)
    set_iowq_max_workers(config.iowq_bounded, config.iowq_unbounded)
    set_force_async(config.force_async)


def save_iouring_tunable_configs(path: str,
                                 write_config: IouringTunableConfig,
                                 read_config: IouringTunableConfig,
                                 concurrent_config: IouringTunableConfig,
                                 metadata: dict = None):
    """Save per-mode best configs (write, read, concurrent) to a single JSON file."""
    d = {
        "write": write_config.to_dict(),
        "read": read_config.to_dict(),
        "concurrent": concurrent_config.to_dict(),
    }
    if metadata:
        d["_metadata"] = metadata
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)


def load_iouring_tunable_configs(path: str) -> dict:
    """Load per-mode configs from a multi-mode JSON file.

    Returns dict with keys 'write', 'read', 'concurrent',
    each containing an IouringTunableConfig. Also returns '_metadata' if present.
    """
    with open(path) as f:
        d = json.load(f)
    configs = {}
    for mode in ("write", "read", "concurrent"):
        if mode in d:
            configs[mode] = IouringTunableConfig.from_dict(d[mode])
    configs["_metadata"] = d.get("_metadata", {})
    return configs

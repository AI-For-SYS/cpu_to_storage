import json
import time
from dataclasses import dataclass, asdict
from enum import Enum

THREADED_TUNABLE_AVAILABLE = False
try:
    import threaded_tunable_ext
    THREADED_TUNABLE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: threaded_tunable extension not available: {e}")
    print("Run 'python setup_threaded_tunable.py build_ext --inplace' to build it")


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
    thread_count: int = 0                              # 0 = auto-detect (hardware_concurrency)
    block_size_mb: int = 0                             # 0 = use benchmark default
    o_noatime: bool = False
    o_direct: bool = False
    fadvise_hint: FadviseHint = FadviseHint.NORMAL
    io_chunk_kb: int = 0                               # 0 = full block
    prefetch_depth: int = 0                             # 0 = disabled
    fallocate: bool = False
    sync_strategy: SyncStrategy = SyncStrategy.NONE
    cpu_affinity: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fadvise_hint"] = self.fadvise_hint.value
        d["sync_strategy"] = self.sync_strategy.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ThreadedTunableConfig":
        """Create config from a dict, converting strings to enums."""
        return cls(
            thread_count=d.get("thread_count", 0),
            block_size_mb=d.get("block_size_mb", 0),
            o_noatime=d.get("o_noatime", False),
            o_direct=d.get("o_direct", False),
            fadvise_hint=FadviseHint(d.get("fadvise_hint", "normal")),
            io_chunk_kb=d.get("io_chunk_kb", 0),
            prefetch_depth=d.get("prefetch_depth", 0),
            fallocate=d.get("fallocate", False),
            sync_strategy=SyncStrategy(d.get("sync_strategy", "none")),
            cpu_affinity=d.get("cpu_affinity", False),
        )


def save_tunable_configs(path: str, write_config: ThreadedTunableConfig,
                          read_config: ThreadedTunableConfig,
                          concurrent_config: ThreadedTunableConfig,
                          metadata: dict = None):
    """Save all three best configs (write, read, concurrent) to a single JSON file."""
    d = {
        "write": write_config.to_dict(),
        "read": read_config.to_dict(),
        "concurrent": concurrent_config.to_dict(),
    }
    if metadata:
        d["_metadata"] = metadata
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)


def load_tunable_configs(path: str) -> dict:
    """Load all three configs from a multi-mode JSON file.

    Returns dict with keys 'write', 'read', 'concurrent',
    each containing a ThreadedTunableConfig.
    Also returns '_metadata' if present.
    """
    with open(path) as f:
        d = json.load(f)
    configs = {}
    for mode in ("write", "read", "concurrent"):
        if mode in d:
            configs[mode] = ThreadedTunableConfig.from_dict(d[mode])
    configs["_metadata"] = d.get("_metadata", {})
    return configs


def configure(config: ThreadedTunableConfig):
    """Apply typed config to C++ backend."""
    threaded_tunable_ext.configure_all(config.to_dict())


def get_config() -> ThreadedTunableConfig:
    """Read current C++ config back as a dataclass."""
    raw = threaded_tunable_ext.get_config()
    return ThreadedTunableConfig(
        thread_count=raw["thread_count"],
        o_noatime=raw["o_noatime"],
        o_direct=raw["o_direct"],
        io_chunk_kb=raw["io_chunk_size"] // 1024 if raw["io_chunk_size"] > 0 else 0,
        prefetch_depth=raw["prefetch_depth"],
        fallocate=raw["fallocate_prealloc"],
        sync_strategy=SyncStrategy({0: "none", 1: "fdatasync", 2: "sync_file_range"}.get(raw["sync_strategy"], "none")),
        cpu_affinity=raw["cpu_affinity"],
    )


async def threaded_tunable_write_blocks(block_size, buffer, block_indices, dest_files):
    start = time.perf_counter()
    success = threaded_tunable_ext.threaded_tunable_write_blocks(
        buffer, block_size, block_indices, dest_files
    )
    end = time.perf_counter()

    if not success:
        print("Writing blocks with threaded_tunable Failed")
        return
    return (end - start)


async def threaded_tunable_read_blocks(block_size, buffer, block_indices, dest_files):
    start = time.perf_counter()
    success = threaded_tunable_ext.threaded_tunable_read_blocks(
        buffer, block_size, block_indices, dest_files
    )
    if not success:
        print("Reading blocks with threaded_tunable Failed")
        return
    end = time.perf_counter()
    return (end - start)

"""
Optuna auto-tuner for iouring backend.

Runs three sequential studies (write, read, concurrent) to find optimal
io_uring parameters for each operation on a given storage system.
Exports a single config file with best params for all three modes.

First-pass search space: queue_depth, batch_size, block_size_mb.
Frozen at defaults: use_sqpoll, use_direct, use_registered_files, use_registered_buffers
(C++ setters for these are not yet implemented; see docs/iouring_implementation_plan.md).

Usage:
    python optuna_tuner_iouring.py                              # full preset (default)
    python optuna_tuner_iouring.py --preset short               # quick sanity check
    python optuna_tuner_iouring.py --preset short --n-trials 5  # ultra-quick smoke (~2-4 min)
    python optuna_tuner_iouring.py --n-trials 500               # override trial count per mode

Requires: kernel.io_uring_disabled=0 (sudo sysctl -w if needed).
"""

import argparse
import asyncio
import os
import random
import sys
import time
from datetime import datetime

import optuna
import torch

from backends.iouring_backend import (
    IouringTunableConfig,
    configure as iouring_configure,
    iouring_write_blocks, iouring_read_blocks,
    save_iouring_tunable_configs,
    IOURING_AVAILABLE,
)
from utils.config import STORAGE_PATH, PIN_MEMORY
from utils.file_utils import generate_dest_file_names, clean_files

# ============================================================================
# Presets
# ============================================================================
PRESETS = {
    # short: pipeline validation only — ~1-2 min wall time. Cache is NOT fully
    # flushed (cleaning_gb=2), so reported GB/s are artificially high on systems
    # with >2GB free RAM. Use `full` for trustworthy numbers.
    "short": {
        "n_trials": 10,
        "timeout_per_mode": 300,
        "data_gb": 0.5,
        "buffer_gb": 1,
        "cleaning_gb": 2,
        "iterations": 1,
        "block_sizes_mb": [4, 8, 16, 32],
        "n_startup_trials": 5,
    },
    "full": {
        "n_trials": 250,
        "timeout_per_mode": None,
        "data_gb": 10,
        "buffer_gb": 100,
        "cleaning_gb": 100,
        "iterations": 3,
        "block_sizes_mb": [2, 4, 8, 16, 32, 64, 128],
        "n_startup_trials": 30,
    },
}

# ============================================================================
# Global state — allocated once, reused across all studies
# ============================================================================
_buffer = None
_buffer_cleaning = None
_cleaning_files = None
_cleaning_indices = None
_cleaning_block_size = None


def allocate_trial_buffers(buffer_gb, cleaning_gb):
    """Allocate buffers once for all trials.

    cleaning_gb controls how much unrelated data is read between trials to
    flush the OS page cache. For trustworthy read-throughput numbers on
    machines with >Nx GB free RAM, cleaning_gb should exceed that. The
    `short` preset uses a small value (e.g. 2GB) for pipeline validation
    only — reported GB/s will be inflated because the cache isn't flushed.
    """
    global _buffer, _buffer_cleaning
    global _cleaning_files, _cleaning_indices, _cleaning_block_size

    buffer_size = buffer_gb * 1024 * 1024 * 1024
    num_elements = buffer_size // 2  # float16 = 2 bytes

    print("Allocating buffers...")
    _buffer = torch.zeros(num_elements, dtype=torch.float16, device='cpu', pin_memory=PIN_MEMORY)

    _cleaning_block_size = 32 * 1024 * 1024  # 32MB per cleaning file
    num_cleaning_blocks = max(1, (cleaning_gb * 1024) // 32)   # cleaning_gb → N × 32MB blocks
    cleaning_size = num_cleaning_blocks * _cleaning_block_size
    cleaning_elements = cleaning_size // 2   # float16 = 2 bytes
    _buffer_cleaning = torch.zeros(cleaning_elements, dtype=torch.float16, device='cpu', pin_memory=PIN_MEMORY)

    _cleaning_files = [f"/dev/shm/cleaning_{j}.bin" for j in range(num_cleaning_blocks)]
    _cleaning_indices = list(range(num_cleaning_blocks))

    # Pre-write cleaning files
    print(f"Pre-writing cache cleaning files to /dev/shm ({cleaning_gb}GB)...")
    asyncio.get_event_loop().run_until_complete(
        iouring_write_blocks(
            _cleaning_block_size, _buffer_cleaning, _cleaning_indices, _cleaning_files
        )
    )
    print(f"Buffers allocated ({buffer_gb}GB main, {cleaning_gb}GB cleaning)\n")


async def _clean_cache():
    """Read cleaning files to evict page cache."""
    await iouring_read_blocks(
        _cleaning_block_size, _buffer_cleaning, _cleaning_indices, _cleaning_files
    )


# ============================================================================
# Trial benchmark functions — one per mode
# ============================================================================
async def _run_write_trial(block_size_mb, data_gb, iterations):
    """Write-only trial: measure write throughput."""
    block_size = block_size_mb * 1024 * 1024
    buffer_size = _buffer.numel() * _buffer.element_size()
    num_blocks = int((data_gb * 1024) / block_size_mb)
    max_block_index = buffer_size // block_size
    if num_blocks > max_block_index:
        num_blocks = max_block_index

    file_names = generate_dest_file_names("optuna_trial", num_blocks)
    times = []

    for i in range(iterations):
        clean_files(file_names)
        block_indices = random.sample(range(max_block_index), num_blocks)
        elapsed = await iouring_write_blocks(
            block_size, _buffer, block_indices, file_names
        )
        times.append(elapsed)

    clean_files(file_names)

    # Discard first iteration as warmup if we have more than 1
    if len(times) > 1:
        times = times[1:]
    return sum(times) / len(times)


async def _run_read_trial(block_size_mb, data_gb, iterations):
    """Read trial: write files first, clean cache, then measure read."""
    block_size = block_size_mb * 1024 * 1024
    buffer_size = _buffer.numel() * _buffer.element_size()
    num_blocks = int((data_gb * 1024) / block_size_mb)
    max_block_index = buffer_size // block_size
    if num_blocks > max_block_index:
        num_blocks = max_block_index

    file_names = generate_dest_file_names("optuna_trial", num_blocks)
    block_indices = random.sample(range(max_block_index), num_blocks)

    # Write files once (setup, not measured)
    clean_files(file_names)
    await iouring_write_blocks(
        block_size, _buffer, block_indices, file_names
    )

    times = []
    for i in range(iterations):
        await _clean_cache()
        elapsed = await iouring_read_blocks(
            block_size, _buffer, block_indices, file_names
        )
        times.append(elapsed)

    clean_files(file_names)

    if len(times) > 1:
        times = times[1:]
    return sum(times) / len(times)


async def _run_concurrent_trial(block_size_mb, data_gb, iterations):
    """Concurrent trial: simultaneous write + read."""
    block_size = block_size_mb * 1024 * 1024
    buffer_size = _buffer.numel() * _buffer.element_size()
    num_half = int((data_gb * 1024 / 2) / block_size_mb)
    max_block_index = buffer_size // block_size
    if num_half > max_block_index // 2:
        num_half = max_block_index // 2

    file_names_write = generate_dest_file_names("optuna_write", num_half)
    file_names_read = generate_dest_file_names("optuna_read", num_half)
    half_max = max_block_index // 2

    # Pre-write read files (setup)
    read_indices = random.sample(range(half_max, max_block_index), num_half)
    await iouring_write_blocks(
        block_size, _buffer, read_indices, file_names_read
    )

    times = []
    for i in range(iterations):
        await _clean_cache()
        clean_files(file_names_write)

        write_indices = random.sample(range(0, half_max), num_half)

        start = time.perf_counter()
        write_task = iouring_write_blocks(
            block_size, _buffer, write_indices, file_names_write
        )
        read_task = iouring_read_blocks(
            block_size, _buffer, read_indices, file_names_read
        )
        await asyncio.gather(write_task, read_task)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    clean_files(file_names_write)
    clean_files(file_names_read)

    if len(times) > 1:
        times = times[1:]
    return sum(times) / len(times)


def run_trial(mode, block_size_mb, data_gb, iterations):
    """Synchronous wrapper for async trial benchmarks."""
    loop = asyncio.get_event_loop()
    if mode == "write":
        return loop.run_until_complete(_run_write_trial(block_size_mb, data_gb, iterations))
    elif mode == "read":
        return loop.run_until_complete(_run_read_trial(block_size_mb, data_gb, iterations))
    elif mode == "concurrent":
        return loop.run_until_complete(_run_concurrent_trial(block_size_mb, data_gb, iterations))


# ============================================================================
# Optuna objective function
# ============================================================================
def create_objective(mode, data_gb, iterations, block_sizes_mb):
    """Create an objective function closure for the given mode."""

    def objective(trial: optuna.Trial) -> float:
        """Single Optuna trial. Returns throughput in GB/s (maximize)."""

        # --- Explored parameters (Optuna samples these) ---
        queue_depth = trial.suggest_int("queue_depth", 32, 1024, log=True)
        batch_size = trial.suggest_int("batch_size", 32, 1024, log=True)
        block_size_mb = trial.suggest_categorical("block_size_mb", block_sizes_mb)

        # --- Frozen parameters (defaults for first pass) ---
        # C++ setters for these are not yet implemented in iouring_utils.cpp.
        # See docs/iouring_implementation_plan.md (Tunable Parameters section).
        use_sqpoll = False
        use_direct = False
        use_registered_files = False
        use_registered_buffers = False

        config = IouringTunableConfig(
            queue_depth=queue_depth,
            batch_size=batch_size,
            block_size_mb=block_size_mb,
            use_sqpoll=use_sqpoll,
            use_direct=use_direct,
            use_registered_files=use_registered_files,
            use_registered_buffers=use_registered_buffers,
        )

        # Apply config
        try:
            iouring_configure(config)
        except RuntimeError as e:
            print(f"  [SKIP] Config error: {e}")
            raise optuna.TrialPruned()

        # Run benchmark
        try:
            elapsed = run_trial(mode, block_size_mb, data_gb, iterations)
        except Exception as e:
            print(f"  [FAIL] Trial error: {e}")
            raise optuna.TrialPruned()

        # Calculate throughput
        throughput = data_gb / elapsed

        # Log (only explored params)
        params_str = f"qd={queue_depth} batch={batch_size} block={block_size_mb}MB"
        print(f"  Trial {trial.number:4d} | {throughput:7.2f} GB/s | {params_str}")

        return throughput

    return objective


# ============================================================================
# Study runner and summary
# ============================================================================
def run_study(mode, preset, n_trials_override, timeout_override):
    """Run a single Optuna study for the given mode."""
    n_trials = n_trials_override or preset["n_trials"]
    timeout = timeout_override or preset["timeout_per_mode"]
    data_gb = preset["data_gb"]
    iterations = preset["iterations"]
    block_sizes_mb = preset["block_sizes_mb"]
    n_startup = preset["n_startup_trials"]

    print(f"\n{'='*60}")
    print(f"  Study: iouring_{mode} optimization")
    timeout_str = f"{timeout}s" if timeout else "none"
    print(f"  Trials: {n_trials}, Timeout: {timeout_str}, Data: {data_gb}GB, Iterations: {iterations}")
    print(f"{'='*60}\n")

    study = optuna.create_study(
        study_name=f"iouring_{mode}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=n_startup,
            multivariate=True,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup // 2,
            n_warmup_steps=0,
        ),
        storage="sqlite:///results/optuna_iouring_study.db",
        load_if_exists=True,
    )

    existing = len(study.trials)
    if existing > 0:
        print(f"  Resuming with {existing} existing trials\n")

    objective = create_objective(mode, data_gb, iterations, block_sizes_mb)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return study


def print_study_summary(study, mode):
    """Print summary for a single study."""
    best = study.best_trial
    print(f"\n  Best iouring_{mode} trial (#{best.number}): {best.value:.2f} GB/s")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\n  Parameter importance (fANOVA):")
        for i, (param, imp) in enumerate(importance.items(), 1):
            print(f"    {i}. {param:20s} {imp:.1%}")
    except Exception as e:
        print(f"  Parameter importance unavailable ({type(e).__name__}): {e}")


def extract_best_config(study):
    """Extract IouringTunableConfig from study's best trial."""
    params = study.best_trial.params
    return IouringTunableConfig(
        queue_depth=params.get("queue_depth", 256),
        batch_size=params.get("batch_size", 256),
        block_size_mb=params.get("block_size_mb", 0),
        use_sqpoll=params.get("use_sqpoll", False),
        use_direct=params.get("use_direct", False),
        use_registered_files=params.get("use_registered_files", False),
        use_registered_buffers=params.get("use_registered_buffers", False),
    )


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Optuna auto-tuner for iouring I/O backend"
    )
    parser.add_argument(
        "--preset", type=str, default="full", choices=["short", "full"],
        help="Preset config: short (20 trials/mode, 1GB) or full (250 trials/mode, 10GB)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override max trials per mode"
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Override max seconds per mode"
    )
    parser.add_argument(
        "--export-config", type=str, default=None,
        help="Path to save best config JSON (default: results/best_iouring_config_{timestamp}.json)"
    )
    args = parser.parse_args()

    if not IOURING_AVAILABLE:
        print("Error: iouring backend not available.")
        print("Check 'kernel.io_uring_disabled' (should be 0) and that iouring_ext is built.")
        sys.exit(1)

    preset = PRESETS[args.preset]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = args.export_config or f"results/best_iouring_config_{ts}.json"

    print("=" * 60)
    print("OPTUNA AUTO-TUNER (iouring)")
    print("=" * 60)
    print(f"Preset:          {args.preset}")
    print(f"Trials/mode:     {args.n_trials or preset['n_trials']}")
    timeout_val = args.timeout or preset['timeout_per_mode']
    print(f"Timeout/mode:    {f'{timeout_val}s' if timeout_val else 'none'}")
    print(f"Data/trial:      {preset['data_gb']}GB")
    print(f"Buffer:          {preset['buffer_gb']}GB")
    print(f"Cleaning:        {preset['cleaning_gb']}GB")
    print(f"Iterations:      {preset['iterations']}")
    print(f"Block sizes:     {preset['block_sizes_mb']} MB")
    print(f"Storage path:    {STORAGE_PATH}")
    print(f"Export config:   {export_path}")
    print("=" * 60)

    # Allocate buffers once for all studies
    allocate_trial_buffers(preset["buffer_gb"], preset["cleaning_gb"])
    os.makedirs("results", exist_ok=True)

    # Run all three studies sequentially
    studies = {}
    best_configs = {}
    metadata = {"storage_path": STORAGE_PATH, "backend": "iouring"}

    for mode in ("write", "read", "concurrent"):
        study = run_study(mode, preset, args.n_trials, args.timeout)
        studies[mode] = study
        best_configs[mode] = extract_best_config(study)

        best = study.best_trial
        metadata[f"{mode}_best_throughput_gbs"] = best.value
        metadata[f"{mode}_best_trial"] = best.number
        metadata[f"{mode}_total_trials"] = len(study.trials)

    # Print final summary
    print(f"\n{'='*60}")
    print("OPTIMIZATION COMPLETE (iouring)")
    print("=" * 60)
    for mode in ("write", "read", "concurrent"):
        print_study_summary(studies[mode], mode)

    # Export combined config
    save_iouring_tunable_configs(
        export_path,
        write_config=best_configs["write"],
        read_config=best_configs["read"],
        concurrent_config=best_configs["concurrent"],
        metadata=metadata,
    )
    print(f"\nConfig exported to: {export_path}")


if __name__ == "__main__":
    main()

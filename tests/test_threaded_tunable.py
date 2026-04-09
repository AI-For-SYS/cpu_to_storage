"""Smoke test for threaded_tunable_ext backend.

Verifies build, API compatibility with baseline cpp, data integrity,
and that each tunable parameter works without crashing.

Usage:
    python tests/test_threaded_tunable.py [--storage-path /path] [--tolerance 0.15]
"""
import sys
import os
import argparse
import time
import torch


def run_tests(storage_path, tolerance):
    passed = 0
    failed = 0
    skipped = 0

    print("=== threaded_tunable backend smoke test ===\n")

    # --- Test: Import ---
    try:
        import threaded_tunable_ext
        print("[PASS] Import threaded_tunable_ext")
        passed += 1
    except ImportError as e:
        print(f"[FAIL] Import threaded_tunable_ext: {e}")
        print("\nBuild with: python setup_threaded_tunable.py build_ext --inplace")
        return 1

    # --- Test: Import baseline cpp for comparison ---
    try:
        import cpp_ext
        has_baseline = True
    except ImportError:
        has_baseline = False
        print("[WARN] cpp_ext not available — skipping baseline comparison")

    from backends.threaded_tunable_backend import (
        ThreadedTunableConfig, FadviseHint, SyncStrategy, configure, get_config
    )

    # --- Test: configure_all / get_config roundtrip ---
    try:
        test_cfg = ThreadedTunableConfig(
            thread_count=16,
            o_noatime=True,
            o_direct=False,
            fadvise_hint=FadviseHint.SEQUENTIAL,
            io_chunk_kb=1024,
            prefetch_depth=4,
            fallocate=True,
            sync_strategy=SyncStrategy.FDATASYNC,
            cpu_affinity=False,
        )
        configure(test_cfg)
        read_back = get_config()
        assert read_back.thread_count == 16, f"thread_count: {read_back.thread_count}"
        assert read_back.o_noatime == True, f"o_noatime: {read_back.o_noatime}"
        assert read_back.io_chunk_kb == 1024, f"io_chunk_kb: {read_back.io_chunk_kb}"
        assert read_back.prefetch_depth == 4, f"prefetch_depth: {read_back.prefetch_depth}"
        assert read_back.fallocate == True, f"fallocate: {read_back.fallocate}"
        assert read_back.sync_strategy == SyncStrategy.FDATASYNC, f"sync_strategy: {read_back.sync_strategy}"
        print("[PASS] configure_all / get_config roundtrip")
        passed += 1
    except Exception as e:
        print(f"[FAIL] configure_all / get_config roundtrip: {e}")
        failed += 1

    # --- Setup test data ---
    num_blocks = 10
    block_size = 8 * 1024 * 1024  # 8MB
    num_elements = (num_blocks * block_size) // 2  # float16 = 2 bytes
    buffer = torch.randn(num_elements, dtype=torch.float16, device='cpu', pin_memory=False)
    block_indices = list(range(num_blocks))
    file_names = [os.path.join(storage_path, f"test_tunable_{i}.bin") for i in range(num_blocks)]

    def cleanup_files():
        for f in file_names:
            for path in [f, f + ".tmp"]:
                if os.path.exists(path):
                    os.remove(path)

    # --- Test: Defaults match baseline ---
    write_diff = 0.0
    read_diff = 0.0
    try:
        # Reset to defaults
        configure(ThreadedTunableConfig(thread_count=16))

        # Write with tunable
        start = time.perf_counter()
        ok = threaded_tunable_ext.threaded_tunable_write_blocks(
            buffer, block_size, block_indices, file_names
        )
        tunable_write = time.perf_counter() - start
        assert ok, "tunable write failed"

        # Read with tunable
        buf_read = torch.zeros_like(buffer)
        start = time.perf_counter()
        ok = threaded_tunable_ext.threaded_tunable_read_blocks(
            buf_read, block_size, block_indices, file_names
        )
        tunable_read = time.perf_counter() - start
        assert ok, "tunable read failed"

        # Verify data integrity
        assert torch.equal(buffer, buf_read), "Data mismatch after write/read!"

        if has_baseline:
            cpp_ext.set_thread_count(16)

            # Write with baseline
            cleanup_files()
            start = time.perf_counter()
            ok = cpp_ext.cpp_write_blocks(buffer, block_size, block_indices, file_names)
            cpp_write = time.perf_counter() - start
            assert ok, "cpp write failed"

            # Read with baseline
            buf_cpp = torch.zeros_like(buffer)
            start = time.perf_counter()
            ok = cpp_ext.cpp_read_blocks(buf_cpp, block_size, block_indices, file_names)
            cpp_read = time.perf_counter() - start
            assert ok, "cpp read failed"

            write_diff = abs(tunable_write - cpp_write) / cpp_write
            read_diff = abs(tunable_read - cpp_read) / cpp_read
            print(f"[PASS] Defaults write: {tunable_write:.4f}s (cpp: {cpp_write:.4f}s, diff: {write_diff:+.1%})")
            print(f"[PASS] Defaults read:  {tunable_read:.4f}s (cpp: {cpp_read:.4f}s, diff: {read_diff:+.1%})")
            passed += 2
        else:
            print(f"[PASS] Defaults write: {tunable_write:.4f}s, read: {tunable_read:.4f}s, data verified")
            passed += 1
    except Exception as e:
        print(f"[FAIL] Defaults test: {e}")
        failed += 1
    finally:
        cleanup_files()

    # --- Test: Each knob individually ---
    knob_tests = [
        ("o_noatime=True",          ThreadedTunableConfig(thread_count=16, o_noatime=True)),
        ("fadvise=sequential",      ThreadedTunableConfig(thread_count=16, fadvise_hint=FadviseHint.SEQUENTIAL)),
        ("io_chunk_kb=1024",        ThreadedTunableConfig(thread_count=16, io_chunk_kb=1024)),
        ("prefetch_depth=4",        ThreadedTunableConfig(thread_count=16, prefetch_depth=4)),
        ("fallocate=True",          ThreadedTunableConfig(thread_count=16, fallocate=True)),
        ("sync_strategy=fdatasync", ThreadedTunableConfig(thread_count=16, sync_strategy=SyncStrategy.FDATASYNC)),
        ("cpu_affinity=True",       ThreadedTunableConfig(thread_count=16, cpu_affinity=True)),
    ]

    for name, cfg in knob_tests:
        try:
            configure(cfg)
            cleanup_files()

            ok = threaded_tunable_ext.threaded_tunable_write_blocks(
                buffer, block_size, block_indices, file_names
            )
            assert ok, "write failed"

            buf_check = torch.zeros_like(buffer)
            ok = threaded_tunable_ext.threaded_tunable_read_blocks(
                buf_check, block_size, block_indices, file_names
            )
            assert ok, "read failed"
            assert torch.equal(buffer, buf_check), "data mismatch"

            print(f"[PASS] Knob: {name:30s} — write OK, read OK, data verified")
            passed += 1
        except Exception as e:
            print(f"[FAIL] Knob: {name:30s} — {e}")
            failed += 1
        finally:
            cleanup_files()

    # --- Test: Combined knobs ---
    try:
        configure(ThreadedTunableConfig(
            thread_count=16,
            o_noatime=True,
            fadvise_hint=FadviseHint.SEQUENTIAL,
            io_chunk_kb=1024,
            prefetch_depth=2,
            fallocate=True,
            sync_strategy=SyncStrategy.NONE,
            cpu_affinity=False,
        ))
        cleanup_files()

        ok = threaded_tunable_ext.threaded_tunable_write_blocks(
            buffer, block_size, block_indices, file_names
        )
        assert ok, "write failed"

        buf_check = torch.zeros_like(buffer)
        ok = threaded_tunable_ext.threaded_tunable_read_blocks(
            buf_check, block_size, block_indices, file_names
        )
        assert ok, "read failed"
        assert torch.equal(buffer, buf_check), "data mismatch"

        print(f"[PASS] Combined knobs               — write OK, read OK, data verified")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Combined knobs: {e}")
        failed += 1
    finally:
        cleanup_files()

    # --- Test: O_DIRECT (may not work on tmpfs) ---
    try:
        configure(ThreadedTunableConfig(thread_count=16, o_direct=True))
        cleanup_files()

        ok = threaded_tunable_ext.threaded_tunable_write_blocks(
            buffer, block_size, block_indices, file_names
        )
        if ok:
            buf_check = torch.zeros_like(buffer)
            ok = threaded_tunable_ext.threaded_tunable_read_blocks(
                buf_check, block_size, block_indices, file_names
            )
            assert ok, "read failed"
            assert torch.equal(buffer, buf_check), "data mismatch"
            print(f"[PASS] O_DIRECT                      — write OK, read OK, data verified")
            passed += 1
        else:
            print(f"[SKIP] O_DIRECT                      — write failed (likely tmpfs), skipped gracefully")
            skipped += 1
    except RuntimeError as e:
        if "O_DIRECT" in str(e) or "Invalid argument" in str(e):
            print(f"[SKIP] O_DIRECT                      — {e}")
            skipped += 1
        else:
            print(f"[FAIL] O_DIRECT: {e}")
            failed += 1
    except Exception as e:
        print(f"[FAIL] O_DIRECT: {e}")
        failed += 1
    finally:
        cleanup_files()

    # --- Summary ---
    total = passed + failed + skipped
    print(f"\n{passed}/{total} passed, {skipped} skipped, {failed} failed")

    if has_baseline:
        print(f"Baseline comparison: write {write_diff:+.1%}, read {read_diff:+.1%} (tolerance {tolerance:.0%})")
        if write_diff > tolerance or read_diff > tolerance:
            print(f"[WARN] Performance difference exceeds {tolerance:.0%} tolerance")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for threaded_tunable backend")
    parser.add_argument("--storage-path", default="/dev/shm", help="Where to write test files")
    parser.add_argument("--tolerance", type=float, default=0.15, help="Max acceptable perf diff vs baseline")
    args = parser.parse_args()

    sys.exit(run_tests(args.storage_path, args.tolerance))

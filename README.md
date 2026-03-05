# CPU-to-Storage I/O Benchmark

Benchmarks for comparing different file I/O implementations in Python and C++. This project measures throughput for parallel read/write operations with various block sizes and thread counts.

## The Benchmarks:

1. **blocks**: Writing\reading a fixed number of *blocks* for each block size, changing the overall data for each block size.
2. **data**: Writing\reading a fixed number of *data* (in GB), changing the number of blocks for each block size.

Each benchmark can use one of the following I/O implementations:
- **C++**: using custom thread pool.
- **Python (self implementation)**: Optimized Python implementation with pre-opened file descriptors.
- **Python (aiofiles)**: Async I/O using aiofiles library.
- **NIXL**: using NVIDIA's NIXL library with POSIX backend.

The benchmarks measures throughput for both read and write operations across different:
- Block sizes (configurable)
- Thread counts (16, 32, 64)
- Storage backends (tmpfs `/dev/shm` or persistent storage)

## 1. Setup

Create the pod and pvc:
```bash
kubeclt apply -f deployment/io_bench_pod.yaml
kubeclt apply -f deployment/io_bench_pvc.yaml      # in case you want to use persistant storage
```

Copy the files to the pod:
```bash
./copy_to_pod.sh
```

Incase you want to test the C++ option, build the extention:
```bash
python setup.py build_ext --inplace
```
This compiles the C++ utilities in `benchmark_cpp_utils/` and creates a `cpp_ext` Python module.


## 2. Running Benchmarks

The benchmark script accepts command-line arguments for easy configuration:

```bash
# Show help and available options
python compare_file_operations.py --help

# Run with default settings (C++, 100GB buffer, 1000 blocks, 5 iterations)
python compare_file_operations.py

```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `blocks` | Benchmark mode: `block` for testing on a fixed number of blocks or `data` for testing a fixed data size |
| `--backend` | str | `python_self_imp` | Backends: `cpp`, `python_aiofiles`, `python_self_imp`, `nixl` |
| `--buffer-size` | int | `100` | Buffer size in GB |
| `--block-sizes` | int list | `2 4 8 16 32 64` | Block sizes in MB to test.  |
| `--iterations` | int | `5` | Number of iterations per test |
| `--num-blocks` | int | `1000` | Number of blocks to transfer (blocks mode only) |
| `--total-gb` | int | `100` | Total data size in GB (total mode only) |
| `--test-name` | str | `""` | Add to results' file name |
| `--verify` | flag | `False` | Verify file contents after write/read operations |

### Checkpoints and Resumption

The benchmark system includes automatic checkpoint functionality that allows you to resume interrupted benchmarks:

- **Automatic Saving**: Results are saved incrementally after each test combination (thread count × block size) completes.
- **Resume on Restart**: If a benchmark is interrupted, simply run the same command again - it will automatically detect existing results and skip completed tests.
- **Configuration Validation**: The system verifies that the configuration matches before resuming (buffer size, iterations, block sizes, etc.)
- **Atomic Writes**: Results are written atomically to prevent corruption if the process is killed during a save

### Environment Variables

Configure storage path and cluster name:

```bash
export CLUSTER_NAME=my-cluster

# Use tmpfs (memory-backed storage)
export STORAGE_PATH=/dev/shm

# Or use persistent storage
export STORAGE_PATH=/mnt/persistent-storage

```

**Note**: If not set, defaults are:
- `STORAGE_PATH`: `/dev/shm` (tmpfs)
- `CLUSTER_NAME`: `unknown`


### 3.  Plotting Results
Copy the all the results from the pod to your local machine:
```bash
./copy_to_pod.sh --get-results
```

Results will be copied to `./results/` directory.

The plotter accepts the benchmark mode and one or more result files:

```bash
python plotter.py <mode> <file1> [file2] [file3] ...

```

## Example

Here is an example of testing cpp, python_self and nixl's backend for writing/reading 100GB at a time from tmpfs and later plotting the results:

Change the path to tmpfs and compile cpp extention:
```bash
export STORAGE_PATH=/dev/shm
python setup.py build_ext --inplace
```

Run the three tests:
```bash
python compare_file_operations.py --mode data --backend python_self_imp --test-name example
python compare_file_operations.py --mode data --backend cpp --test-name example
python compare_file_operations.py --mode data --backend nixl --test-name example
```
Each test will create a json file in `./results/` directory (backend and test mode will be included in the result's file name).

Copy the results back to local machine:
```bash
./copy_to_pod.sh --get-results
```

Plot the results:
```bash
python plotter.py data results/data_example_100gb_memory_cpp.json results/data_example_100gb_memory_python_self_imp.json results/data_example_100gb_memory_nixl.json
```

This will generate a comparison plot showing throughput vs block size for all three implementations:

![Throughput Comparison](plots/data_example_100gb_memory_cpp_3way_total_throughput_plots.png)

#!/bin/bash
# Benchmark job script — called by run_benchmark.sh via LSF

source "$(dirname "$0")/.env"

cd $PROJ_DIR
source .venv/bin/activate
export STORAGE_PATH
export CLUSTER_NAME

MODE=${1:-full}

echo "=== Job started at $(date) on $(hostname) ==="
echo "Run mode: $MODE"
echo "Python: $(python --version)"
echo "Storage path: $STORAGE_PATH"
echo "Working dir: $(pwd)"
echo "==="

if [ "$MODE" = "short" ]; then
    # Short test: 2 block sizes × 3 threads × 1 iteration × 10 blocks
    # Peak disk usage: 10 × 8MB = 80MB
    # Expected runtime: ~2 minutes
    python compare_file_operations.py \
        --mode blocks --backend cpp \
        --buffer-size 1 --iterations 1 \
        --block-sizes 4 8 --num-blocks 10 \
        --test-name io_throughput_short
elif [ "$MODE" = "full" ]; then
    # Full test: 6 block sizes × 3 threads × 5 iterations × 1000 blocks
    # Peak disk usage: 1000 × 64MB = 64GB
    # Expected runtime: ~15-20 minutes
    python compare_file_operations.py \
        --mode blocks --backend cpp \
        --buffer-size 100 --iterations 5 \
        --block-sizes 2 4 8 16 32 64 --num-blocks 1000 \
        --test-name io_throughput
else
    echo "Unknown mode: $MODE. Use 'short' or 'full'."
    exit 1
fi

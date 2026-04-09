#!/bin/bash
# Benchmark job script — called by run_benchmark.sh via LSF

source "$(dirname "$0")/.env"

cd $PROJ_DIR
source .venv/bin/activate
export STORAGE_PATH
export CLUSTER_NAME

MODE=${1:-full}
BACKEND=${2:-cpp}
TUNABLE_CONFIG=${3:-}

echo "=== Job started at $(date) on $(hostname) ==="
echo "Run mode: $MODE"
echo "Backend: $BACKEND"
echo "Python: $(python --version)"
echo "Storage path: $STORAGE_PATH"
echo "Working dir: $(pwd)"
if [ -n "$TUNABLE_CONFIG" ]; then
    echo "Tunable config: $TUNABLE_CONFIG"
fi
echo "==="

# Build --tunable-config flag if provided
TUNABLE_FLAG=""
if [ -n "$TUNABLE_CONFIG" ]; then
    TUNABLE_FLAG="--tunable-config $TUNABLE_CONFIG"
fi

if [ "$MODE" = "short" ]; then
    # Short test: 2 block sizes × 3 threads × 1 iteration × 10 blocks
    # Peak disk usage: 10 × 8MB = 80MB
    # Expected runtime: ~2 minutes
    python compare_file_operations.py \
        --mode blocks --backend $BACKEND \
        --buffer-size 1 --iterations 1 \
        --block-sizes 4 8 --num-blocks 10 \
        --test-name io_throughput_short \
        $TUNABLE_FLAG
elif [ "$MODE" = "full" ]; then
    # Full test: 6 block sizes × 3 threads × 5 iterations × 1000 blocks
    # Peak disk usage: 1000 × 64MB = 64GB
    # Expected runtime: ~15-20 minutes
    python compare_file_operations.py \
        --mode blocks --backend $BACKEND \
        --buffer-size 100 --iterations 5 \
        --block-sizes 2 4 8 16 32 64 --num-blocks 1000 \
        --test-name io_throughput \
        $TUNABLE_FLAG
elif [ "$MODE" = "compare-short" ] || [ "$MODE" = "compare-full" ]; then
    # Compare multiple backends on same node
    # Usage: benchmark_job.sh compare-short "cpp threaded_tunable"
    #        benchmark_job.sh compare-full "all"
    BACKENDS=${2:-all}
    if [ "$BACKENDS" = "all" ]; then
        BACKENDS="cpp threaded_tunable python_self_imp python_aiofiles"
    fi

    for CURRENT_BACKEND in $BACKENDS; do
        echo ""
        echo "=========================================="
        echo "--- Running backend: $CURRENT_BACKEND ---"
        echo "=========================================="

        # Only pass tunable config for threaded_tunable
        CURRENT_TUNABLE_FLAG=""
        if [ "$CURRENT_BACKEND" = "threaded_tunable" ] && [ -n "$TUNABLE_CONFIG" ]; then
            CURRENT_TUNABLE_FLAG="--tunable-config $TUNABLE_CONFIG"
        fi

        if [ "$MODE" = "compare-short" ]; then
            python compare_file_operations.py \
                --mode blocks --backend $CURRENT_BACKEND \
                --buffer-size 1 --iterations 1 \
                --block-sizes 4 8 --num-blocks 10 \
                --test-name io_throughput_short \
                $CURRENT_TUNABLE_FLAG
        else
            python compare_file_operations.py \
                --mode blocks --backend $CURRENT_BACKEND \
                --buffer-size 100 --iterations 5 \
                --block-sizes 2 4 8 16 32 64 --num-blocks 1000 \
                --test-name io_throughput \
                $CURRENT_TUNABLE_FLAG
        fi
    done
else
    echo "Unknown mode: $MODE. Use 'short', 'full', 'compare-short', or 'compare-full'."
    exit 1
fi

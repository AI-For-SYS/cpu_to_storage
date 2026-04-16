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

    # Set parameters based on compare mode
    if [ "$MODE" = "compare-short" ]; then
        BUFFER_SIZE=1
        ITERATIONS=1
        BLOCK_SIZES="4 8"
        TOTAL_GB=0.5
        TEST_NAME=io_throughput_short
    else
        BUFFER_SIZE=100
        ITERATIONS=5
        BLOCK_SIZES="2 4 8 16 32 64"
        TOTAL_GB=30
        TEST_NAME=io_throughput
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

        # Data mode (write + read + concurrent for tunable)
        python compare_file_operations.py \
            --mode data --backend $CURRENT_BACKEND \
            --buffer-size $BUFFER_SIZE --iterations $ITERATIONS \
            --block-sizes $BLOCK_SIZES --total-gb $TOTAL_GB \
            --test-name $TEST_NAME \
            $CURRENT_TUNABLE_FLAG

        # Concurrent mode for non-tunable backends (tunable already runs concurrent in data mode)
        if [ "$CURRENT_BACKEND" != "threaded_tunable" ]; then
            python compare_file_operations.py \
                --mode concurrent --backend $CURRENT_BACKEND \
                --buffer-size $BUFFER_SIZE --iterations $ITERATIONS \
                --block-sizes $BLOCK_SIZES --total-gb $TOTAL_GB \
                --test-name $TEST_NAME
        fi

        echo "--- Finished $CURRENT_BACKEND ---"
    done
else
    echo "Unknown mode: $MODE. Use 'short', 'full', 'compare-short', or 'compare-full'."
    exit 1
fi

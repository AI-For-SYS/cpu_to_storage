#!/bin/bash
# Benchmark job script — called by run_benchmark_on_lsf.sh via LSF (or directly via bash).
#
# Usage:
#   bash benchmark_job.sh <MODE> <BACKEND|BACKENDS> [threads_config=PATH] [iouring_config=PATH]
#
# Modes:
#   short         — single backend, blocks mode, quick
#   full          — single backend, blocks mode, full sweep
#   compare-short — multiple backends side-by-side, data mode, quick
#   compare-full  — multiple backends side-by-side, data mode, full sweep
#
# Examples:
#   bash benchmark_job.sh short threaded_tunable threads_config=results/best_config.json
#   bash benchmark_job.sh short iouring iouring_config=results/best_iouring_config.json
#   bash benchmark_job.sh compare-short "cpp threaded_tunable iouring" \
#        threads_config=results/best_config.json \
#        iouring_config=results/best_iouring_config.json

set -eo pipefail

source "$(dirname "$0")/.env"

cd $PROJ_DIR
source .venv/bin/activate
export STORAGE_PATH
export CLUSTER_NAME

MODE=${1:-full}
BACKEND=${2:-cpp}

# Parse named config args from $3 onward: threads_config=PATH or iouring_config=PATH
THREADS_CONFIG=""
IOURING_CONFIG=""
for arg in "${@:3}"; do
    case "$arg" in
        threads_config=*)
            THREADS_CONFIG="${arg#*=}"
            ;;
        iouring_config=*)
            IOURING_CONFIG="${arg#*=}"
            ;;
        *)
            echo "Error: unknown argument '$arg'"
            echo "Expected: threads_config=PATH or iouring_config=PATH"
            exit 1
            ;;
    esac
done

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/benchmark_${MODE}_${TS}.log"

{
echo "=== Job started at $(date) on $(hostname) ==="
echo "Run mode: $MODE"
echo "Backend: $BACKEND"
echo "Python: $(python --version)"
echo "Storage path: $STORAGE_PATH"
echo "Working dir: $(pwd)"
echo "Log file:    $LOG_FILE"
if [ -n "$THREADS_CONFIG" ]; then
    echo "threads_config: $THREADS_CONFIG"
fi
if [ -n "$IOURING_CONFIG" ]; then
    echo "iouring_config: $IOURING_CONFIG"
fi
echo "==="

# Build per-backend config flag
# (routes threads_config to threaded_tunable, iouring_config to iouring)
config_flag_for() {
    local backend="$1"
    if [ "$backend" = "threaded_tunable" ] && [ -n "$THREADS_CONFIG" ]; then
        echo "--threads-config $THREADS_CONFIG"
    elif [ "$backend" = "iouring" ] && [ -n "$IOURING_CONFIG" ]; then
        echo "--iouring-config $IOURING_CONFIG"
    fi
}

if [ "$MODE" = "short" ]; then
    # Short test: 2 block sizes × 3 threads × 1 iteration × 10 blocks
    # Peak disk usage: 10 × 8MB = 80MB
    # Expected runtime: ~2 minutes
    CONFIG_FLAG=$(config_flag_for "$BACKEND")
    python compare_file_operations.py \
        --mode blocks --backend $BACKEND \
        --buffer-size 1 --iterations 1 \
        --block-sizes 4 8 --num-blocks 10 \
        --test-name io_throughput_short \
        $CONFIG_FLAG
elif [ "$MODE" = "full" ]; then
    # Full test: 6 block sizes × 3 threads × 5 iterations × 1000 blocks
    # Peak disk usage: 1000 × 64MB = 64GB
    # Expected runtime: ~15-20 minutes
    CONFIG_FLAG=$(config_flag_for "$BACKEND")
    python compare_file_operations.py \
        --mode blocks --backend $BACKEND \
        --buffer-size 100 --iterations 5 \
        --block-sizes 2 4 8 16 32 64 --num-blocks 1000 \
        --test-name io_throughput \
        $CONFIG_FLAG
elif [ "$MODE" = "compare-short" ] || [ "$MODE" = "compare-full" ]; then
    # Compare multiple backends on same node
    BACKENDS="$BACKEND"
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

        CURRENT_CONFIG_FLAG=$(config_flag_for "$CURRENT_BACKEND")

        # Data mode (write + read + concurrent for tunable backends)
        python compare_file_operations.py \
            --mode data --backend $CURRENT_BACKEND \
            --buffer-size $BUFFER_SIZE --iterations $ITERATIONS \
            --block-sizes $BLOCK_SIZES --total-gb $TOTAL_GB \
            --test-name $TEST_NAME \
            $CURRENT_CONFIG_FLAG

        # Concurrent mode for non-tunable backends.
        # Tunable backends (threaded_tunable, iouring with iouring_config) already
        # run a concurrent pass inside their data-mode tunable benchmark.
        if [ "$CURRENT_BACKEND" != "threaded_tunable" ] && \
           ! { [ "$CURRENT_BACKEND" = "iouring" ] && [ -n "$IOURING_CONFIG" ]; }; then
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
} 2>&1 | tee "$LOG_FILE"

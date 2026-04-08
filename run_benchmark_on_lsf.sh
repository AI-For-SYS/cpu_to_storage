#!/bin/bash
# Usage: ./run_benchmark_on_lsf.sh short   (quick sanity check)
#        ./run_benchmark_on_lsf.sh full    (full benchmark, default)

source "$(dirname "$0")/.env"

MODE=${1:-full}

bsub -J "io_bench_${MODE}" \
     -o "${PROJ_DIR}/logs/benchmark_%J.out" \
     -e "${PROJ_DIR}/logs/benchmark_%J.err" \
     -n 4 \
     -R "span[hosts=1]" \
     -gpu "num=1" \
     -q normal \
     bash "${PROJ_DIR}/benchmark_job.sh" "$MODE"

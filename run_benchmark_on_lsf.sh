#!/bin/bash
# Usage: ./run_benchmark_on_lsf.sh short                                          (quick sanity, cpp backend)
#        ./run_benchmark_on_lsf.sh full                                           (full benchmark, cpp backend)
#        ./run_benchmark_on_lsf.sh short threaded_tunable                          (quick sanity, tunable backend)
#        ./run_benchmark_on_lsf.sh full threaded_tunable config.json               (full, tunable with config)
#        ./run_benchmark_on_lsf.sh compare-short                                   (all backends, same node, short)
#        ./run_benchmark_on_lsf.sh compare-short "cpp threaded_tunable"            (selected backends, short)
#        ./run_benchmark_on_lsf.sh compare-full "cpp threaded_tunable" config.json (selected backends, full, with config)

source "$(dirname "$0")/.env"

MODE=${1:-full}
BACKEND=${2:-cpp}
TUNABLE_CONFIG=${3:-}

bsub -J "io_bench_${MODE}" \
     -o "${PROJ_DIR}/logs/benchmark_%J.out" \
     -e "${PROJ_DIR}/logs/benchmark_%J.err" \
     -n 4 \
     -R "span[hosts=1]" \
     -gpu "num=1" \
     -q normal \
     bash "${PROJ_DIR}/benchmark_job.sh" "$MODE" "$BACKEND" "$TUNABLE_CONFIG"

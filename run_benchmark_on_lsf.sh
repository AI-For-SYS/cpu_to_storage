#!/bin/bash
# Usage: ./run_benchmark_on_lsf.sh short                                                         (cpp, blocks short)
#        ./run_benchmark_on_lsf.sh full                                                          (cpp, blocks full)
#        ./run_benchmark_on_lsf.sh short threaded_tunable                                        (threads, blocks short)
#        ./run_benchmark_on_lsf.sh full threaded_tunable config.json                             (threads, blocks full, tuned)
#        ./run_benchmark_on_lsf.sh compare-short                                                 (default backends, compare-short)
#        ./run_benchmark_on_lsf.sh compare-short "cpp threaded_tunable"                          (selected backends)
#        ./run_benchmark_on_lsf.sh compare-full "cpp threaded_tunable" config.json               (selected, tuned threads)
#
# Note: iouring cannot run via LSF — compute nodes have kernel.io_uring_disabled=2.
# For iouring runs, invoke benchmark_job.sh directly on lsf-gpu4.

source "$(dirname "$0")/.env"

MODE=${1:-full}
BACKEND=${2:-cpp}
TUNABLE_CONFIG=${3:-}

ARGS=("$MODE" "$BACKEND")
if [ -n "$TUNABLE_CONFIG" ]; then
    ARGS+=("threads_config=$TUNABLE_CONFIG")
fi

bsub -J "io_bench_${MODE}" \
     -o "${PROJ_DIR}/logs/benchmark_%J.out" \
     -e "${PROJ_DIR}/logs/benchmark_%J.err" \
     -n 4 \
     -R "span[hosts=1]" \
     -gpu "num=1" \
     -q normal \
     bash "${PROJ_DIR}/benchmark_job.sh" "${ARGS[@]}"

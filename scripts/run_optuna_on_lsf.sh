#!/bin/bash
# Usage: ./scripts/run_optuna_on_lsf.sh              (full preset, default)
#        ./scripts/run_optuna_on_lsf.sh short         (quick sanity check)

source "$(dirname "$0")/../.env"

PRESET=${1:-full}

bsub -J "optuna_${PRESET}" \
     -o "${PROJ_DIR}/logs/optuna_%J.out" \
     -e "${PROJ_DIR}/logs/optuna_%J.err" \
     -n 4 \
     -R "span[hosts=1]" \
     -gpu "num=1" \
     -q normal \
     bash "${PROJ_DIR}/scripts/optuna_job.sh" "$PRESET"

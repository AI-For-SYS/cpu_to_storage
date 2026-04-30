#!/bin/bash
# Optuna tuning job script — called by run_optuna_threads_on_lsf.sh via LSF

source "$(dirname "$0")/../.env"

cd $PROJ_DIR
source .venv/bin/activate
export STORAGE_PATH
export CLUSTER_NAME

PRESET=${1:-full}

echo "=== Optuna job started at $(date) on $(hostname) ==="
echo "Preset: $PRESET"
echo "Python: $(python --version)"
echo "Storage path: $STORAGE_PATH"
echo "Working dir: $(pwd)"
echo "==="

python optuna_tuner_threads.py --preset $PRESET

#!/bin/bash
# Usage: ./scripts/run_optuna_iouring.sh                     (full preset, default)
#        ./scripts/run_optuna_iouring.sh short                (short preset, ~5-10 min)
#        ./scripts/run_optuna_iouring.sh short 5              (short preset, 5 trials/mode — smoke, ~2-4 min)
#
# Runs directly on the current host (no LSF). iouring requires
# kernel.io_uring_disabled=0 — LSF compute nodes block it.

set -eo pipefail

source "$(dirname "$0")/../.env"

cd "$PROJ_DIR"
source .venv/bin/activate

CURRENT=$(sysctl -n kernel.io_uring_disabled 2>/dev/null || echo "?")
if [ "$CURRENT" != "0" ]; then
    echo "WARNING: kernel.io_uring_disabled=$CURRENT (need 0)."
    echo "         Run: sudo sysctl -w kernel.io_uring_disabled=0"
    echo "         Aborting."
    exit 1
fi

PRESET=${1:-full}
N_TRIALS=${2:-}

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/optuna_iouring_${PRESET}_${TS}.log"

{
  echo "=== Optuna iouring tuning started at $(date) on $(hostname) ==="
  echo "Preset:       $PRESET"
  echo "N_trials:     ${N_TRIALS:-(preset default)}"
  echo "Storage path: $STORAGE_PATH"
  echo "Log file:     $LOG_FILE"
  echo "==="

  python optuna_tuner_iouring.py --preset "$PRESET" ${N_TRIALS:+--n-trials "$N_TRIALS"}
} 2>&1 | tee "$LOG_FILE"

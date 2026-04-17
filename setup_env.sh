#!/bin/bash
# Environment setup for LSF cluster (idempotent — safe to re-run)
# Usage: cd <project_dir> && bash setup_env.sh

source .env
cd $PROJ_DIR
LOG_FILE=$PROJ_DIR/setup_env.log

echo "=== Setup started at $(date) ===" | tee $LOG_FILE

# --- Venv (skip if exists) ---
if [ -d ".venv" ]; then
    echo "--- Venv already exists, skipping creation ---" | tee -a $LOG_FILE
else
    echo "--- Creating venv ---" | tee -a $LOG_FILE
    python3 -m venv .venv 2>&1 | tee -a $LOG_FILE
fi

source .venv/bin/activate

echo "--- Upgrading pip ---" | tee -a $LOG_FILE
pip install --no-cache-dir --upgrade pip 2>&1 | tee -a $LOG_FILE

echo "--- Installing dependencies ---" | tee -a $LOG_FILE
pip install --no-cache-dir torch aiofiles matplotlib ninja numpy optuna 2>&1 | tee -a $LOG_FILE

echo "--- Building C++ extension ---" | tee -a $LOG_FILE
python setup.py build_ext --inplace 2>&1 | tee -a $LOG_FILE

echo "--- Building threaded_tunable extension ---" | tee -a $LOG_FILE
python setup_threaded_tunable.py build_ext --inplace 2>&1 | tee -a $LOG_FILE

# --- liburing (skip if already installed) ---
if [ -f "$HOME/.local/lib/liburing.so" ]; then
    echo "--- liburing already installed, skipping ---" | tee -a $LOG_FILE
else
    echo "--- Building liburing ---" | tee -a $LOG_FILE
    git clone https://github.com/axboe/liburing.git $HOME/liburing 2>&1 | tee -a $LOG_FILE
    cd $HOME/liburing
    ./configure --prefix=$HOME/.local 2>&1 | tee -a $LOG_FILE
    make -j$(nproc) 2>&1 | tee -a $LOG_FILE
    make install 2>&1 | tee -a $LOG_FILE
    cd $PROJ_DIR
fi

echo "--- Building io_uring extension ---" | tee -a $LOG_FILE
python setup_iouring.py build_ext --inplace 2>&1 | tee -a $LOG_FILE

echo "--- Creating benchmark storage dir ---" | tee -a $LOG_FILE
mkdir -p $STORAGE_PATH

echo "=== Setup finished at $(date) ===" | tee -a $LOG_FILE
echo "Verify with: source .venv/bin/activate && python -c \"import cpp_ext; print('cpp OK'); import threaded_tunable_ext; print('tunable OK'); import iouring_ext; print('iouring OK')\""
echo "Full log saved to: $LOG_FILE"

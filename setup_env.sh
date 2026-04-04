#!/bin/bash
# One-time environment setup for LSF cluster
# Usage: cd <project_dir> && bash setup_env.sh

source .env

cd $PROJ_DIR
LOG_FILE=$PROJ_DIR/setup_env.log

echo "=== Setup started at $(date) ===" | tee $LOG_FILE

echo "--- Creating venv ---" | tee -a $LOG_FILE
python3 -m venv .venv 2>&1 | tee -a $LOG_FILE

echo "--- Activating venv ---" | tee -a $LOG_FILE
source .venv/bin/activate

echo "--- Upgrading pip ---" | tee -a $LOG_FILE
pip install --no-cache-dir --upgrade pip 2>&1 | tee -a $LOG_FILE

echo "--- Installing dependencies ---" | tee -a $LOG_FILE
pip install --no-cache-dir torch aiofiles matplotlib ninja numpy 2>&1 | tee -a $LOG_FILE

echo "--- Building C++ extension ---" | tee -a $LOG_FILE
python setup.py build_ext --inplace 2>&1 | tee -a $LOG_FILE

echo "--- Creating benchmark storage dir ---" | tee -a $LOG_FILE
mkdir -p $STORAGE_PATH

echo "=== Setup finished at $(date) ===" | tee -a $LOG_FILE
echo "Verify with: source .venv/bin/activate && python -c \"import cpp_ext; print('OK')\""
echo "Full log saved to: $LOG_FILE"

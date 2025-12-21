#!/bin/bash
set -e  # Exit immediately if a command fails
set -o pipefail

# Ensure Python virtual environment is activated (optional)
# source .venv/bin/activate

# Load environment variables
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Default model path if not set
ML_MODEL_PATH="${ML_MODEL_PATH:-xgb_model.pkl}"

# Check if model exists
if [ ! -f "$ML_MODEL_PATH" ]; then
    echo "[INFO] Model not found at $ML_MODEL_PATH. Training..."
    python train_model.py
else
    echo "[INFO] Model already exists at $ML_MODEL_PATH. Skipping training."
fi

# Default PORT
PORT="${PORT:-8000}"

# Start the FastAPI bot
echo "[INFO] Starting FastAPI bot on port $PORT..."
uvicorn bot:app --host 0.0.0.0 --port "$PORT" --reload

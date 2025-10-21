#!/bin/bash

# Use Render's persistent disk path for model and DB
DATA_DIR="/mnt/data"
mkdir -p "$DATA_DIR"

# Paths
MODEL_PATH="$DATA_DIR/xgb_model.pkl"
DB_PATH="$DATA_DIR/signals.db"

# Export env variables for bot.py
export ML_MODEL_PATH="$MODEL_PATH"
export SQLITE_DB="$DB_PATH"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "[INFO] Model not found. Training..."
    python train_model.py
else
    echo "[INFO] Model already exists. Skipping training."
fi

# Start the FastAPI bot on the port Render assigns
echo "[INFO] Starting FastAPI bot..."
uvicorn bot:app --host 0.0.0.0 --port $PORT

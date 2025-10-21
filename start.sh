#!/bin/bash

# Create persistent data folder
mkdir -p /opt/render/project/data

# Path to your model
MODEL_PATH="/opt/render/project/data/xgb_model.pkl"
DATA_HASH_FILE="/opt/render/project/data/data_hash.txt"

# Compute current dataset hash
CURRENT_HASH=$(md5sum signals.csv | awk '{print $1}')

TRAIN_MODEL=false

if [ ! -f "$MODEL_PATH" ]; then
    echo "[INFO] Model not found. Training..."
    TRAIN_MODEL=true
elif [ ! -f "$DATA_HASH_FILE" ]; then
    TRAIN_MODEL=true
else
    OLD_HASH=$(cat "$DATA_HASH_FILE")
    if [ "$CURRENT_HASH" != "$OLD_HASH" ]; then
        echo "[INFO] Data changed. Retraining model..."
        TRAIN_MODEL=true
    fi
fi

if [ "$TRAIN_MODEL" = true ]; then
    python train_model.py
    echo "$CURRENT_HASH" > "$DATA_HASH_FILE"
else
    echo "[INFO] Model up-to-date. Skipping training."
fi

# Start the FastAPI bot
echo "[INFO] Starting FastAPI bot..."
uvicorn bot:app --host 0.0.0.0 --port $PORT

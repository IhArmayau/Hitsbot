#!/bin/bash

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH="./xgb_model.pkl"

# -----------------------------
# Train model if missing
# -----------------------------
if [ ! -f "$MODEL_PATH" ]; then
    echo "[INFO] Model not found. Training..."
    python train_model.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Model training failed. Exiting."
        exit 1
    fi
else
    echo "[INFO] Model already exists. Skipping training."
fi

# -----------------------------
# Start FastAPI bot
# -----------------------------
echo "[INFO] Starting FastAPI bot..."
# Replace shell with uvicorn so Render keeps the process alive
exec uvicorn bot:app --host 0.0.0.0 --port $PORT --reload

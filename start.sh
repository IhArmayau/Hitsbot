#!/bin/bash

# Path to your model
MODEL_PATH="/mnt/data/xgb_model.pkl"

# Ensure /mnt/data exists
mkdir -p /mnt/data

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model not found. Training..."
    python train_model.py
else
    echo "Model already exists. Skipping training."
fi

# Start the FastAPI bot
uvicorn bot:app --host 0.0.0.0 --port $PORT

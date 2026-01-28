#!/bin/bash

# Neurolab AI: Automation Pipeline
# This script automates the generation of training data and the subsequent model training.

# Exit immediately if a command exits with a non-zero status.
set -e

# Configuration
GEN_SCRIPT="scripts/generation/generate_train_datasets.py"
TRAIN_SCRIPT="scripts/training/train_model.py"

# Detect Python binary
if [ -f "./venv/bin/python" ]; then
    PYTHON_BIN="./venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

echo "=========================================="
echo "   Neurolab AI: Automation Pipeline"
echo "   Using: $PYTHON_BIN"
echo "=========================================="

# Step 1: Data Generation
echo "[1/2] Generating training data..."
if [ -f "$GEN_SCRIPT" ]; then
    $PYTHON_BIN "$GEN_SCRIPT"
else
    echo "[-] Error: Generation script not found at $GEN_SCRIPT"
    exit 1
fi

# Step 2: Model Training
echo ""
echo "[2/2] Training the model..."
if [ -f "$TRAIN_SCRIPT" ]; then
    $PYTHON_BIN "$TRAIN_SCRIPT"
else
    echo "[-] Error: Training script not found at $TRAIN_SCRIPT"
    exit 1
fi

echo ""
echo "=========================================="
echo "   Pipeline completed successfully!"
echo "=========================================="

# Execute trailing commands if provided (for Docker ENTRYPOINT)
if [ $# -gt 0 ]; then
    echo "Executing: $@"
    exec "$@"
fi

#!/bin/bash
# RunPod T4 Dress Rehearsal Script
# Run this on a RunPod T4 pod to test exact eval conditions

set -e

echo "=== IOL-AI 2026 T4 Dress Rehearsal ==="

cd /workspace/iol

# Build eval-matching Docker image
docker build -t iol-t4-rehearsal -f Dockerfile.t4 .

# Run rehearsal inside container with your chosen model
MODEL_PATH="${MODEL_PATH:-./models/qwen3.5-4b-awq}"
MODEL_NAME=$(basename "$MODEL_PATH")

echo "Testing model: $MODEL_NAME"
docker run --gpus all --rm \
    -v "$(pwd):/workspace" \
    -e MODEL_ID=/workspace/$MODEL_PATH \
    -e QUANT=awq \
    -e MAX_NEW_TOKENS=512 \
    -e SEED=42 \
    iol-t4-rehearsal \
    bash -c "cd /workspace && python script.py"

# Verify outputs
if [ -f "/workspace/submission.csv" ]; then
    ROWS=$(wc -l < /workspace/submission.csv)
    echo "SUCCESS: submission.csv has $ROWS rows"
else
    echo "FAILURE: submission.csv not found"
    exit 1
fi

echo "=== Rehearsal complete ==="

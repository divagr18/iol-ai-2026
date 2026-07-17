#!/bin/bash
# RunPod H100 Auto-Setup Script for IOL-AI 2026
# Paste this into your RunPod H100 terminal after starting a pod

set -e

echo "=== IOL-AI 2026 H100 Setup ==="

REPO_URL="${REPO_URL:-https://github.com/divagr18/iol-ai-2026}"
if [ ! -d "/workspace/iol" ]; then
    git clone "$REPO_URL" /workspace/iol || true
fi
cd /workspace/iol

pip install -q -r requirements.txt

# Download models in parallel
echo "Downloading Gemma-4-E4B-AWQ (~3GB)..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('cyankiwi/gemma-4-E4B-it-AWQ-4bit', local_dir='models/gemma4-e4b-awq', local_dir_use_symlinks=False)" &
PID1=$!

echo "Downloading Qwen3.5-9B-AWQ (~6GB)..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('QuantTrio/Qwen3.5-9B-AWQ', local_dir='models/qwen3.5-9b-awq', local_dir_use_symlinks=False)" &
PID2=$!

echo "Downloading Gemma-4-12B (~13GB)..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-12B-it', local_dir='models/gemma4-12b', local_dir_use_symlinks=False)" &
PID3=$!

wait $PID1 $PID2 $PID3
echo "All models downloaded"

# Smoke test with Gemma-4-E4B (fast, no reasoning overhead)
echo "Smoke test: Gemma-4-E4B on 5 problems..."
MODEL_ID=./models/gemma4-e4b-awq QUANT=awq MAX_NEW_TOKENS=512 \
    python -m src.harness --limit 5 --output data/smoke_test --no_score

# Full bake-off
echo "Running bake-off on 40 problems..."
python -m src.bakeoff --config configs/bakeoff_models.json --limit 40 --output data/bakeoff/summary.csv

echo "=== Setup complete ==="
echo "Check data/bakeoff/summary.csv for results"

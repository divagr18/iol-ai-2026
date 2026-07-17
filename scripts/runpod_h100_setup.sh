#!/bin/bash
# RunPod H100 Auto-Setup Script for IOL-AI 2026
# Paste this into your RunPod H100 terminal after starting a pod

set -e

echo "=== IOL-AI 2026 H100 Setup ==="

# 1. Clone your repo (replace with your actual repo URL after you push)
REPO_URL="${REPO_URL:-https://github.com/YOUR_USERNAME/iol-ai-2026}"
if [ ! -d "/workspace/iol" ]; then
    git clone "$REPO_URL" /workspace/iol || true
fi
cd /workspace/iol

# 2. Install dependencies
pip install -q -r requirements.txt

# 3. Download latest models (parallel where possible)
echo "Downloading Qwen3.5-4B-AWQ..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('cyankiwi/Qwen3.5-4B-AWQ-4bit', local_dir='models/qwen3.5-4b-awq', local_dir_use_symlinks=False)" &
PID1=$!

echo "Downloading Qwen3.5-9B-AWQ..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('QuantTrio/Qwen3.5-9B-AWQ', local_dir='models/qwen3.5-9b-awq', local_dir_use_symlinks=False)" &
PID2=$!

echo "Downloading Gemma-4-12B..."
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-12B-it', local_dir='models/gemma4-12b', local_dir_use_symlinks=False)" &
PID3=$!

# Wait for downloads
wait $PID1 $PID2 $PID3
echo "All models downloaded"

# 4. Quick smoke test on 5 problems with Qwen3.5-4B
echo "Smoke test: Qwen3.5-4B on 5 problems..."
MODEL_ID=./models/qwen3.5-4b-awq QUANT=awq MAX_NEW_TOKENS=256 \
    python -m src.harness --limit 5 --output data/smoke_test --no_score

# 5. Run full bake-off
echo "Running bake-off on 40 problems..."
python -m src.bakeoff --config configs/bakeoff_models.json --limit 40 --output data/bakeoff/summary.csv

echo "=== Setup complete ==="
echo "Check data/bakeoff/summary.csv for results"

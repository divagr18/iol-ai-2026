#!/bin/bash
# Quick Qwen2.5-14B-AWQ setup and test

cd /workspace/iol

echo "=== Step 1: Install sandbox-compatible versions ==="
pip install -q transformers==4.44.1 autoawq==0.2.7.post3

echo "=== Step 2: Download Qwen2.5-14B-AWQ (~7GB) ==="
if [ ! -d "models/qwen2.5-14b-awq" ]; then
    hf download Qwen/Qwen2.5-14B-Instruct-AWQ \
        --local-dir models/qwen2.5-14b-awq
fi

echo "=== Step 3: Copy weights to repo root for sandbox ==="
# The sandbox loads from "." so we need weights in repo root or subdir
mkdir -p qwen2.5-14b-awq
cp -r models/qwen2.5-14b-awq/* qwen2.5-14b-awq/

echo "=== Step 4: Test locally ==="
MODEL_ID=./qwen2.5-14b-awq QUANT=awq MAX_NEW_TOKENS=512 \
    python script.py

echo "=== Step 5: Score ==="
python -m src.scorer submission.csv data/linguini_test_sample.csv

echo ""
echo "=== Step 6: Upload to HF (if score is good) ==="
echo "Run: python scripts/upload_to_hf.py"

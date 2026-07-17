#!/bin/bash
# Setup Qwen2.5-14B-AWQ for eval sandbox submission

cd /workspace/iol

# Download Qwen2.5-14B-AWQ (~7-8GB, fits T4 16GB)
echo "Downloading Qwen2.5-14B-AWQ..."
hf download Qwen/Qwen2.5-14B-Instruct-AWQ \
    --local-dir models/qwen2.5-14b-awq

# Copy to repo root (eval sandbox loads from ".")
mkdir -p models/qwen2.5-14b-awq
cp -r models/qwen2.5-14b-awq/* . 2>/dev/null || true

# Test locally with sandbox-compatible transformers
pip install -q transformers==4.44.1 autoawq==0.2.7.post3

# Run test
python script.py

# Score
python -m src.scorer submission.csv data/linguini_test_sample.csv

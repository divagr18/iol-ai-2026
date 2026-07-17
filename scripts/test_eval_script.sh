#!/bin/bash
# Quick eval test on RunPod H100 with AWQ model (same as eval sandbox)

cd /workspace/iol

# Install autoawq if not present
pip install -q autoawq==0.2.6 autoawq-kernels==0.0.9

# Download Gemma-4 E4B AWQ (~3GB, eval-compatible)
if [ ! -d "models/gemma4-e4b-awq" ]; then
    echo "Downloading Gemma-4 E4B AWQ..."
    huggingface-cli download cyankiwi/gemma-4-E4B-it-AWQ-4bit \
        --local-dir models/gemma4-e4b-awq \
        --local-dir-use-symlinks False
fi

# Test script.py on 10 problems (local sample)
MODEL_ID=./models/gemma4-e4b-awq QUANT=awq MAX_NEW_TOKENS=512 \
    python -m src.harness --limit 10 --output data/script_test.csv

# Score it
python -m src.scorer data/script_test.csv data/linguini_test_sample.csv

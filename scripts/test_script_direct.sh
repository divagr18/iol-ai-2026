#!/bin/bash
# Direct eval script test (mimics eval sandbox, no harness overhead)

cd /workspace/iol

# Setup eval sandbox path
mkdir -p /tmp/data
cp data/linguini_test_sample.csv /tmp/data/test.csv

# Configure to match your working 25.94 setup
export MODEL_ID=./models/gemma4-e4b-awq
export QUANT=awq
export MAX_NEW_TOKENS=512
export TEMPERATURE=0.0
export USE_EXPLANATION=false
export MAX_REPAIR=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Check model exists
if [ ! -d "$MODEL_ID" ]; then
    echo "Model not found. Downloading..."
    hf download cyankiwi/gemma-4-E4B-it-AWQ-4bit \
        --local-dir models/gemma4-e4b-awq
fi

echo "Running script.py (eval sandbox style)..."
python script.py

echo ""
echo "Scoring..."
python -m src.scorer submission.csv data/linguini_test_sample.csv --weights test

#!/bin/bash
# Fix autoawq installation for local testing

cd /workspace/iol

# Install autoawq without version conflicts
# The sandbox has autoawq 0.2.7.post3 preinstalled, but locally we may need a compatible version
pip install -q autoawq autoawq-kernels

# Test Qwen2.5 loading
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
print('torch:', torch.__version__)
tok = AutoTokenizer.from_pretrained('./qwen2.5-14b-awq', trust_remote_code=True)
print('Tokenizer OK')
model = AutoModelForCausalLM.from_pretrained(
    './qwen2.5-14b-awq',
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto',
)
print('Model loaded OK')
"

# Run full test
MODEL_ID=./qwen2.5-14b-awq MAX_NEW_TOKENS=512 python script.py

# Score
python -m src.scorer submission.csv data/linguini_test_sample.csv

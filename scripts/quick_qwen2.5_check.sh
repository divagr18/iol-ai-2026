#!/bin/bash
# Fast Qwen2.5 test — skip autoawq compilation, just verify load and run

cd /workspace/iol

echo "=== Step 1: Check if autoawq is already available ==="
python -c "import autoawq; print('autoawq version:', autoawq.__version__)" 2>/dev/null || echo "autoawq not installed"

echo ""
echo "=== Step 2: Quick load test (no full model weights, just config+tokenizer) ==="
python -c "
from transformers import AutoTokenizer
import json

# Just test tokenizer loads
tok = AutoTokenizer.from_pretrained('./qwen2.5-14b-awq', trust_remote_code=True)
print('Tokenizer OK, vocab size:', len(tok))

# Check if autoawq is importable
try:
    import autoawq
    print('autoawq importable')
except ImportError as e:
    print('autoawq NOT importable:', e)
"

echo ""
echo "=== Step 3: If autoawq import fails, use conda env or wait for compile ==="
echo "The eval sandbox already has autoawq 0.2.7.post3 preinstalled."
echo "For local testing, you need autoawq. Try:"
echo "  conda install -c conda-forge autoawq"
echo "Or wait for pip compile to finish."

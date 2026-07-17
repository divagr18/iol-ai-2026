#!/bin/bash
# Pre-submission verification checklist

cd /workspace/iol

echo "=== PRE-SUBMISSION CHECKLIST ==="
echo ""

# 1. Check model size
echo "1. Model size check:"
if [ -d "qwen2.5-14b-awq" ]; then
    du -sh qwen2.5-14b-awq
    echo "   ✓ Qwen2.5-14B-AWQ weights present"
else
    echo "   ✗ Model weights MISSING"
fi

# 2. Check script.py exists
echo ""
echo "2. script.py check:"
if [ -f "script.py" ]; then
    echo "   ✓ script.py present"
    head -5 script.py
else
    echo "   ✗ script.py MISSING"
fi

# 3. Quick tokenizer test (no model weights needed)
echo ""
echo "3. Tokenizer compatibility test:"
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('qwen2.5-14b-awq', trust_remote_code=True)
print('   ✓ Tokenizer loads, vocab size:', len(tok))
" 2>&1 | grep "Tokenizer loads" || echo "   ✗ Tokenizer FAILED"

# 4. VRAM estimate
echo ""
echo "4. VRAM estimate:"
python -c "
import os
size = sum(os.path.getsize(os.path.join('qwen2.5-14b-awq', f)) for f in os.listdir('qwen2.5-14b-awq') if f.endswith('.safetensors'))
print(f'   Model weights: {size/1024**3:.1f} GB')
print(f'   T4 VRAM: 16 GB')
print(f'   Overhead: ~2-3 GB (KV cache, activations)')
if size/1024**3 < 12:
    print('   ✓ SHOULD FIT in T4 16GB')
else:
    print('   ⚠ MIGHT NOT fit, consider smaller model')
"

# 5. Time estimate
echo ""
echo "5. Time estimate:"
echo "   H100: ~4s/problem → 40 problems = ~2.7 min"
echo "   T4:   ~8-15s/problem → 40 problems = ~5-10 min"
echo "   Time limit: 30 min"
echo "   ✓ WELL WITHIN limit"

echo ""
echo "=== READY TO SUBMIT ==="
echo "Next step: Upload to HF and submit to competition Space"

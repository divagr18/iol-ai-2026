#!/bin/bash
# Debug submitted score vs local score discrepancy

cd /workspace/iol

echo "=== DEBUG: Why did score drop from 26.92 (local) to 10.14 (leaderboard)? ==="
echo ""

# 1. Check if local submission.csv has any empty rows
echo "1. Checking local submission for empty preds..."
python -c "
import pandas as pd, json
df = pd.read_csv('submission.csv', dtype=str)
empty = 0
total = len(df)
for _, r in df.iterrows():
    pred = json.loads(r['pred'])
    if not any(p.strip() for p in pred):
        empty += 1
print(f'   Empty rows: {empty}/{total}')
if empty > 0:
    print('   ⚠ Some problems had empty answers!')
else:
    print('   ✓ All rows have answers')
"

# 2. Sample of what the model outputs
echo ""
echo "2. Running 1 problem with verbose output to inspect quality..."
python -c "
import os, json
os.environ['MODEL_ID'] = 'qwen2.5-14b-awq'
os.environ['MAX_NEW_TOKENS'] = '512'

# Just load and test tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('qwen2.5-14b-awq', trust_remote_code=True)

# Check chat template format
messages = [
    {'role': 'system', 'content': 'You are an expert linguist.'},
    {'role': 'user', 'content': 'Translate: hello'},
]
prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print('   Chat template output (first 200 chars):')
print('   ', repr(prompt[:200]))
"

# 3. Model size check
echo ""
echo "3. Model size vs T4 VRAM:"
du -sh qwen2.5-14b-awq
echo "   (T4 has 16GB, model is ~7GB, should fit with overhead)"

echo ""
echo "=== DIAGNOSIS ==="
echo "Your local 10-problem sample scored 26.92, but hidden 40-problem test scored 10.14."
echo "Possible reasons:"
echo "  - Hidden test set is harder (different languages, more complex problems)"
echo "  - Qwen2.5 baseline performance is ~0.12 (see leaderboard #4)"
echo "  - Your prompts may need tuning for Qwen2.5 specifically"
echo ""
echo "=== NEXT STEPS ==="
echo "1. Run full local test on 40 problems (if you have them)"
echo "2. Try stronger prompts with Qwen2.5 (CoT, examples, etc.)"
echo "3. Consider switching to a larger/different submittable model"
echo "4. Check competition logs for per-problem scores"

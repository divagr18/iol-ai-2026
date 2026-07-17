#!/usr/bin/env python3
"""Debug tokenizer loading issue."""
import json
import os
from pathlib import Path

# Same cache fix as script.py
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

model_id = os.environ.get("MODEL_ID", "qwen2.5-14b-awq")
print(f"MODEL_ID: {model_id}")
print(f"CWD: {Path.cwd()}")

# Check model directory
model_path = Path(model_id)
if not model_path.is_absolute():
    model_path = Path.cwd() / model_path
print(f"Model path: {model_path}")
print(f"Exists: {model_path.exists()}")

if model_path.exists():
    files = list(model_path.iterdir())
    print(f"Files: {[f.name for f in files]}")
    
    config = model_path / "config.json"
    if config.exists():
        with open(config) as f:
            cfg = json.load(f)
        print(f"model_type from config: {cfg.get('model_type', 'NOT FOUND')}")
    else:
        print("NO config.json in model directory!")
    
    tok_config = model_path / "tokenizer_config.json"
    if tok_config.exists():
        with open(tok_config) as f:
            tcfg = json.load(f)
        print(f"tokenizer_class: {tcfg.get('tokenizer_class', 'NOT FOUND')}")
        extra = tcfg.get('extra_special_tokens')
        print(f"extra_special_tokens type: {type(extra).__name__}, len: {len(extra) if extra else 'None'}")
    else:
        print("NO tokenizer_config.json in model directory!")

# Check if there's a config.json in repo root
root_config = Path.cwd() / "config.json"
if root_config.exists():
    with open(root_config) as f:
        rcfg = json.load(f)
    print(f"\nWARNING: config.json exists in repo root!")
    print(f"  Root model_type: {rcfg.get('model_type', 'NOT FOUND')}")
    print(f"  This could confuse AutoTokenizer if MODEL_ID='.'")

print("\n=== Attempting to load tokenizer ===")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    print(f"SUCCESS: {type(tok).__name__}, vocab: {len(tok)}")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

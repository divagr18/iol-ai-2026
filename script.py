"""
IOL-AI 2026 Submission Script — Qwen2.5-14B-AWQ edition

Eval sandbox compatible:
  - transformers 4.44.1 ✓
  - autoawq 0.2.7.post3 ✓
  - T4 16GB VRAM ✓
  - 30min time limit ✓

Ship Qwen2.5-14B-AWQ weights in repo, load with AutoModelForCausalLM.
No internet at runtime.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Configuration
MODEL_ID = os.environ.get("MODEL_ID", "qwen2.5-14b-awq")  # weights shipped in repo subdir
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
DO_SAMPLE = False

# System prompt (same as your 25.94 winning prompt)
SYSTEM_BASE = (
    "You are an expert linguist solving International Linguistics Olympiad problems. "
    "Answer every numbered item. Put each answer on its own line, "
    "with NO numbering and NO extra text. "
    "NEVER show your reasoning, thinking, or analysis. "
    "Output ONLY the final answers, nothing else."
)

TASK_HINTS = {
    "translation": "Output only the translated sentence.",
    "matching": "Output only matched labels, one per line.",
    "fill_blanks": "Output only the missing form for each blank.",
    "text_to_num": "Output only the numeral.",
    "num_to_text": "Output only the written number form.",
}


def load_model():
    print("Loading Qwen2.5-14B-AWQ...", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
    except Exception:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded", flush=True)
    return tok, model


def build_messages(context, query, task_type):
    system = "You solve International Linguistics Olympiad problems. Answer every numbered item. Put each answer on its own line, in order, with no numbering and no extra text."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context.strip()}\n\n{query.strip()}"},
    ]


def generate(tok, model, messages, max_tokens):
    try:
        inputs = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        # Build attention mask (pad == eos for Qwen, so explicit mask prevents warning)
        attention_mask = torch.ones_like(inputs)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE if DO_SAMPLE else None,
                top_p=1.0,
                top_k=50,
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = outputs[0][inputs.shape[-1]:]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()
    except Exception as e:
        print(f"ERROR generating: {e}", flush=True)
        return ""


def parse_answers(text, expected_count=None):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Thinking Process:.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\*\*.*?\*\*", "", text, flags=re.DOTALL)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    filtered = []
    for line in lines:
        low = line.lower()
        if low.startswith(("answer", "solution", "output", "result", "explanation", "note", "thinking")):
            continue
        line = re.sub(r"\*\*", "", line)
        line = re.sub(r"\*", "", line)
        cleaned = re.sub(r"^\s*(\d+[\.\)]\s+|\w[\.\)]\s+|-\s+)", "", line)
        if cleaned:
            filtered.append(cleaned)
    if expected_count:
        while len(filtered) < expected_count:
            filtered.append("")
        filtered = filtered[:expected_count]
    return filtered


def count_expected_items(query):
    numbers = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s+", query)
    if numbers:
        return max(int(n) for n in numbers)
    range_match = re.search(r"\(\s*(\d+)\s*[-–—]\s*(\d+)\s*\)", query)
    if range_match:
        return int(range_match.group(2)) - int(range_match.group(1)) + 1
    return len([ln for ln in query.splitlines() if re.match(r"^\d+[\.\)]", ln.strip())])


def main():
    data_path = Path("/tmp/data/test.csv")
    if not data_path.exists():
        data_path = Path("data/run_output/tmp_data/test.csv")
    if not data_path.exists():
        print(f"ERROR: test.csv not found at {data_path}", flush=True)
        sys.exit(1)

    print(f"Reading test set from {data_path}", flush=True)
    df = pd.read_csv(data_path, dtype=str).fillna("")
    print(f"Loaded {len(df)} problems", flush=True)

    tok, model = load_model()

    rows = []
    total = len(df)
    start_time = time.time()
    time_budget = 25 * 60

    for idx, r in df.iterrows():
        problem_id = str(r["id"])
        context = str(r.get("context", ""))
        query = str(r.get("query", ""))
        task_type = str(r.get("task_type", "unknown"))
        expected = count_expected_items(query)

        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        per_problem = remaining / max(total - idx, 1)
        max_new = min(MAX_NEW_TOKENS, 128) if per_problem < 10 else MAX_NEW_TOKENS

        messages = build_messages(context, query, task_type)
        text = generate(tok, model, messages, max_new)
        answers = parse_answers(text, expected)
        pred = json.dumps(answers, ensure_ascii=False)
        rows.append({"id": problem_id, "pred": pred})

        print(
            f"  [{idx + 1}/{total}] {problem_id}: {len(answers)} answers",
            flush=True,
        )

    submission = pd.DataFrame(rows)
    submission.to_csv("submission.csv", index=False, encoding="utf-8")
    print(f"Wrote submission.csv with {len(submission)} rows", flush=True)

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()

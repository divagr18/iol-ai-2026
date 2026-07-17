"""
Local 4060 GPU Test Script (Unsloth GGUF Q4_K_M)

Fast inference via llama-cpp-python. For LOCAL DEVELOPMENT ONLY.
NOT compatible with the competition eval sandbox (which uses transformers).
Use this for quick prompt iteration on your 4060, then switch to AWQ/BNB
for the actual submission.

Setup:
    pip install llama-cpp-python pandas
    # Download GGUF from https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
    # Place the Q4_K_M file in models/ directory

Run:
    python local_4060_gguf.py --model models/qwen3.5-4b-q4_k_m.gguf --limit 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Must have llama-cpp-python installed
# pip install llama-cpp-python
from llama_cpp import Llama

# Add src for harness/scorer imports
sys.path.insert(0, str(Path(__file__).parent / "src"))
from scorer import score_submission


def load_model(model_path: str, n_ctx: int = 4096):
    """Load GGUF model with full GPU offload on 4060."""
    print(f"Loading GGUF: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1,  # offload ALL layers to GPU
        verbose=False,
        n_batch=512,
    )
    print("Model loaded")
    return llm


def run_inference(llm, context: str, query: str, task_type: str, max_tokens: int = 256) -> str:
    """Run a single problem through the GGUF model."""
    system = (
        "You are an expert linguist solving International Linguistics Olympiad problems. "
        "Answer every numbered item. Put each answer on its own line, "
        "with NO numbering and NO extra text."
    )

    task_hints = {
        "translation": "Output only the translated sentence.",
        "matching": "Output only matched labels (letters/numbers), one per line.",
        "fill_blanks": "Output only the missing form for each blank.",
        "text_to_num": "Output only the numeral.",
        "num_to_text": "Output only the written number form.",
    }

    hint = task_hints.get(task_type, "")
    if hint:
        system += f" {hint}"

    # Qwen chat format (manual since llama.cpp may not have exact template)
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{context.strip()}\n\n{query.strip()}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        stop=["<|im_end|>", "<|im_start|>"],
    )
    return output["choices"][0]["text"].strip()


def parse_answers(text: str, expected_count: int = None) -> list:
    """Parse answer lines from model output."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Remove common headers
    filtered = []
    for line in lines:
        if line.lower().startswith(("answer", "solution", "output", "result")):
            continue
        filtered.append(line)
    if expected_count:
        while len(filtered) < expected_count:
            filtered.append("")
        filtered = filtered[:expected_count]
    return filtered


def main():
    parser = argparse.ArgumentParser(description="Local 4060 GGUF test")
    parser.add_argument("--model", type=str, required=True, help="Path to .gguf file")
    parser.add_argument("--data", type=str, default="data/linguini_test_sample.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default="data/gguf_submission.csv")
    parser.add_argument("--score", action="store_true", help="Score against gold answers")
    args = parser.parse_args()

    df = pd.read_csv(args.data, dtype=str)
    if args.limit:
        df = df.head(args.limit)

    llm = load_model(args.model)

    rows = []
    times = []

    for idx, r in df.iterrows():
        start = time.time()
        text = run_inference(
            llm,
            r["context"],
            r["query"],
            r.get("task_type", "unknown"),
            max_tokens=args.max_tokens,
        )
        elapsed = time.time() - start
        times.append(elapsed)

        answers = parse_answers(text)
        pred = json.dumps(answers, ensure_ascii=False)
        rows.append({"id": str(r["id"]), "pred": pred})

        print(f"[{idx+1}/{len(df)}] {r['id']}: {len(answers)} answers in {elapsed:.1f}s")
        if idx < 3:
            print(f"  Raw output:\n{text[:200]}")

    submission = pd.DataFrame(rows)
    submission.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSubmission saved to {args.output}")
    print(f"Avg time per problem: {sum(times)/len(times):.1f}s")
    print(f"Total time: {sum(times):.1f}s")

    if args.score:
        gold_path = Path(args.data)
        if gold_path.exists():
            result = score_submission(args.output, str(gold_path))
            print(f"\nScore: {result['score']:.2f} (EM={result['exact_match']:.4f}, chrF={result['chrF']:.4f})")


if __name__ == "__main__":
    main()

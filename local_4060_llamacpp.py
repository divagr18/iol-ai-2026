"""
Local 4060 Test via llama.cpp CLI (Native, Pre-compiled CUDA)

Uses the actual llama.cpp binaries you already have:
    C:\Users\Keshav\Downloads\llama-b10064-bin-win-cuda-13.3-x64\llama-cli.exe

This avoids all Python binding issues. The binary talks directly to your 4060.

Setup:
    1. Download Q4_K_M GGUF to models/ (browser or python)
    2. Run this script

Usage:
    python local_4060_llamacpp.py --model models/qwen3.5-4b-q4_k_m.gguf --limit 5 --score
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Path to llama.cpp binary — check local llama/ folder first, fallback to Downloads
LLAMA_CLI_LOCAL = Path(__file__).parent / "llama" / "llama-cli.exe"
LLAMA_CLI_FALLBACK = Path(r"C:\Users\Keshav\Downloads\llama-b10064-bin-win-cuda-13.3-x64\llama-cli.exe")
LLAMA_CLI = LLAMA_CLI_LOCAL if LLAMA_CLI_LOCAL.exists() else LLAMA_CLI_FALLBACK

sys.path.insert(0, str(Path(__file__).parent / "src"))
from scorer import score_submission


def run_llama_cli(model_path: str, prompt: str, max_tokens: int = 256, temp: float = 0.0) -> str:
    """Call llama-cli.exe as a subprocess. Returns generated text."""
    if not LLAMA_CLI.exists():
        raise FileNotFoundError(f"llama-cli.exe not found at {LLAMA_CLI}. Download from https://github.com/ggerganov/llama.cpp/releases")

    cmd = [
        str(LLAMA_CLI),
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temp),
        "-ngl", "999",  # offload all layers to GPU
        "--no-display-prompt",  # don't echo the prompt back
        "-t", "6",  # threads
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(f"llama-cli stderr: {result.stderr}", flush=True)
    return result.stdout.strip()


def build_prompt(context: str, query: str, task_type: str) -> str:
    """Build a Qwen-format prompt for llama.cpp."""
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

    # Qwen chat template (manual since llama.cpp doesn't know it)
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{context.strip()}\n\n{query.strip()}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def parse_answers(text: str, expected_count: int = None) -> list:
    """Parse answer lines from model output."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    filtered = []
    for line in lines:
        if line.lower().startswith(("answer", "solution", "output", "result")):
            continue
        # Remove leading numbers/bullets
        cleaned = re.sub(r"^\s*(\d+[\.\)]\s+|\w[\.\)]\s+|-\s+)", "", line)
        if cleaned:
            filtered.append(cleaned)
    if expected_count:
        while len(filtered) < expected_count:
            filtered.append("")
        filtered = filtered[:expected_count]
    return filtered


def count_expected_items(query: str) -> int:
    """Count numbered items in query."""
    numbers = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s+", query)
    if numbers:
        return max(int(n) for n in numbers)
    return len([ln for ln in query.splitlines() if re.match(r"^\d+[\.\)]", ln.strip())])


def main():
    parser = argparse.ArgumentParser(description="Local 4060 llama.cpp CLI test")
    parser.add_argument("--model", type=str, required=True, help="Path to .gguf file")
    parser.add_argument("--data", type=str, default="data/linguini_test_sample.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--output", type=str, default="data/llamacpp_submission.csv")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    # Verify binary exists
    if not LLAMA_CLI.exists():
        print(f"ERROR: llama-cli.exe not found at {LLAMA_CLI}")
        print("Download from: https://github.com/ggerganov/llama.cpp/releases")
        print("Look for: llama-bXXXX-bin-win-cuda-x64.zip")
        sys.exit(1)

    df = pd.read_csv(args.data, dtype=str)
    if args.limit:
        df = df.head(args.limit)

    rows = []
    times = []

    print(f"Using llama.cpp: {LLAMA_CLI}")
    print(f"Model: {args.model}")
    print(f"Running {len(df)} problems...\n")

    for idx, r in df.iterrows():
        prompt = build_prompt(r["context"], r["query"], r.get("task_type", "unknown"))
        expected = count_expected_items(r["query"])

        start = time.time()
        text = run_llama_cli(args.model, prompt, max_tokens=args.max_tokens)
        elapsed = time.time() - start
        times.append(elapsed)

        answers = parse_answers(text, expected)
        pred = json.dumps(answers, ensure_ascii=False)
        rows.append({"id": str(r["id"]), "pred": pred})

        print(f"[{idx+1}/{len(df)}] {r['id']}: {len(answers)} answers in {elapsed:.1f}s")
        if idx < 3:
            print(f"  Raw:\n{text[:200]}\n")

    submission = pd.DataFrame(rows)
    submission.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSubmission saved to {args.output}")
    print(f"Avg time: {sum(times)/len(times):.1f}s | Total: {sum(times):.1f}s")

    if args.score:
        result = score_submission(args.output, args.data)
        print(f"Score: {result['score']:.2f} (EM={result['exact_match']:.4f}, chrF={result['chrF']:.4f})")


if __name__ == "__main__":
    main()

r"""
Local 4060 Test via llama.cpp CLI (Native, Pre-compiled CUDA)

Uses the actual llama.cpp binaries you already have:
    C:\Users\Keshav\Downloads\llama-b10064-bin-win-cuda-13.3-x64\llama-cli.exe

Optimized for speed: configurable threads, context, batch size, full GPU offload.
Rich progress tracking with ETA, per-problem timing, and error recovery.

Usage:
    python local_4060_llamacpp.py --model models\qwen3.5-4b-q4_k_m.gguf --limit 5 --score --verbose
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ── Auto-detect llama.cpp binary ─────────────────────────────────────────────
LLAMA_CLI_LOCAL = Path(__file__).parent / "llama" / "llama-cli.exe"
LLAMA_CLI_FALLBACK = Path(r"C:\Users\Keshav\Downloads\llama-b10064-bin-win-cuda-13.3-x64\llama-cli.exe")
LLAMA_CLI = LLAMA_CLI_LOCAL if LLAMA_CLI_LOCAL.exists() else LLAMA_CLI_FALLBACK

sys.path.insert(0, str(Path(__file__).parent / "src"))
from scorer import score_submission


def print_progress_bar(current: int, total: int, prefix: str = "", suffix: str = "", bar_len: int = 40):
    """Print a simple ASCII progress bar."""
    filled = int(bar_len * current // total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = 100 * current / total
    sys.stdout.write(f"\r{prefix} |{bar}| {pct:5.1f}% {suffix}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def run_llama_cli(
    model_path: str,
    prompt: str,
    max_tokens: int = 256,
    temp: float = 0.0,
    ngl: int = 999,
    threads: int = 8,
    ctx: int = 4096,
    batch: int = 512,
    verbose: bool = False,
) -> str:
    """Call llama-cli.exe as a subprocess. Returns generated text."""
    if not LLAMA_CLI.exists():
        raise FileNotFoundError(
            f"llama-cli.exe not found at {LLAMA_CLI}. "
            "Download from https://github.com/ggerganov/llama.cpp/releases"
        )

    cmd = [
        str(LLAMA_CLI),
        "-m", model_path,
        "-p", prompt,
        "-n", str(max_tokens),
        "--temp", str(temp),
        "-ngl", str(ngl),          # GPU layers
        "-t", str(threads),        # CPU threads
        "-c", str(ctx),            # context size
        "-b", str(batch),          # batch size
        "--no-display-prompt",     # don't echo prompt
        "--mlock",                 # keep model in RAM/VRAM
    ]

    if verbose:
        print(f"  [CMD] {' '.join(cmd[:10])} ... (truncated)")

    start_proc = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    proc_time = time.time() - start_proc

    if result.returncode != 0:
        err_preview = result.stderr[:300].replace("\n", " ")
        print(f"  [ERROR] llama-cli failed ({result.returncode}): {err_preview}")
    elif verbose and result.stderr:
        # llama.cpp prints info to stderr even on success
        info = [ln for ln in result.stderr.splitlines() if "offloaded" in ln.lower() or "alloc" in ln.lower()]
        if info:
            print(f"  [INFO] {info[-1]}")

    # Sometimes llama-cli prints performance stats at end of stderr
    if verbose and result.stderr:
        perf = [ln for ln in result.stderr.splitlines() if "ms/tok" in ln or "tokens/s" in ln]
        if perf:
            print(f"  [PERF] {perf[-1]}")

    return result.stdout.strip(), proc_time


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

    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{context.strip()}\n\n{query.strip()}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def parse_answers(text: str, expected_count: int = None) -> list:
    """Parse answer lines from model output."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    filtered = []
    for line in lines:
        low = line.lower()
        if low.startswith(("answer", "solution", "output", "result", "explanation", "note")):
            continue
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


def format_eta(seconds: float) -> str:
    """Format seconds into mm:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Local 4060 llama.cpp CLI test")
    parser.add_argument("--model", type=str, required=True, help="Path to .gguf file")
    parser.add_argument("--data", type=str, default="data/linguini_test_sample.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (default: auto)")
    parser.add_argument("--ngl", type=int, default=999, help="GPU layers to offload (default: 999=all)")
    parser.add_argument("--ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--output", type=str, default="data/llamacpp_submission.csv")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Show llama-cli commands and stderr")
    args = parser.parse_args()

    if not LLAMA_CLI.exists():
        print(f"ERROR: llama-cli.exe not found at {LLAMA_CLI}")
        print("Download from: https://github.com/ggerganov/llama.cpp/releases")
        print("Look for: llama-bXXXX-bin-win-cuda-x64.zip")
        sys.exit(1)

    threads = args.threads or min(os.cpu_count() or 8, 16)

    df = pd.read_csv(args.data, dtype=str)
    if args.limit:
        df = df.head(args.limit)
    total = len(df)

    print(f"=" * 60)
    print(f"llama.cpp : {LLAMA_CLI}")
    print(f"Model     : {args.model}")
    print(f"Problems  : {total}")
    print(f"Threads   : {threads}")
    print(f"GPU layers: {args.ngl}")
    print(f"Context   : {args.ctx}")
    print(f"Batch     : {args.batch}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"=" * 60)

    rows = []
    times = []
    errors = 0
    overall_start = time.time()

    for idx, r in df.iterrows():
        problem_id = str(r["id"])
        task_type = r.get("task_type", "unknown")
        expected = count_expected_items(r["query"])

        prompt = build_prompt(r["context"], r["query"], task_type)

        # ETA estimate
        if times:
            avg_sofar = sum(times) / len(times)
            remaining = (total - idx) * avg_sofar
            eta_str = format_eta(remaining)
            rate = 60.0 / avg_sofar if avg_sofar > 0 else 0
        else:
            eta_str = "??:??"
            rate = 0

        prefix = f"[{idx+1}/{total}] {problem_id} ({task_type}, {expected} items)"
        suffix = f"ETA {eta_str} @ {rate:.1f} prob/min"
        print_progress_bar(idx, total, prefix=prefix, suffix=suffix)

        try:
            text, proc_time = run_llama_cli(
                args.model, prompt,
                max_tokens=args.max_tokens,
                ngl=args.ngl,
                threads=threads,
                ctx=args.ctx,
                batch=args.batch,
                verbose=args.verbose,
            )
            elapsed = proc_time
            times.append(elapsed)

            answers = parse_answers(text, expected)
            pred = json.dumps(answers, ensure_ascii=False)
            rows.append({"id": problem_id, "pred": pred})

            # Inline result
            ans_preview = " | ".join(answers[:3])
            if len(answers) > 3:
                ans_preview += f" ... ({len(answers)} total)"
            print(f"  -> {len(answers)} answers in {elapsed:.1f}s | {ans_preview[:80]}")

            if args.verbose and text:
                print(f"  RAW: {text[:120].replace(chr(10), ' ')}")

        except Exception as e:
            errors += 1
            print(f"  -> FAILED: {e}")
            rows.append({"id": problem_id, "pred": json.dumps([""], ensure_ascii=False)})
            times.append(0.0)

    overall_elapsed = time.time() - overall_start
    print_progress_bar(total, total, prefix=f"[{total}/{total}] Done", suffix="")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(rows)
    submission.to_csv(args.output, index=False, encoding="utf-8")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUBMISSION : {args.output}")
    print(f"TOTAL TIME : {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"PROBLEMS   : {total} ({errors} errors)")
    if times and any(t > 0 for t in times):
        valid = [t for t in times if t > 0]
        print(f"AVG/PROB   : {sum(valid)/len(valid):.1f}s")
        print(f"MIN/MAX    : {min(valid):.1f}s / {max(valid):.1f}s")
        print(f"THROUGHPUT : {60.0*len(valid)/sum(valid):.1f} prob/min")
    print(f"{'='*60}")

    if args.score:
        result = score_submission(args.output, args.data)
        print(f"\nSCORE      : {result['score']:.2f}")
        print(f"  exact_match: {result['exact_match']:.4f}")
        print(f"  chrF       : {result['chrF']:.4f}")


if __name__ == "__main__":
    main()

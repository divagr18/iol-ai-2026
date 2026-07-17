r"""
Local 4060 Test via llama.cpp SERVER (Native, Pre-compiled CUDA)

Uses llama-server.exe to keep the model HOT in VRAM. Spawns the server once,
then sends HTTP requests per problem. Massive speedup vs spawning llama-cli
for every problem.

Server loads model once → ~10-20s startup, then each problem is ~1-3s.

Usage:
    python local_4060_llamaserver.py --model models\qwen3.5-4b-q4_k_m.gguf --limit 5 --score
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

# Auto-detect llama.cpp binary
LLAMA_SERVER_LOCAL = Path(__file__).parent / "llama" / "llama-server.exe"
LLAMA_SERVER_FB = Path(r"C:\Users\Keshav\Downloads\llama-b10064-bin-win-cuda-13.3-x64\llama-server.exe")
LLAMA_SERVER = LLAMA_SERVER_LOCAL if LLAMA_SERVER_LOCAL.exists() else LLAMA_SERVER_FB

sys.path.insert(0, str(Path(__file__).parent / "src"))
from scorer import score_submission

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def start_server(model_path: str, ngl: int = 999, threads: int = 4, ctx: int = 2048, batch: int = 256):
    """Start llama-server.exe in background. Returns Popen handle."""
    if not LLAMA_SERVER.exists():
        raise FileNotFoundError(
            f"llama-server.exe not found at {LLAMA_SERVER}. "
            "Download from https://github.com/ggerganov/llama.cpp/releases"
        )

    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "-ngl", str(ngl),
        "-t", str(threads),
        "-c", str(ctx),
        "-b", str(batch),
        "--mlock",
        "--no-webui",
        "--flash-attn", "auto",
    ]

    print(f"Starting server: {LLAMA_SERVER.name}")
    print(f"  Model : {model_path}")
    print(f"  GPU   : {ngl} layers")
    print(f"  Ctx   : {ctx}")
    print(f"  Port  : {SERVER_PORT}")
    print("  (This takes ~10-30s to load the model into VRAM...)")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    import threading
    server_output = []
    def _read_stdout():
        try:
            for line in iter(proc.stdout.readline, ""):
                server_output.append(line)
        except Exception:
            pass
    reader = threading.Thread(target=_read_stdout, daemon=True)
    reader.start()

    start_wait = time.time()
    max_wait = 180
    ready = False
    gpu_detected = False
    cpu_detected = False

    while time.time() - start_wait < max_wait:
        try:
            req = urllib.request.Request(f"{SERVER_URL}/health", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    ready = True
                    break
        except Exception:
            pass

        if proc.poll() is not None:
            print("\nERROR: Server crashed during startup!")
            print("Last 30 lines of output:")
            print("".join(server_output[-30:]))
            sys.exit(1)

        time.sleep(0.3)

    output_text = "".join(server_output)
    if "CUDA" in output_text or "GPU" in output_text or "offload" in output_text.lower():
        gpu_detected = True
    if "CPU" in output_text and not gpu_detected:
        cpu_detected = True

    if not ready:
        print("\nERROR: Server failed to start within 180s")
        print("Last 20 lines:")
        print("".join(server_output[-20:]))
        proc.terminate()
        sys.exit(1)

    elapsed = time.time() - start_wait
    print(f"Server ready in {elapsed:.1f}s")

    if cpu_detected and not gpu_detected:
        print("\n⚠️  WARNING: Model appears to be running on CPU, not GPU!")
        print("   Inference will be very slow (~30-60s per problem).")
        print("   Check that you have the CUDA build of llama.cpp and NVIDIA drivers.")
    elif gpu_detected:
        print("   GPU offload detected ✓")

    return proc


def stop_server(proc):
    """Gracefully stop the server."""
    print("\nStopping server...")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("Server stopped.")


def chat_completion(prompt: str, max_tokens: int = 256, temp: float = 0.0, timeout: int = 300) -> str:
    """Send a chat completion request to the running server."""
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temp,
        "stop": ["<|im_end|>", "<|im_start|>"],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("content", "").strip()


def build_prompt(context: str, query: str, task_type: str) -> str:
    system = (
        "You are an expert linguist solving International Linguistics Olympiad problems. "
        "Answer every numbered item. Put each answer on its own line, "
        "with NO numbering and NO extra text. "
        "NEVER show your reasoning, thinking, or analysis. "
        "Output ONLY the final answers, nothing else."
    )
    hints = {
        "translation": "Output only the translated sentence.",
        "matching": "Output only matched labels, one per line.",
        "fill_blanks": "Output only the missing form for each blank.",
        "text_to_num": "Output only the numeral.",
        "num_to_text": "Output only the written number form.",
    }
    h = hints.get(task_type, "")
    if h:
        system += f" {h}"

    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{context.strip()}\n\n{query.strip()}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def parse_answers(text: str, expected_count: int = None) -> list:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"Thinking Process:.*", "", text, flags=re.DOTALL)
    text = re.sub(r"\*\*Analyze the Request:\*\*.*", "", text, flags=re.DOTALL)

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


def count_expected_items(query: str) -> int:
    # Numbered items like "17." or "18)"
    numbers = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s+", query)
    if numbers:
        return max(int(n) for n in numbers)
    # Range notation like "(1-4)" or "(1–10)"
    range_match = re.search(r"\(\s*(\d+)\s*[-–—]\s*(\d+)\s*\)", query)
    if range_match:
        return int(range_match.group(2)) - int(range_match.group(1)) + 1
    # Count lines starting with numbers
    return len([ln for ln in query.splitlines() if re.match(r"^\d+[\.\)]", ln.strip())])


def print_progress_bar(current: int, total: int, prefix: str = "", suffix: str = "", bar_len: int = 40):
    filled = int(bar_len * current // total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = 100 * current / total
    sys.stdout.write(f"\r{prefix} |{bar}| {pct:5.1f}% {suffix}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def format_eta(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Local 4060 llama.cpp SERVER test")
    parser.add_argument("--model", type=str, required=True, help="Path to .gguf file")
    parser.add_argument("--data", type=str, default="data/linguini_test_sample.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--output", type=str, default="data/llamaserver_submission.csv")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.data, dtype=str)
    if args.limit:
        df = df.head(args.limit)
    total = len(df)

    # Start server
    server_proc = start_server(
        args.model,
        ngl=args.ngl,
        threads=args.threads,
        ctx=args.ctx,
        batch=args.batch,
    )

    try:
        rows = []
        times = []
        errors = 0
        overall_start = time.time()

        print(f"\nRunning {total} problems...\n")

        for idx, r in df.iterrows():
            problem_id = str(r["id"])
            task_type = r.get("task_type", "unknown")
            expected = count_expected_items(r["query"])
            prompt = build_prompt(r["context"], r["query"], task_type)

            # ETA
            if times:
                avg = sum(times) / len(times)
                remaining = (total - idx) * avg
                eta_str = format_eta(remaining)
                rate = 60.0 / avg if avg > 0 else 0
            else:
                eta_str = "??:??"
                rate = 0

            prefix = f"[{idx+1}/{total}] {problem_id} ({task_type}, {expected} items)"
            suffix = f"ETA {eta_str} @ {rate:.1f} prob/min"
            print_progress_bar(idx, total, prefix=prefix, suffix=suffix)

            try:
                start = time.time()
                text = chat_completion(prompt, max_tokens=args.max_tokens)
                elapsed = time.time() - start
                times.append(elapsed)

                answers = parse_answers(text, expected)
                pred = json.dumps(answers, ensure_ascii=False)
                rows.append({"id": problem_id, "pred": pred})

                preview = " | ".join(answers[:3])
                if len(answers) > 3:
                    preview += f" ... ({len(answers)} total)"
                print(f"  -> {len(answers)} answers in {elapsed:.1f}s | {preview[:80]}")

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
        print(f"SERVER LOAD: included in total")
        print(f"PROBLEMS   : {total} ({errors} errors)")
        if times and any(t > 0 for t in times):
            valid = [t for t in times if t > 0]
            print(f"AVG/PROB   : {sum(valid)/len(valid):.1f}s (excl. server load)")
            print(f"MIN/MAX    : {min(valid):.1f}s / {max(valid):.1f}s")
            print(f"THROUGHPUT : {60.0*len(valid)/sum(valid):.1f} prob/min")
        print(f"{'='*60}")

        if args.score:
            result = score_submission(args.output, args.data)
            print(f"\nSCORE      : {result['score']:.2f}")
            print(f"  exact_match: {result['exact_match']:.4f}")
            print(f"  chrF       : {result['chrF']:.4f}")

    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    main()

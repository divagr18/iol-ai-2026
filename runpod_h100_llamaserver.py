#!/usr/bin/env python3
"""
RunPod H100 llama.cpp SERVER Test (Linux)

Uses llama-server with OpenAI-compatible /v1/chat/completions endpoint.
Model stays HOT in VRAM for fast iteration.

Usage:
    python runpod_h100_llamaserver.py --model models/gemma-4-12b-it-q4_k_m.gguf --limit 5 --score
    python runpod_h100_llamaserver.py --model models/gemma-4-12b-it-q4_k_m.gguf --limit 40 --score --use_analysis
"""

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

# RunPod H100: uses Docker image instead of local binary
LLAMA_SERVER = Path("/workspace/llama.cpp/build/bin/llama-server")

sys.path.insert(0, str(Path(__file__).parent / "src"))
from scorer import score_submission
from analyzers import analyze_problem

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def start_server(model_path: str, ngl: int = 999, threads: int = 16, ctx: int = 4096, batch: int = 2048, reasoning: str = "off", parallel: int = 4):
    if not LLAMA_SERVER.exists():
        raise FileNotFoundError(
            f"llama-server not found at {LLAMA_SERVER}. "
            "Run: bash scripts/runpod_h100_llamaserver_setup.sh"
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
        "-np", str(parallel),
        "--mlock",
        "--no-webui",
        "--flash-attn", "auto",
        "--reasoning", reasoning,
    ]

    print(f"Starting server: {LLAMA_SERVER.name}")
    print(f"  Model : {model_path}")
    print(f"  GPU   : {ngl} layers (H100)")
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
    print("\nStopping server...")
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    print("Server stopped.")


def chat_completion(messages: list, max_tokens: int = 512, temp: float = 0.0, timeout: int = 300) -> str:
    """Send OpenAI-compatible chat completion to /v1/chat/completions."""
    payload = {
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    choices = result.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content", "").strip()
        reasoning = msg.get("reasoning_content", "").strip()
        if content:
            return content
        elif reasoning:
            return reasoning
    return ""


def build_messages(context: str, query: str, task_type: str, analysis_text: str = "") -> list:
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

    user_content = context.strip()
    if analysis_text:
        # Truncate if too long to prevent context overflow
        if len(analysis_text) > 1500:
            analysis_text = analysis_text[:1500] + "..."
        user_content += f"\n\nLinguistic analysis (mined from the data):\n{analysis_text}"
    user_content += f"\n\n{query.strip()}"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def parse_answers(text: str, expected_count: int = None) -> list:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Thinking Process:.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\*\*Analyze the Request:\*\*.*", "", text, flags=re.DOTALL | re.IGNORECASE)
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


def count_expected_items(query: str) -> int:
    numbers = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s+", query)
    if numbers:
        return max(int(n) for n in numbers)
    range_match = re.search(r"\(\s*(\d+)\s*[-–—]\s*(\d+)\s*\)", query)
    if range_match:
        return int(range_match.group(2)) - int(range_match.group(1)) + 1
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
    parser = argparse.ArgumentParser(description="RunPod H100 llama.cpp SERVER test")
    parser.add_argument("--model", type=str, required=True, help="Path to .gguf file")
    parser.add_argument("--data", type=str, default="data/linguini_test_sample.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per problem (default 512)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--ngl", type=int, default=999)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--output", type=str, default="data/h100_submission.csv")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--use_analysis", action="store_true", help="Inject deterministic linguistic analyzers into prompt")
    parser.add_argument("--reasoning", type=str, choices=["on", "off"], default="off", help="Enable model thinking/reasoning (on=generate thinking tokens, off=final answers only)")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel slots in llama-server (-np)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent HTTP requests (should match --parallel)")
    args = parser.parse_args()

    df = pd.read_csv(args.data, dtype=str)
    if args.limit:
        df = df.head(args.limit)
    total = len(df)

    server_proc = start_server(
        args.model,
        ngl=args.ngl,
        threads=args.threads,
        ctx=args.ctx,
        batch=args.batch,
        reasoning=args.reasoning,
        parallel=args.parallel,
    )

    def solve_one(idx_r):
        idx, r = idx_r
        problem_id = str(r["id"])
        task_type = r.get("task_type", "unknown")
        expected = count_expected_items(r["query"])
        if args.use_analysis:
            analysis_text = analyze_problem(r["context"], r["query"], task_type)
        else:
            analysis_text = ""
        messages = build_messages(r["context"], r["query"], task_type, analysis_text)
        try:
            start = time.time()
            text = chat_completion(messages, max_tokens=args.max_tokens)
            elapsed = time.time() - start
            answers = parse_answers(text, expected)
            pred = json.dumps(answers, ensure_ascii=False)
            return {"id": problem_id, "pred": pred, "elapsed": elapsed, "answers": answers, "error": None}
        except Exception as e:
            return {"id": problem_id, "pred": json.dumps([""], ensure_ascii=False), "elapsed": 0.0, "answers": [], "error": str(e)}

    try:
        rows = []
        times = []
        errors = 0
        overall_start = time.time()

        print(f"\nRunning {total} problems with {args.workers} concurrent workers...\n")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(solve_one, (idx, r)): idx for idx, r in df.iterrows()}
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                completed += 1
                res = future.result()
                rows.append({"id": res["id"], "pred": res["pred"]})
                if res["error"]:
                    errors += 1
                    print(f"  [{completed}/{total}] {res['id']} -> FAILED: {res['error']}")
                else:
                    times.append(res["elapsed"])
                    preview = " | ".join(res["answers"][:3])
                    if len(res["answers"]) > 3:
                        preview += f" ... ({len(res['answers'])} total)"
                    print(f"  [{completed}/{total}] {res['id']} -> {len(res['answers'])} answers in {res['elapsed']:.1f}s | {preview[:80]}")

        overall_elapsed = time.time() - overall_start

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        submission = pd.DataFrame(rows)
        submission.to_csv(args.output, index=False, encoding="utf-8")

        print(f"\n{'='*60}")
        print(f"SUBMISSION : {args.output}")
        print(f"TOTAL TIME : {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
        print(f"SERVER LOAD: included in total")
        print(f"PROBLEMS   : {total} ({errors} errors)")
        if times and any(t > 0 for t in times):
            valid = [t for t in times if t > 0]
            print(f"AVG/PROB   : {sum(valid)/len(valid):.1f}s (wall-clock per batch)")
            print(f"MIN/MAX    : {min(valid):.1f}s / {max(valid):.1f}s")
            print(f"THROUGHPUT : {60.0*len(valid)/sum(valid):.1f} prob/min")
            print(f"WALL-CLOCK : {60.0*total/overall_elapsed:.1f} prob/min (includes all parallel work)")
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

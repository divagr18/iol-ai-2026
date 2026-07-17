"""
IOL-AI 2026 Submission Script — llama-server edition

Uses llama-server with OpenAI-compatible /v1/chat/completions endpoint.
Model (GGUF) stays HOT in VRAM for fast batched inference.

No torch/transformers/autoawq dependency hell — just a single binary + weights.
"""

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

LLAMA_SERVER = Path(__file__).parent / "llama-server"
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

MODEL_GGUF = os.environ.get("MODEL_GGUF", "models/gemma-4-12b-it-q4_k_m.gguf")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
NG_LAYERS = int(os.environ.get("NG_LAYERS", "999"))
THREADS = int(os.environ.get("THREADS", "8"))
CTX = int(os.environ.get("CTX", "4096"))
BATCH = int(os.environ.get("BATCH", "2048"))
PARALLEL = int(os.environ.get("PARALLEL", "4"))


def start_server(model_path: str):
    if not LLAMA_SERVER.exists():
        raise FileNotFoundError(
            f"llama-server not found at {LLAMA_SERVER}. "
            "It must be built from source or extracted from a release."
        )

    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "-ngl", str(NG_LAYERS),
        "-t", str(THREADS),
        "-c", str(CTX),
        "-b", str(BATCH),
        "-np", str(PARALLEL),
        "--mlock",
        "--no-webui",
        "--flash-attn", "auto",
        "--reasoning", "off",
    ]

    print(f"Starting llama-server with {model_path}", flush=True)
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
            print("ERROR: Server crashed during startup!", flush=True)
            print("".join(server_output[-30:]), flush=True)
            sys.exit(1)
        time.sleep(0.3)

    if not ready:
        print("ERROR: Server failed to start within 180s", flush=True)
        print("".join(server_output[-20:]), flush=True)
        proc.terminate()
        sys.exit(1)

    print(f"Server ready in {time.time() - start_wait:.1f}s", flush=True)
    return proc


def stop_server(proc):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def chat_completion(messages: list, max_tokens: int = 512, temp: float = 0.0, timeout: int = 300) -> str:
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
        return content if content else reasoning
    return ""


def build_messages(context: str, query: str, task_type: str) -> list:
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
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{context.strip()}\n\n{query.strip()}"},
    ]


def parse_answers(text: str, expected_count: int = None) -> list:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Thinking Process:.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\*\*.*?

\*\*", "", text, flags=re.DOTALL)

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

    # Resolve model path
    model_path = Path(MODEL_GGUF)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}", flush=True)
        sys.exit(1)

    server_proc = start_server(str(model_path))
    try:
        rows = []
        total = len(df)
        for idx, r in df.iterrows():
            problem_id = str(r["id"])
            context = str(r.get("context", ""))
            query = str(r.get("query", ""))
            task_type = str(r.get("task_type", "unknown"))
            expected = count_expected_items(query)

            messages = build_messages(context, query, task_type)
            text = chat_completion(messages, max_tokens=MAX_TOKENS)
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
    finally:
        stop_server(server_proc)


if __name__ == "__main__":
    main()

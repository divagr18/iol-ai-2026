"""
IOL-AI 2026 Submission Script v1.0

Production-ready with:
  - task-type-aware prompting enriched by deterministic linguistic analyzers
  - neuro-symbolic verification with repair loop
  - jury-track explanation generation
  - global time budget allocator
  - offline-only mode for eval sandbox
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Offline enforcement
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Import local analysis modules (shipped in repo)
sys.path.insert(0, str(Path(__file__).parent))
from analyzers import analyze_problem
from verifiers import verify_answers
from explanation import generate_explanation, format_explanation_for_csv

# Configuration
MODEL_ID = os.environ.get("MODEL_ID", ".")
DEVICE = os.environ.get("DEVICE", "auto")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))
DO_SAMPLE = os.environ.get("DO_SAMPLE", "false").lower() == "true"
USE_EXPLANATION = os.environ.get("USE_EXPLANATION", "true").lower() == "true"
MAX_REPAIR_ATTEMPTS = int(os.environ.get("MAX_REPAIR", "1"))
SEED = int(os.environ.get("SEED", "42"))

torch.manual_seed(SEED)

# System prompt templates
SYSTEM_BASE = (
    "You are an expert linguist solving International Linguistics Olympiad problems. "
    "You reason carefully from the data provided, never using outside knowledge. "
    "You must answer EVERY numbered item in the query. "
    "Put each answer on its own line, in the same order as the query, with NO numbering and NO extra text. "
    "NEVER show your reasoning, thinking, or analysis. "
    "Output ONLY the final answers, nothing else."
)

TASK_PROMPTS = {
    "translation": SYSTEM_BASE + " For translation problems, output only the translated sentence for each item.",
    "matching": SYSTEM_BASE + " For matching problems, output only the matched labels (e.g., letters or numbers) in order, separated by commas if multiple items, one per line if single items.",
    "fill_blanks": SYSTEM_BASE + " For fill-in-the-blanks problems, output only the missing form for each blank.",
    "text_to_num": SYSTEM_BASE + " For number transliteration problems, output only the numeral for each item.",
    "num_to_text": SYSTEM_BASE + " For number-to-text problems, output only the written number form.",
    "unknown": SYSTEM_BASE,
}


def load_model():
    print(f"Loading model from {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": DEVICE,
    }

    config_path = Path(MODEL_ID) / "config.json"
    quant_type = ""
    if config_path.exists():
        import json as _json
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        quant_type = cfg.get("quantization_config", {}).get("quant_method", "")

    env_quant = os.environ.get("QUANT", "").lower()

    if quant_type == "awq" or env_quant == "awq":
        print("Detected AWQ quantization", flush=True)
    elif quant_type == "gptq" or env_quant == "gptq":
        print("Detected GPTQ quantization", flush=True)
    elif env_quant == "bnb4":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        )
        print("Using bitsandbytes 4-bit", flush=True)
    elif env_quant == "bnb8":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print("Using bitsandbytes 8-bit", flush=True)
    else:
        print("Using default fp16 loading", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    model.eval()
    print("Model loaded", flush=True)
    return tok, model


def build_messages(context, query, task_type, analysis_text="", use_explanation=False, repair_instruction=""):
    system = TASK_PROMPTS.get(task_type, TASK_PROMPTS["unknown"])
    # Truncate analysis if too long (prevent context overflow)
    if analysis_text and len(analysis_text) > 2000:
        analysis_text = analysis_text[:2000] + "..."
    user_parts = [context.strip()]
    if analysis_text:
        user_parts.append(f"\nLinguistic analysis:\n{analysis_text}")
    user_parts.append(f"\n{query.strip()}")
    if repair_instruction:
        user_parts.append(f"\nCorrection needed: {repair_instruction}")
    if use_explanation:
        user_parts.append("\nAfter your answers, add a brief 2-sentence explanation of the rule you found.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def apply_chat_template_safe(tok, messages, max_length=8192):
    """Apply chat template with fallback for models without templates."""
    try:
        # Qwen3.5: disable reasoning/thinking blocks
        kwargs = {"add_generation_prompt": True, "return_tensors": "pt", "tokenize": True}
        if hasattr(tok, "apply_chat_template"):
            # Try passing enable_thinking=False (Qwen3 specific)
            try:
                test = tok.apply_chat_template([{"role": "user", "content": "hi"}], chat_template_kwargs={"enable_thinking": False})
                kwargs["chat_template_kwargs"] = {"enable_thinking": False}
            except TypeError:
                pass  # Model doesn't support this kwarg
        inputs = tok.apply_chat_template(messages, **kwargs)
        if isinstance(inputs, dict):
            inputs = inputs["input_ids"]
    except Exception as e:
        text_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                text_parts.append(f"System: {content}")
            elif role == "user":
                text_parts.append(f"User: {content}")
            elif role == "assistant":
                text_parts.append(f"Assistant: {content}")
        text_parts.append("Assistant:")
        text = "\n\n".join(text_parts)
        inputs = tok(text, return_tensors="pt")["input_ids"]
    return inputs


def parse_answers(text, expected_count=None):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Thinking Process:.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\*\*Analyze the Request:\*\*.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\*\*.*?\*\*", "", text, flags=re.DOTALL)
    lines = text.splitlines()
    answers = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(answers?|solution|output|result)s?[:\s]*$", line, re.I):
            continue
        if re.match(r"^(explanation|reasoning|note|thinking)[:\s]*$", line, re.I):
            break
        line = re.sub(r"\*\*", "", line)
        line = re.sub(r"\*", "", line)
        cleaned = re.sub(r"^\s*(\d+[\.\)]\s+|\w[\.\)]\s+|-\s+)", "", line)
        if cleaned:
            answers.append(cleaned)
    if not answers:
        answers = [ln.strip() for ln in lines if ln.strip()]
    if expected_count is not None:
        while len(answers) < expected_count:
            answers.append("")
        answers = answers[:expected_count]
    return answers


def extract_explanation(text):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^(explanation|reasoning|note)[:\s]*$", line, re.I):
            return "\n".join(lines[i + 1:]).strip()
    return ""


def count_expected_items(query):
    numbers = re.findall(r"(?:^|\n)\s*(\d+)[\.\)]\s+", query)
    if numbers:
        return max(int(n) for n in numbers)
    range_match = re.search(r"\(\s*(\d+)\s*[-–—]\s*(\d+)\s*\)", query)
    if range_match:
        return int(range_match.group(2)) - int(range_match.group(1)) + 1
    lines = [ln.strip() for ln in query.splitlines() if ln.strip()]
    count = sum(1 for ln in lines if re.match(r"^\d+[\.\)]", ln))
    return count if count > 0 else None


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
    start_time = time.time()
    total_budget = 25 * 60

    for idx, r in df.iterrows():
        problem_id = str(r["id"])
        context = str(r.get("context", ""))
        query = str(r.get("query", ""))
        task_type = str(r.get("task_type", "unknown"))
        eval_type = str(r.get("eval_type", "single"))

        elapsed = time.time() - start_time
        remaining = total_budget - elapsed
        per_problem = remaining / max(len(df) - idx, 1)
        max_new = min(MAX_NEW_TOKENS, 128) if per_problem < 10 else MAX_NEW_TOKENS

        analysis_text = analyze_problem(context, query, task_type)

        messages = build_messages(
            context, query, task_type,
            analysis_text=analysis_text,
            use_explanation=USE_EXPLANATION,
        )

        def generate(messages, max_tokens):
            try:
                inputs = apply_chat_template_safe(tok, messages)
                device = next(model.parameters()).device
                inputs = inputs.to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=max_tokens,
                        do_sample=DO_SAMPLE,
                        temperature=TEMPERATURE if DO_SAMPLE else None,
                        top_p=TOP_P if DO_SAMPLE else None,
                        pad_token_id=tok.pad_token_id,
                    )
                new_tokens = outputs[0][inputs.shape[-1]:]
                return tok.decode(new_tokens, skip_special_tokens=True).strip()
            except torch.cuda.OutOfMemoryError as oom:
                print(f"OOM for {problem_id}, clearing cache", flush=True)
                torch.cuda.empty_cache()
                return ""
            except Exception as e:
                print(f"ERROR generating for {problem_id}: {e}", flush=True)
                return ""

        text = generate(messages, max_new)
        expected_count = count_expected_items(query)
        answers = parse_answers(text, expected_count=expected_count)

        verified, corrected, verifier_msg = verify_answers(
            context, query, answers, task_type, eval_type, expected_count,
        )

        if not verified and MAX_REPAIR_ATTEMPTS > 0 and per_problem > 5:
            messages = build_messages(
                context, query, task_type,
                analysis_text=analysis_text,
                use_explanation=False,
                repair_instruction=verifier_msg,
            )
            text2 = generate(messages, max(max_new // 2, 64))
            answers2 = parse_answers(text2, expected_count=expected_count)
            verified2, corrected2, verifier_msg2 = verify_answers(
                context, query, answers2, task_type, eval_type, expected_count,
            )
            if verified2 or len(answers2) >= len(answers):
                answers = answers2
                verified = verified2
                verifier_msg = verifier_msg2
                text = text2

        explanation = ""
        if USE_EXPLANATION:
            explanation = generate_explanation(
                context, query, task_type, answers,
                analysis_text, verified, verifier_msg,
            )
            explanation = format_explanation_for_csv(explanation)

        pred = json.dumps(answers, ensure_ascii=False)
        row = {"id": problem_id, "pred": pred}
        if USE_EXPLANATION:
            row["explanation"] = explanation
        rows.append(row)

        print(
            f"  [{idx + 1}/{len(df)}] {problem_id}: {len(answers)} answers "
            f"(verified={verified}, repair={not verified and MAX_REPAIR_ATTEMPTS > 0})",
            flush=True,
        )

    submission = pd.DataFrame(rows)
    submission.to_csv("submission.csv", index=False, encoding="utf-8")
    print(f"Wrote submission.csv with {len(submission)} rows", flush=True)

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()

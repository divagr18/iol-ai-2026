"""
IOL-AI 2026 Local Test Harness

Downloads the Linguini dataset (facebook/linguini), formats it as mock test.csv
matching the competition schema, runs script.py against it, and scores answers.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from scorer import score_submission


def load_linguini(split: str = "test") -> pd.DataFrame:
    """Load Linguini dataset from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    ds = load_dataset("facebook/linguini", split=split)
    return ds.to_pandas()


def linguini_to_competition_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Linguini format to IOL-AI competition schema.
    Linguini columns likely include: id, context, question, answer, language, task_type, etc.
    We normalize to: id, context, query, work_lang, task_lang, task_type, eval_type
    """
    # Linguini schema detection
    cols = set(df.columns)

    # Map columns
    out = pd.DataFrame()
    out["id"] = df["id"].astype(str)

    # context: problem statement / bilingual data
    if "context" in cols:
        out["context"] = df["context"].fillna("").astype(str)
    elif "problem" in cols:
        out["context"] = df["problem"].fillna("").astype(str)
    else:
        # Fallback: concatenate relevant columns
        context_parts = []
        for c in ["instruction", "examples", "data"]:
            if c in cols:
                context_parts.append(df[c].fillna("").astype(str))
        out["context"] = context_parts[0] if context_parts else ""

    # query: the question / items to answer
    if "query" in cols:
        out["query"] = df["query"].fillna("").astype(str)
    elif "question" in cols:
        out["query"] = df["question"].fillna("").astype(str)
    elif "items" in cols:
        out["query"] = df["items"].fillna("").astype(str)
    else:
        out["query"] = ""

    # work_lang: language of instructions (usually English)
    out["work_lang"] = "eng_Latn"

    # task_lang: the problem language
    if "language" in cols:
        out["task_lang"] = df["language"].fillna("unk").astype(str)
    elif "task_lang" in cols:
        out["task_lang"] = df["task_lang"].fillna("unk").astype(str)
    else:
        out["task_lang"] = "unk"

    # task_type
    if "task_type" in cols:
        out["task_type"] = df["task_type"].fillna("unknown").astype(str)
    elif "type" in cols:
        out["task_type"] = df["type"].fillna("unknown").astype(str)
    else:
        out["task_type"] = "unknown"

    # eval_type: single vs multi
    if "eval_type" in cols:
        out["eval_type"] = df["eval_type"].fillna("single").astype(str)
    else:
        out["eval_type"] = "single"

    return out


def save_mock_test(df: pd.DataFrame, path: Path):
    """Save DataFrame as test.csv in competition format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def run_script(script_path: Path, test_csv: Path, output_dir: Path, env: Optional[Dict[str, str]] = None):
    """Run script.py in a subprocess with test.csv mounted at /tmp/data/test.csv."""
    # Create a temp dir structure mimicking /tmp/data
    tmp_data = output_dir / "tmp_data"
    tmp_data.mkdir(parents=True, exist_ok=True)
    test_link = tmp_data / "test.csv"
    if test_link.exists() or test_link.is_symlink():
        test_link.unlink()
    # On Windows, use copy; on Unix, symlink
    import shutil
    shutil.copy2(test_csv, test_link)

    env_vars = os.environ.copy()
    env_vars["HF_HUB_OFFLINE"] = "1"
    env_vars["TRANSFORMERS_OFFLINE"] = "1"
    if env:
        env_vars.update(env)

    cmd = [sys.executable, str(script_path)]
    subprocess.run(cmd, cwd=str(output_dir), env=env_vars, check=True)


def build_gold_csv(df: pd.DataFrame, path: Path):
    """Build a gold answers CSV from Linguini for local scoring."""
    gold = pd.DataFrame()
    gold["id"] = df["id"].astype(str)
    if "answer" in df.columns:
        gold["answer"] = df["answer"].fillna("").astype(str)
    elif "gold" in df.columns:
        gold["answer"] = df["gold"].fillna("").astype(str)
    else:
        gold["answer"] = ""
    if "eval_type" in df.columns:
        gold["eval_type"] = df["eval_type"].fillna("single").astype(str)
    else:
        gold["eval_type"] = "single"
    gold.to_csv(path, index=False, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Local IOL-AI harness")
    parser.add_argument("--script", type=Path, default=Path("script.py"), help="Path to script.py")
    parser.add_argument("--model_id", type=str, default=".", help="Model ID or path")
    parser.add_argument("--split", type=str, default="test", help="Linguini split")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--output", type=Path, default=Path("data/run_output"), help="Output directory")
    parser.add_argument("--no_score", action="store_true", help="Skip scoring (for smoke test)")
    args = parser.parse_args()

    print("Loading Linguini dataset...")
    df = load_linguini(args.split)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} problems")

    print("Converting to competition schema...")
    comp_df = linguini_to_competition_schema(df)

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    test_csv = out_dir / "test.csv"
    save_mock_test(comp_df, test_csv)
    print(f"Mock test set written to {test_csv}")

    gold_csv = out_dir / "gold.csv"
    build_gold_csv(df, gold_csv)
    print(f"Gold answers written to {gold_csv}")

    # Also save a copy of the raw sample for reference
    df.to_csv(out_dir / "linguini_raw.csv", index=False, encoding="utf-8")

    # Run script
    print(f"Running {args.script}...")
    env = {"MODEL_ID": args.model_id}
    run_script(args.script, test_csv, out_dir, env=env)

    submission_csv = out_dir / "submission.csv"
    if not submission_csv.exists():
        print(f"ERROR: {submission_csv} not found after script run")
        return 1

    print(f"Submission written to {submission_csv}")

    if not args.no_score and gold_csv.exists():
        print("Scoring...")
        result = score_submission(submission_csv, gold_csv)
        print(f"Score: {result['score']:.2f} (EM={result['exact_match']:.4f}, chrF={result['chrF']:.4f})")

        # Save detailed results
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

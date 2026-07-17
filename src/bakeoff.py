"""
H100 Bake-off Harness

Batch-evaluate multiple model configs on the same Linguini subset and
produce a comparison table with score, EM, chrF, time, and peak VRAM.

Usage:
    python -m src.bakeoff --models configs/bakeoff_models.json --limit 30
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def run_single_model(model_cfg: dict, limit: int, split: str = "test") -> dict:
    """Run harness for one model config and return metrics."""
    model_id = model_cfg["model_id"]
    quant = model_cfg.get("quant", "")
    name = model_cfg.get("name", model_id.split("/")[-1])

    output_dir = Path(f"data/bakeoff/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "MODEL_ID": model_id,
        "QUANT": quant,
        "MAX_NEW_TOKENS": str(model_cfg.get("max_new_tokens", 512)),
        "SEED": "42",
    }

    cmd = [
        sys.executable, "-m", "src.harness",
        "--script", "script.py",
        "--model_id", model_id,
        "--split", split,
        "--limit", str(limit),
        "--output", str(output_dir),
    ]

    print(f"\n{'='*60}")
    print(f"Running: {name} ({model_id}, quant={quant})")
    print(f"{'='*60}")

    start = time.time()
    try:
        subprocess.run(cmd, env={**dict(os.environ), **env}, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {name} -> {e}")
        return {
            "name": name,
            "model_id": model_id,
            "quant": quant,
            "score": 0.0,
            "exact_match": 0.0,
            "chrF": 0.0,
            "time_sec": 0.0,
            "status": "FAILED",
        }
    elapsed = time.time() - start

    # Load results
    results_path = output_dir / "results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            res = json.load(f)
    else:
        res = {"score": 0.0, "exact_match": 0.0, "chrF": 0.0}

    return {
        "name": name,
        "model_id": model_id,
        "quant": quant,
        "score": res.get("score", 0.0),
        "exact_match": res.get("exact_match", 0.0),
        "chrF": res.get("chrF", 0.0),
        "time_sec": elapsed,
        "status": "OK",
    }


def main():
    parser = argparse.ArgumentParser(description="Model bake-off harness")
    parser.add_argument("--config", type=Path, required=True, help="JSON config file with model list")
    parser.add_argument("--limit", type=int, default=30, help="Number of problems to evaluate")
    parser.add_argument("--split", type=str, default="test", help="Linguini split")
    parser.add_argument("--output", type=Path, default=Path("data/bakeoff/summary.csv"), help="Summary CSV path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        configs = json.load(f)

    results = []
    for cfg in configs:
        res = run_single_model(cfg, args.limit, args.split)
        results.append(res)
        # Print running summary
        print(f"  -> Score={res['score']:.2f} EM={res['exact_match']:.4f} chrF={res['chrF']:.4f} "
              f"Time={res['time_sec']:.1f}s Status={res['status']}")

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nSummary written to {args.output}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    import os
    main()

"""
IOL-AI 2026 Scorer

Replicates the competition metric:
    score = 100 * sqrt(weighted_exact_match * weighted_chrF)

Where weights are the official IOL point values per item.
For local development without point weights, uniform weighting is used
or point weights can be supplied via a CSV/JSON file.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sacrebleu import CHRF


def parse_pred(raw: str) -> List[str]:
    """Parse a prediction cell: JSON list or comma-separated fallback."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass
    # Fallback: comma-separated, stripping numbers if needed
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def parse_gold(raw: str, eval_type: str = "single") -> List[List[str]]:
    """Parse gold answer.
    For single: one expected answer per item.
    For multi: semicolon-separated alternatives.
    Handles both JSON (double quotes) and Python list literals (single quotes).
    """
    raw = raw.strip()
    if not raw:
        return [[]]
    # Try JSON first
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            result = []
            for p in parsed:
                if isinstance(p, list):
                    result.append([str(x).strip() for x in p])
                else:
                    result.append([str(p).strip()])
            return result
    except Exception:
        pass
    # Try Python literal eval (for single-quoted lists)
    try:
        import ast
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            result = []
            for p in parsed:
                if isinstance(p, list):
                    result.append([str(x).strip() for x in p])
                else:
                    result.append([str(p).strip()])
            return result
    except Exception:
        pass
    # Plain text fallback: semicolon separates alternatives for multi
    if eval_type in ("multi", "simple") and ";" in raw:
        items = raw.split(",")
        result = []
        for item in items:
            alts = [a.strip() for a in item.split(";") if a.strip()]
            result.append(alts)
        return result
    parts = [p.strip() for p in raw.split(",")]
    return [[p] for p in parts if p]


def exact_match(pred: str, gold_alternatives: List[str]) -> bool:
    """Check if prediction exactly matches any gold alternative."""
    pred_norm = pred.strip()
    return any(pred_norm == g.strip() for g in gold_alternatives)


def chrf_score(preds: List[str], golds: List[str]) -> float:
    """Compute chrF score between prediction list and gold list (one item each)."""
    if not preds or not golds:
        return 0.0
    # sacrebleu expects list of hypotheses and list of references
    chrf = CHRF()
    # We score item-by-item; here preds and golds are single strings
    score = chrf.corpus_score(preds, [golds])
    return score.score / 100.0  # sacrebleu returns 0-100, normalize to 0-1


def score_row(
    pred_answers: List[str],
    gold_answers: List[List[str]],
    weights: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """
    Score a single problem row.
    Returns: (exact_match_score, chrF_score, final_score)
    All on 0-1 scale (before the 100x multiplier for display).
    """
    n = max(len(pred_answers), len(gold_answers))
    if weights is None:
        weights = [1.0] * n
    total_weight = sum(weights)

    w_em = 0.0
    w_chrf = 0.0

    for i in range(n):
        pred = pred_answers[i] if i < len(pred_answers) else ""
        gold_alts = gold_answers[i] if i < len(gold_answers) else [""]
        w = weights[i] if i < len(weights) else 1.0

        em = 1.0 if exact_match(pred, gold_alts) else 0.0
        w_em += w * em

        # chrF between pred and each alternative, take best
        best_chrf = max(
            chrf_score([pred], [alt]) for alt in gold_alts
        ) if gold_alts else 0.0
        w_chrf += w * best_chrf

    if total_weight == 0:
        return 0.0, 0.0, 0.0

    w_em /= total_weight
    w_chrf /= total_weight
    final = math.sqrt(w_em * w_chrf) if (w_em * w_chrf) >= 0 else 0.0
    return w_em, w_chrf, final


def score_submission(
    pred_path: Path,
    gold_path: Path,
    weights_path: Optional[Path] = None,
) -> Dict:
    """
    Score a full submission CSV against gold answers CSV.
    Both CSVs must have columns: id, pred (, optional explanation)
    Gold CSV should have: id, answer (, eval_type)
    """
    # Load predictions
    preds = {}
    with open(pred_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds[row["id"]] = parse_pred(row.get("pred", ""))

    # Load golds
    golds = {}
    eval_types = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            golds[row["id"]] = parse_gold(row.get("answer", ""), row.get("eval_type", "single"))
            eval_types[row["id"]] = row.get("eval_type", "single")

    # Load weights if provided
    weights = {}
    if weights_path and weights_path.exists():
        with open(weights_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                weights[row["id"]] = json.loads(row.get("weights", "[1.0]"))

    # Score each row
    total_weight = 0.0
    w_em_sum = 0.0
    w_chrf_sum = 0.0
    row_scores = []

    common_ids = sorted(set(preds.keys()) & set(golds.keys()))

    for pid in common_ids:
        w = weights.get(pid)
        em, chrf, final = score_row(preds[pid], golds[pid], w)
        # For aggregation, weight by problem if available, else uniform
        problem_weight = sum(w) if w else max(len(preds[pid]), len(golds[pid]))
        total_weight += problem_weight
        w_em_sum += em * problem_weight
        w_chrf_sum += chrf * problem_weight
        row_scores.append({
            "id": pid,
            "exact_match": em,
            "chrF": chrf,
            "score": final,
            "pred_count": len(preds[pid]),
            "gold_count": len(golds[pid]),
        })

    if total_weight == 0:
        return {"score": 0.0, "exact_match": 0.0, "chrF": 0.0, "rows": row_scores}

    overall_em = w_em_sum / total_weight
    overall_chrf = w_chrf_sum / total_weight
    overall_score = math.sqrt(overall_em * overall_chrf)

    return {
        "score": overall_score * 100,
        "exact_match": overall_em,
        "chrF": overall_chrf,
        "rows": row_scores,
    }


def main():
    parser = argparse.ArgumentParser(description="Score IOL-AI submission")
    parser.add_argument("pred", type=Path, help="Path to submission.csv")
    parser.add_argument("gold", type=Path, help="Path to gold answers CSV")
    parser.add_argument("--weights", type=Path, default=None, help="Optional weights CSV")
    parser.add_argument("--verbose", action="store_true", help="Print per-row scores")
    args = parser.parse_args()

    result = score_submission(args.pred, args.gold, args.weights)
    print(f"Score: {result['score']:.2f} (EM={result['exact_match']:.4f}, chrF={result['chrF']:.4f})")

    if args.verbose:
        print("\nPer-row breakdown:")
        for r in result["rows"]:
            print(f"  {r['id']}: EM={r['exact_match']:.4f} chrF={r['chrF']:.4f} score={r['score']:.4f} "
                  f"(pred={r['pred_count']}, gold={r['gold_count']})")


if __name__ == "__main__":
    main()

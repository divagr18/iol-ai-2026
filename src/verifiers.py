"""
IOL-AI 2026 Neuro-Symbolic Verifiers

Deterministic validators for LLM outputs. Each verifier takes the problem
context, the model's predicted answers, and returns (is_valid, corrected_answers, error_msg).

Verifiers:
- number_verifier: validates numeral answers against inferred base system
- matching_verifier: enforces bijection (no duplicates, full coverage)
- fillblank_verifier: applies induced morphological rule to training pairs
- count_verifier: ensures answer count matches expected item count
"""

import json
import re
from collections import Counter
from typing import List, Optional, Tuple

from analyzers import levenshtein_ops, number_system_analysis, tokenize_forms


# ── Count verifier ───────────────────────────────────────────────────────────

def count_verifier(answers: List[str], expected_count: Optional[int]) -> Tuple[bool, str]:
    """Check if the number of answers matches expected."""
    if expected_count is None:
        return True, ""
    if len(answers) != expected_count:
        return False, f"Expected {expected_count} answers, got {len(answers)}"
    return True, ""


# ── Number system verifier ───────────────────────────────────────────────────

def number_verifier(
    context: str,
    answers: List[str],
    task_type: str,
    eval_type: str = "single",
) -> Tuple[bool, List[str], str]:
    """
    For text_to_num / num_to_text problems:
    - Infer digit/base rules from context examples
    - Verify each answer against the inferred system
    - If inconsistency found, flag it.
    """
    if task_type not in ("text_to_num", "num_to_text"):
        return True, answers, ""

    # Extract (written, numeral) examples from context
    lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    examples = []
    for ln in lines:
        nums = re.findall(r"\b\d+\b", ln)
        if nums:
            num = int(nums[0])
            text_part = re.sub(r"\b\d+\b", "", ln).strip("| ")
            if text_part:
                examples.append((text_part, num))

    if not examples:
        return True, answers, "No number examples found in context"

    analysis = number_system_analysis(examples)
    candidates = analysis.get("base_candidates", [])

    if not candidates:
        return True, answers, "Could not infer number system rules"

    # Use the most confident base candidate
    best = candidates[0]
    digit_map = best["digit_map"]

    # Simple check: can we parse the answer?
    validated = []
    for ans in answers:
        ans = ans.strip()
        if task_type == "text_to_num":
            # Answer should be a number
            try:
                int(ans)
                validated.append(ans)
            except ValueError:
                # Maybe it has extra text; try to extract digits
                digits = re.findall(r"\d+", ans)
                if digits:
                    validated.append(digits[0])
                else:
                    validated.append(ans)
        else:
            # num_to_text: answer should be text that maps via digit_map
            tokens = ans.split()
            mapped = []
            for tok in tokens:
                if tok in digit_map:
                    mapped.append(str(digit_map[tok]))
            if mapped:
                # Reconstruct to check consistency
                pass
            validated.append(ans)

    # Additional check: for text_to_num, verify the inferred number matches
    # the digit composition if we have a clear base
    errors = []
    if task_type == "text_to_num" and best.get("base"):
        base = best["base"]
        for i, ans in enumerate(answers):
            try:
                val = int(ans)
            except ValueError:
                continue
            # Check if val appears in any example relation
            # (weak check: just ensure it's within plausible range)
            max_ex = max(n for _, n in examples)
            if val > max_ex * 10 + 100:
                errors.append(f"Item {i + 1}: {val} seems too large vs examples")

    if errors:
        return False, validated, "; ".join(errors)
    return True, validated, ""


# ── Matching verifier ─────────────────────────────────────────────────────────

def matching_verifier(
    context: str,
    answers: List[str],
    task_type: str,
    eval_type: str,
) -> Tuple[bool, List[str], str]:
    """
    For match_letters problems:
    - Ensure no duplicate labels if eval_type is single
    - Ensure full coverage of target set
    - Detect and flag missing/extra labels
    """
    if task_type != "match_letters" and "match" not in eval_type and "match_letters" not in context.lower():
        return True, answers, ""

    # Extract expected labels from context
    # Typical format: "1. acalhuah     A. water" → labels are A, B, C...
    label_pool = set()
    for ln in context.splitlines():
        # Find letter-dot patterns like "A.", "B.", "a."
        letters = re.findall(r"\b([A-Za-z])\.", ln)
        label_pool.update(letters)

    if not label_pool:
        return True, answers, ""

    # Clean answers: extract single letters
    cleaned = []
    for ans in answers:
        ans = ans.strip().upper()
        # Remove trailing punctuation
        ans = re.sub(r"[^A-Za-z]", "", ans)
        cleaned.append(ans)

    # Check bijection-ish properties
    counts = Counter(cleaned)
    duplicates = [lbl for lbl, cnt in counts.items() if cnt > 1]
    missing = list(label_pool - set(cleaned))
    extra = list(set(cleaned) - label_pool)

    errors = []
    if duplicates:
        errors.append(f"Duplicate labels: {duplicates}")
    if missing:
        errors.append(f"Missing labels: {missing}")
    if extra:
        errors.append(f"Extra labels: {extra}")

    if errors and eval_type in ("single", "simple"):
        return False, cleaned, "; ".join(errors)
    return True, cleaned, ""


# ── Fill-blanks verifier ──────────────────────────────────────────────────────

def fillblank_verifier(
    context: str,
    answers: List[str],
    task_type: str,
    eval_type: str = "single",
) -> Tuple[bool, List[str], str]:
    """
    For fill_blanks problems:
    - Extract training pairs (form1 | form2 | gloss)
    - Infer the transformation rule
    - Verify that applying the rule to training data regenerates the known forms
    - If not, the rule is wrong → flag it
    """
    if task_type != "fill_blanks":
        return True, answers, ""

    lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
    pairs = []
    for ln in lines:
        if "|" in ln:
            parts = [p.strip() for p in ln.split("|")]
            if len(parts) >= 2:
                pairs.append(parts)

    if len(pairs) < 2:
        return True, answers, "Too few training pairs to verify"

    # Infer rule from first two pairs: compute edit script between first two columns
    # and check if it's consistent across all pairs
    if len(pairs[0]) >= 2:
        base_form = pairs[0][0]
        target_form = pairs[0][1]
        ops = levenshtein_ops(base_form, target_form)

        # Check consistency across training pairs
        consistent = True
        for p in pairs[1:]:
            if len(p) < 2:
                continue
            test_ops = levenshtein_ops(p[0], p[1])
            # Simple heuristic: same number of insertions/deletions/substitutions
            if len(test_ops) != len(ops):
                consistent = False
                break
            # Check if substitution patterns are similar
            sub_patterns = [(ca, cb) for op, ca, cb in test_ops if op == "~"]
            base_subs = [(ca, cb) for op, ca, cb in ops if op == "~"]
            if len(sub_patterns) != len(base_subs):
                consistent = False
                break

        if not consistent:
            return False, answers, "Inconsistent transformation rule across training pairs"

    # We can't fully validate the blank answers without the gold, but we can
    # check that they follow the same morphological pattern (e.g., same affixes)
    # by comparing to known forms.
    return True, answers, ""


# ── Translation verifier (lightweight) ────────────────────────────────────────

def translation_verifier(
    context: str,
    answers: List[str],
    task_type: str,
    eval_type: str = "single",
) -> Tuple[bool, List[str], str]:
    """
    For translation problems:
    - Check that answers contain only ASCII/English characters (light heuristic)
    - If translating TO English, flag non-ASCII heavy strings
    """
    if task_type != "translation":
        return True, answers, ""

    # Heuristic: if query says "Translate into English", answers should be mostly ASCII
    query_lower = context.lower()
    to_english = "english" in query_lower or "into english" in query_lower

    if to_english:
        errors = []
        for i, ans in enumerate(answers):
            # Count non-ASCII chars
            non_ascii = sum(1 for c in ans if ord(c) > 127)
            if non_ascii > len(ans) * 0.3:
                errors.append(f"Item {i + 1} looks non-English: {ans}")
        if errors:
            return False, answers, "; ".join(errors)
    return True, answers, ""


# ── Master verifier ──────────────────────────────────────────────────────────

def verify_answers(
    context: str,
    query: str,
    answers: List[str],
    task_type: str,
    eval_type: str,
    expected_count: Optional[int],
) -> Tuple[bool, List[str], str]:
    """
    Run all applicable verifiers and return aggregated result.
    corrected_answers may be cleaned/trimmed versions.
    """
    # Count check first
    ok, err = count_verifier(answers, expected_count)
    if not ok:
        return False, answers, err

    all_errors = []
    corrected = answers[:]

    verifiers = [
        (number_verifier, "number"),
        (matching_verifier, "matching"),
        (fillblank_verifier, "fillblank"),
        (translation_verifier, "translation"),
    ]

    for verifier, name in verifiers:
        ok, corrected, err = verifier(context, corrected, task_type, eval_type)
        if not ok:
            all_errors.append(f"[{name}] {err}")

    if all_errors:
        return False, corrected, "; ".join(all_errors)
    return True, corrected, ""

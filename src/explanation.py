"""
IOL-AI 2026 Jury-Track Explanation Generator

Produces a concise, structured explanation for each problem that is
digestible by human IOL judges in a few minutes. Not a raw reasoning trace.
"""

import re
from typing import List


def generate_explanation(
    context: str,
    query: str,
    task_type: str,
    answers: List[str],
    analysis_text: str,
    verified: bool,
    verifier_msg: str,
) -> str:
    """
    Generate a concise explanation for the jury track.
    Format: bullet points or short paragraphs, max ~150 words.
    """
    lines = []

    # Header
    lines.append(f"Task type: {task_type.replace('_', ' ').title()}.")

    # Data summary: what did we observe?
    data_lines = []
    for ln in context.splitlines()[:5]:
        ln = ln.strip()
        if ln and not ln.startswith("-"):
            data_lines.append(ln)
    if data_lines:
        lines.append(f"Key data: {data_lines[0]}{' ...' if len(data_lines) > 1 else ''}")

    # Rule induction (from analysis_text, take first 2 lines)
    if analysis_text:
        analysis_summary = analysis_text.replace("\n", " ").strip()
        if len(analysis_summary) > 200:
            analysis_summary = analysis_summary[:200] + "..."
        lines.append(f"Induced rule: {analysis_summary}")

    # Answers
    if answers:
        ans_summary = ", ".join(answers[:5])
        if len(answers) > 5:
            ans_summary += f", ... ({len(answers)} total)"
        lines.append(f"Answers: {ans_summary}")

    # Verification note
    if not verified and verifier_msg:
        lines.append(f"Note: {verifier_msg[:120]}")
    elif verified:
        lines.append("Verification: consistent with training data.")

    return "\n".join(lines)


def format_explanation_for_csv(text: str) -> str:
    """Sanitize for CSV: replace newlines with spaces, strip quotes."""
    text = text.replace("\n", " ").replace("\r", "")
    text = text.replace('"', "'")
    return text.strip()

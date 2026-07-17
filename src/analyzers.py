"""
IOL-AI 2026 Deterministic Linguistic Analyzers

Pure-CPU modules that mine structure from problem data and return
compact prompt-ready tables. No GPU, no model weights.

Modules:
- frequency_analysis: char / n-gram inventories
- morpheme_mining: repeated-substring affix/stem candidates
- alignment_table: edit-script alignment of paired forms
- paradigm_matrix: paradigm grid from verb/noun conjugation data
- number_system_analysis: base detection and arithmetic rule mining
"""

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple


# ── Utilities ────────────────────────────────────────────────────────────────

def tokenize_forms(text: str, strip_punct: bool = True) -> List[str]:
    """Extract space-separated word-like tokens from text, preserving Unicode."""
    if strip_punct:
        # Keep letters, numbers, IPA diacritics, apostrophe-like chars
        text = re.sub(r"[^\w\s\u02bc\u02be\u02bf\u02c8\u02cc'ʼˈˌ]|_", " ", text)
    tokens = [t.strip() for t in text.split() if t.strip()]
    return tokens


def longest_common_substring(a: str, b: str) -> str:
    """Return longest contiguous common substring."""
    sm = SequenceMatcher(None, a, b)
    match = sm.find_longest_match(0, len(a), 0, len(b))
    if match.size == 0:
        return ""
    return a[match.a: match.a + match.size]


def levenshtein_ops(a: str, b: str) -> List[Tuple[str, str, str]]:
    """
    Compute character-level edit operations between a and b.
    Returns list of (op, char_a, char_b) where op in ('=', '+', '-', '~').
    '=' = match, '+' = insertion in b, '-' = deletion from a, '~' = substitution.
    """
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    # Backtrack
    i, j = m, n
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1]:
            ops.append(("=", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("~", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("+", "", b[j - 1]))
            j -= 1
        else:
            ops.append(("-", a[i - 1], ""))
            i -= 1
    ops.reverse()
    return ops


# ── Frequency analysis ───────────────────────────────────────────────────────

def frequency_analysis(forms: List[str], max_n: int = 4) -> Dict:
    """
    Compute character and n-gram frequency tables.
    Returns dict with keys: chars, bigrams, trigrams, ... up to max_n.
    """
    result = {}
    all_text = " ".join(forms)
    result["chars"] = Counter(all_text.replace(" ", "")).most_common()
    for n in range(2, max_n + 1):
        ngrams = Counter()
        for form in forms:
            for i in range(len(form) - n + 1):
                ngrams[form[i:i + n]] += 1
        result[f"{n}grams"] = ngrams.most_common(20)
    return result


def format_frequency_table(freqs: Dict, top_k: int = 10) -> str:
    """Convert frequency dict to a compact prompt table."""
    lines = ["Character & n-gram inventory:"]
    chars = freqs.get("chars", [])[:top_k]
    lines.append("  Characters: " + ", ".join(f"{ch}:{cnt}" for ch, cnt in chars))
    for key in sorted(freqs.keys()):
        if key == "chars":
            continue
        items = freqs[key][:top_k]
        if items:
            lines.append(f"  {key}: " + ", ".join(f"'{g}':{c}" for g, c in items))
    return "\n".join(lines)


# ── Morpheme mining ──────────────────────────────────────────────────────────

def find_repeated_substrings(forms: List[str], min_len: int = 2, min_freq: int = 2) -> List[Tuple[str, int, str]]:
    """
    Find substrings that appear in multiple forms.
    Returns list of (substring, frequency, position_hint) where position_hint is
    'prefix' if it starts many forms, 'suffix' if it ends many, 'stem' otherwise.
    """
    substrings = Counter()
    for form in forms:
        seen = set()
        for i in range(len(form)):
            for j in range(i + min_len, min(len(form), i + 12) + 1):
                substr = form[i:j]
                if substr not in seen:
                    substrings[substr] += 1
                    seen.add(substr)

    candidates = []
    for substr, freq in substrings.most_common(100):
        if freq < min_freq:
            continue
        # Determine position bias
        prefix_count = sum(1 for f in forms if f.startswith(substr))
        suffix_count = sum(1 for f in forms if f.endswith(substr))
        total = len([f for f in forms if substr in f])
        if prefix_count >= total * 0.7:
            pos = "prefix"
        elif suffix_count >= total * 0.7:
            pos = "suffix"
        else:
            pos = "stem"
        candidates.append((substr, freq, pos))
    return candidates


def format_morpheme_table(candidates: List[Tuple[str, int, str]], top_k: int = 15) -> str:
    """Format morpheme candidates for prompt injection."""
    lines = ["Candidate morphemes (from repeated substrings):"]
    lines.append("  Form | Freq | Position")
    for substr, freq, pos in candidates[:top_k]:
        lines.append(f"  '{substr}' | {freq} | {pos}")
    return "\n".join(lines)


# ── Alignment table ────────────────────────────────────────────────────────────

def build_alignment_table(pairs: List[Tuple[str, str]]) -> str:
    """
    Build an edit-script alignment table from (source, target) pairs.
    Useful for fill-in-the-blanks and translation alignment.
    """
    lines = ["Alignment of paired forms (edit operations):"]
    lines.append("  Source -> Target | Operations")
    for src, tgt in pairs[:20]:
        ops = levenshtein_ops(src, tgt)
        # Compress ops into a compact string
        compressed = []
        for op, ca, cb in ops:
            if op == "=":
                compressed.append(ca)
            elif op == "+":
                compressed.append(f"+{cb}")
            elif op == "-":
                compressed.append(f"-{ca}")
            elif op == "~":
                compressed.append(f"{ca}>{cb}")
        ops_str = "".join(compressed)
        lines.append(f"  {src} -> {tgt} | {ops_str}")
    return "\n".join(lines)


# ── Paradigm matrix ────────────────────────────────────────────────────────────

def build_paradigm_matrix(forms: List[str], glosses: List[str]) -> str:
    """
    Try to cluster forms by longest common substring and show glosses.
    Returns a markdown-style table string.
    """
    if len(forms) != len(glosses):
        return ""
    # Group by shared root (longest common substring heuristic)
    groups = defaultdict(list)
    for i, f in enumerate(forms):
        key = f  # simplistic: use full form; better heuristics possible offline
        groups[key[:4] if len(key) >= 4 else key].append((f, glosses[i]))

    lines = ["Paradigm matrix (forms vs glosses):"]
    lines.append("  Form | Gloss")
    for root, items in groups.items():
        for form, gloss in items:
            lines.append(f"  {form} | {gloss}")
        if len(items) > 1:
            lines.append("  ---")
    return "\n".join(lines)


# ── Number system analysis ───────────────────────────────────────────────────

def number_system_analysis(examples: List[Tuple[str, int]]) -> Dict:
    """
    Given (written_form, numeral) examples, try to infer number system rules.
    Returns dict with base candidates, digit mappings, and operation hints.
    """
    if not examples:
        return {}
    # Extract tokens from written forms
    tokens = set()
    for form, num in examples:
        tokens.update(form.split())

    # Map tokens to their appearances and associated numbers
    token_to_nums = defaultdict(list)
    for form, num in examples:
        for tok in form.split():
            token_to_nums[tok].append(num)

    # Infer base by checking if tokens correspond to digit-like values
    base_candidates = []
    for base in range(2, 21):
        digit_map = {}
        consistent = True
        for tok, nums in token_to_nums.items():
            # Check if token appears with numbers that share a digit in this base
            digits = {n % base for n in nums}
            if len(digits) == 1:
                digit_map[tok] = list(digits)[0]
        if len(digit_map) >= 2:
            base_candidates.append({"base": base, "digit_map": digit_map})

    return {
        "tokens": sorted(tokens),
        "base_candidates": base_candidates[:5],
        "token_to_nums": dict(token_to_nums),
    }


def format_number_analysis(analysis: Dict) -> str:
    """Format number system analysis for prompt."""
    if not analysis:
        return ""
    lines = ["Number system analysis:"]
    lines.append("  Tokens: " + ", ".join(analysis["tokens"]))
    for cand in analysis["base_candidates"]:
        lines.append(f"  Base {cand['base']} digit map: " + ", ".join(
            f"{tok}={d}" for tok, d in cand["digit_map"].items()
        ))
    return "\n".join(lines)


# ── High-level wrapper: analyze_problem ───────────────────────────────────────

def analyze_problem(context: str, query: str, task_type: str) -> str:
    """
    Analyze a single problem's context and return a compact prompt appendix.
    This is the main entry point called by script.py.
    """
    # Extract all word-like tokens from the context (both languages)
    tokens = tokenize_forms(context)

    # Heuristic: separate source-language tokens from English
    # English tokens are ASCII-only (rough heuristic)
    source_tokens = [t for t in tokens if not t.isascii() or len(t) > 1 and not t[0].isascii()]
    english_tokens = [t for t in tokens if t.isascii() and t.isalpha()]

    sections = []

    # Frequency analysis on source language tokens
    if source_tokens:
        freqs = frequency_analysis(source_tokens, max_n=3)
        sections.append(format_frequency_table(freqs, top_k=8))

    # Morpheme mining
    if source_tokens and len(source_tokens) >= 4:
        morphs = find_repeated_substrings(source_tokens, min_len=2, min_freq=2)
        if morphs:
            sections.append(format_morpheme_table(morphs, top_k=10))

    # For fill_blanks / translation: build alignment from paired lines
    if task_type in ("fill_blanks", "translation", "match_letters"):
        pairs = []
        lines = [ln.strip() for ln in context.splitlines() if ln.strip()]
        for ln in lines:
            if "|" in ln:
                parts = [p.strip() for p in ln.split("|")]
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        if pairs:
            sections.append(build_alignment_table(pairs))

    # Number system
    if task_type in ("text_to_num", "num_to_text"):
        # Extract (written, numeral) pairs if numerals appear
        num_examples = []
        for ln in lines:
            nums = re.findall(r"\b\d+\b", ln)
            if nums:
                num = int(nums[0])
                text_part = re.sub(r"\b\d+\b", "", ln).strip("| ")
                if text_part:
                    num_examples.append((text_part, num))
        if num_examples:
            analysis = number_system_analysis(num_examples)
            sections.append(format_number_analysis(analysis))

    return "\n\n".join(sections)

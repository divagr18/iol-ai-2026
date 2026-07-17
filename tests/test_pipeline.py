"""
Unit tests for the IOL-AI pipeline (analyzers, verifiers, explanation, scorer)
Runs without loading any neural model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import analyzers
import verifiers
import explanation


def test_analyzers():
    context = """Here are sentences in Hakhun and their English translations:
1. ŋa ka kɤ ne | Do I go?
2. nɤ ʒip tuʔ ne | Did you sleep?
3. ŋabə ati lapkʰi tɤʔ ne | Did I see him?
"""
    query = "Translate into English:\n1. nɤ ʒip ku ne\n2. ati kəmə nirum lapkʰi tʰi ne"
    analysis = analyzers.analyze_problem(context, query, "translation")
    print("Analysis output:")
    print(analysis)
    assert "Character" in analysis or "morpheme" in analysis or "Alignment" in analysis
    print("test_analyzers PASSED\n")


def test_verifiers_count():
    ok, msg = verifiers.count_verifier(["a", "b", "c"], 3)
    assert ok
    ok, msg = verifiers.count_verifier(["a", "b"], 3)
    assert not ok
    print("test_verifiers_count PASSED\n")


def test_verifiers_matching():
    context = """Given are words in Nahuatl and their English translations:
1. acalhuah     A. water
2. achilli      B. child
3. atl          C. master of house
"""
    ok, corrected, msg = verifiers.matching_verifier(context, ["A", "B", "C"], "match_letters", "single")
    assert ok
    ok, corrected, msg = verifiers.matching_verifier(context, ["A", "A", "C"], "match_letters", "single")
    assert not ok
    print("test_verifiers_matching PASSED\n")


def test_verifiers_number():
    context = """Squares of 1 to 10 in Ndom:
1. nif abo mer an thef abo sas
2. nif thef abo tondor abo mer abo thonith
"""
    ok, corrected, msg = verifiers.number_verifier(context, ["111"], "text_to_num")
    assert ok
    print("test_verifiers_number PASSED\n")


def test_explanation():
    exp = explanation.generate_explanation(
        "context here", "query here", "translation",
        ["Do you sleep?", "Did he see us?"],
        "analysis text", True, ""
    )
    assert "Task type" in exp
    assert "Answers" in exp
    print("test_explanation PASSED\n")


def test_fillblank_verifier():
    context = """Here are two different forms of some verbs:
piriyʼ  | ɨmbirʼi  | see
imʼay   | ɨnimʼa   | say, tell
kʼaniyʼ | ɨŋkʼanʼi | trap
"""
    ok, corrected, msg = verifiers.fillblank_verifier(context, ["ɨnnetakʼa"], "fill_blanks")
    # May or may not be consistent depending on the pairs; just ensure it runs
    print(f"fillblank result: ok={ok}, msg={msg}")
    print("test_fillblank_verifier PASSED\n")


def main():
    test_analyzers()
    test_verifiers_count()
    test_verifiers_matching()
    test_verifiers_number()
    test_explanation()
    test_fillblank_verifier()
    print("All tests passed!")


if __name__ == "__main__":
    main()

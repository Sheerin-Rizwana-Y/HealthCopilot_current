"""Synthetic CareCopilot RAG evaluation (not a clinical validation)."""

import argparse
import re


TEST_CASES = [
    {
        "name": "Original report symptoms",
        "question": (
            "In demo_report.pdf, which symptoms are "
            "explicitly listed under Symptoms?"
        ),
        "expected": ["dizziness", "tiredness"],
        "forbidden": ["cough", "headache"],
        "source": "demo_report.pdf",
    },
    {
        "name": "Follow-up report symptoms",
        "question": (
            "In demo_followup_2026-10-10.pdf, "
            "which symptoms are explicitly listed "
            "under Symptoms?"
        ),
        "expected": ["cough", "headache"],
        "forbidden": ["dizziness", "tiredness"],
        "source": "demo_followup_2026-10-10.pdf",
    },
    {
        "name": "Missing blood pressure",
        "question": (
            "What blood pressure value is "
            "recorded in the medical reports?"
        ),
        "uncertainty": True,
    },
    {
        "name": "Unsupported symptom progression",
        "question": (
            "Based only on the medical reports, "
            "did cough become more severe by "
            "October 18, 2026?"
        ),
        "uncertainty": True,
    },
]

UNCERTAINTY_TERMS = [
    "not available",
    "not provided",
    "not recorded",
    "not mentioned",
    "not specified",
    "not documented",
    "no information",
    "no evidence",
    "insufficient",
    "cannot",
    "can't",
    "unable",
    "unknown",
    "does not contain",
    "doesn't contain",
    "do not contain",
    "don't contain",
    "does not provide",
    "doesn't provide",
    "do not provide",
    "don't provide",
    "not enough",
]

# This checks a few known unsupported claim patterns in the synthetic fixtures.
# It cannot verify every medical fact or every possible paraphrase.
VISIT_PATTERNS = [
    re.compile(
        r"\b(?:october\s+10(?:,\s*2026)?|2026-10-10)"
        r"\s+(?:visit|appointment)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:visit|appointment)\s+(?:on\s+)?"
        r"(?:october\s+10(?:,\s*2026)?|2026-10-10)\b",
        re.IGNORECASE,
    ),
]

PROGRESSION_PATTERN = re.compile(
    r"\bcough\s+(?:became|got|has become|is)\s+"
    r"(?:more\s+)?(?:severe|worse)\b|\bcough\s+worsened\b",
    re.IGNORECASE,
)

NEGATING_PREFIX = re.compile(
    r"\b(?:"
    r"no\s+evidence(?:\s+(?:that|to\s+suggest))?|"
    r"not\s+enough\s+evidence|"
    r"insufficient\s+evidence|"
    r"cannot\s+(?:say|determine|conclude|establish)|"
    r"can't\s+(?:say|determine|conclude|establish)|"
    r"(?:does|do)\s+not\s+(?:show|prove|establish|indicate)|"
    r"(?:doesn't|don't)\s+(?:show|prove|establish|indicate)|"
    r"not\s+(?:known|established|documented|shown)|"
    r"whether|unclear\s+(?:whether|if)"
    r")\b",
    re.IGNORECASE,
)


def has_unsupported_progression_claim(answer):
    """Flag direct affirmative progression claims, avoiding common negations."""
    for match in PROGRESSION_PATTERN.finditer(answer):
        prefix = answer[:match.start()]
        boundary = max(
            prefix.rfind("."), prefix.rfind("!"),
            prefix.rfind("?"), prefix.rfind(";"),
            prefix.rfind("\n"),
        )
        clause_prefix = prefix[boundary + 1:]
        if not NEGATING_PREFIX.search(clause_prefix):
            return True
    return False


def evaluate_case(case, result):
    """Return a list of issues for a single synthetic question/answer."""
    answer_text = result.get("answer") or ""
    answer = answer_text.lower()
    issues = []

    for word in case.get("expected", []):
        if not re.search(rf"\b{re.escape(word)}\b", answer):
            issues.append(f"Missing expected term: {word}")

    for word in case.get("forbidden", []):
        if re.search(rf"\b{re.escape(word)}\b", answer):
            issues.append(f"Unexpected term: {word}")

    source_map = {
        str(source["number"]): source["file"]
        for source in result.get("sources", [])
    }
    citations = re.findall(r"\[(\d+)\]", answer_text)

    for citation in citations:
        if citation not in source_map:
            issues.append(f"Invalid citation: [{citation}]")

    expected_source = case.get("source")
    if expected_source:
        cited_files = {
            source_map[citation]
            for citation in citations
            if citation in source_map
        }
        if expected_source not in cited_files:
            issues.append("Expected source was not cited")

    if case.get("uncertainty") and not any(
        term in answer for term in UNCERTAINTY_TERMS
    ):
        issues.append("Uncertainty not clearly expressed")

    if case["name"] == "Missing blood pressure":
        if re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\b", answer):
            issues.append("Possible invented blood pressure value")

    if case["name"] == "Unsupported symptom progression":
        if any(pattern.search(answer) for pattern in VISIT_PATTERNS):
            issues.append("Unsupported claim: report called a visit")

        if re.search(
            r"\bno\s+(?:later|newer|subsequent)\s+reports?\b",
            answer,
        ):
            issues.append("Unsupported claim: no later report exists")

        if has_unsupported_progression_claim(answer):
            issues.append("Unsupported claim: cough worsened")

    return issues


def run_self_test():
    """Verify evaluator logic locally; this makes NO Groq API calls."""
    def check(name, answer):
        return evaluate_case(
            {"name": name, "uncertainty": True},
            {"answer": answer, "sources": []},
        )

    blood_pressure = check(
        "Missing blood pressure",
        "The reports do not contain blood pressure information.",
    )
    assert not blood_pressure, blood_pressure

    uncertain = check(
        "Unsupported symptom progression",
        "The reports do not provide enough evidence to determine "
        "whether cough worsened.",
    )
    assert not uncertain, uncertain

    negated = check(
        "Unsupported symptom progression",
        "There is no evidence that cough became more severe.",
    )
    assert not negated, negated

    unsupported = check(
        "Unsupported symptom progression",
        "Cough became more severe. No later report is available.",
    )
    assert any("Unsupported claim" in issue for issue in unsupported), unsupported

    invented_bp = check(
        "Missing blood pressure",
        "Blood pressure is 120/80 mmHg, but the report does not contain it.",
    )
    assert any("invented blood pressure" in issue for issue in invented_bp), invented_bp

    wrong_citation = evaluate_case(
        {"name": "Original report symptoms", "source": "demo_report.pdf"},
        {
            "answer": "Dizziness and tiredness [3].",
            "sources": [{"number": 1, "file": "demo_report.pdf"}],
        },
    )
    assert any("Invalid citation" in issue for issue in wrong_citation), wrong_citation

    visit = check(
        "Unsupported symptom progression",
        "The October 10, 2026 visit documents cough, "
        "but the reports do not provide enough evidence about progression.",
    )
    assert any("report called a visit" in issue for issue in visit), visit

    print("All 7 local evaluator checks passed (no Groq calls).")


def run_evaluation():
    # Avoid importing heavyweight retrieval libraries during --self-test.
    from generate_answer import generate_answer

    print("\n--- CARECOPILOT RAG + LLM EVALUATION ---")
    passed = 0
    durations = []

    for i, case in enumerate(TEST_CASES, start=1):
        print("\n" + "=" * 55)
        print(f"TEST {i}: {case['name']}")
        print("Question:", case["question"])

        try:
            result = generate_answer(
                patient_id="demo_patient",
                question=case["question"],
                top_k=2,
            )
            print("\nLLM Answer:\n", result["answer"], sep="")

            print("\nSources:")
            for source in result["sources"]:
                print(
                    f"[{source['number']}] "
                    f"{source['file']}, page {source['page']}"
                )

            issues = evaluate_case(case, result)
            if issues:
                print("\nResult: FAIL / REVIEW")
                for issue in issues:
                    print("-", issue)
            else:
                print("\nResult: PASS (automated checks)")
                passed += 1

            if result.get("latency") is not None:
                durations.append(float(result["latency"]))
                print("Latency:", result["latency"], "seconds")

        except Exception as error:
            print("\nResult: ERROR")
            print(type(error).__name__, str(error))

    total = len(TEST_CASES)
    print("\n" + "=" * 55)
    print("--- EVALUATION SUMMARY ---")
    print("Total tests:", total)
    print("Passed automated checks:", passed)
    print("Failed or requiring review:", total - passed)
    if durations:
        print(
            "Average latency:",
            round(sum(durations) / len(durations), 2),
            "seconds",
        )
    print(
        "\nThese are basic synthetic-fixture checks, not proof of "
        "complete grounding, citation support, or clinical correctness."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Test only the local evaluator rules; no Groq API calls.",
    )
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
    else:
        run_evaluation()

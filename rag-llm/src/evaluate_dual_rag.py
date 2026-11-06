"""CareCopilot Phase 3 - Dual-RAG evaluation (synthetic demo only).

Run from the HealthCopilot project root:
    python rag-llm/src/evaluate_dual_rag.py --self-test
    python rag-llm/src/evaluate_dual_rag.py --verify
    python rag-llm/src/evaluate_dual_rag.py --live

--self-test checks evaluator behavior locally (no FAISS/Groq).
--verify checks evidence retrieval (no Groq calls).
--live checks two Groq explanations and reveals an unverified draft when
       dual_rag.run() raises because a required citation is missing.

Automated checks cannot establish clinical accuracy or full factual grounding.
"""

import argparse
import contextlib
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "rag-llm" / "evaluation"

CASES = [
    {
        "id": "cough",
        "question": (
            "Which of this demo patient's medical reports mention cough? "
            "What does the retrieved MedlinePlus reference explain about cough? "
            "Keep patient facts and general information separate, citing both."
        ),
        "patient_file": "demo_followup_2026-10-10.pdf",
        "reference_topic": "cough",
        "answer_keyword": "cough",
    },
    {
        "id": "dizziness",
        "question": (
            "Which of this demo patient's reports mention dizziness? "
            "What does the retrieved MedlinePlus reference explain about "
            "dizziness or vertigo? Cite patient and general sources separately."
        ),
        "patient_file": "demo_third_2026-10-15.pdf",
        "reference_topic": "dizziness",
        "answer_keyword": "dizziness",
    },
]

REQUIRED_PATIENT_FILES = {
    "demo_report.pdf",
    "demo_followup_2026-10-10.pdf",
    "demo_third_2026-10-15.pdf",
}

ANSWER_MARKER = "--- CARECOPILOT DUAL-RAG ANSWER ---"
DRAFT_MARKER = "--- GROQ DRAFT (UNVERIFIED) ---"
SUCCESS_MARKER = "CITATION FORMAT CHECK: PASS"


def validate_evidence(facts, case):
    """Check the provenance of retrieved synthetic patient and medical data."""
    issues = []
    patients = facts.get("patient_reports", [])
    medical = facts.get("medical_references", [])

    if facts.get("patient_id") != "demo_patient":
        issues.append("Incorrect patient ID")

    if {p.get("file") for p in patients} != REQUIRED_PATIENT_FILES:
        issues.append("Expected three distinct demo PDF reports")

    if len(patients) != 3 or len({p.get("ref") for p in patients}) != 3:
        issues.append("Patient reference IDs are missing or repeated")

    for item in patients:
        if not re.fullmatch(r"P[1-9]\d*", str(item.get("ref", ""))):
            issues.append("Invalid patient citation label")
        if not isinstance(item.get("page"), int) or item["page"] < 1:
            issues.append("Invalid patient report page")
        if "SYNTHETIC MEDICAL REPORT - TEST ONLY" not in (item.get("text") or ""):
            issues.append("Non-synthetic patient report text detected")

    if not any(p.get("file") == case["patient_file"] for p in patients):
        issues.append("Question-specific patient report not retrieved")

    if not medical or len({m.get("ref") for m in medical}) != len(medical):
        issues.append("Medical reference IDs missing or repeated")

    for item in medical:
        if not re.fullmatch(r"M[1-9]\d*", str(item.get("ref", ""))):
            issues.append("Invalid medical citation label")
        url = urlparse(item.get("url") or "")
        if url.scheme != "https" or url.hostname != "medlineplus.gov":
            issues.append("Reference URL is not an approved MedlinePlus URL")
        if not item.get("title") or not item.get("text"):
            issues.append("Medical reference content missing")
        if "patient_id" in item or "demo_patient" in (item.get("text") or "").casefold():
            issues.append("Patient information appears in medical reference")

    if not any(
        case["reference_topic"] in (m.get("title") or "").casefold()
        for m in medical
    ):
        issues.append("Question-specific MedlinePlus topic absent from retrieved top-k")

    return issues


def extract_answer(console_text):
    """Extract the validated answer printed by dual_rag.run()."""
    if ANSWER_MARKER not in console_text or SUCCESS_MARKER not in console_text:
        raise ValueError("No completed dual-RAG answer / validation marker")
    return console_text.split(ANSWER_MARKER, 1)[1].split(SUCCESS_MARKER, 1)[0].strip()


def extract_unverified_draft(console_text):
    """Extract draft printed BEFORE dual_rag.run() raises a citation error.

    Requires dual_rag.py to print DRAFT_MARKER and the answer immediately
    after receiving the Groq response, before citation validation.
    """
    if DRAFT_MARKER not in console_text:
        return None
    draft = console_text.split(DRAFT_MARKER, 1)[1]
    for marker in (ANSWER_MARKER, SUCCESS_MARKER, "\n--- SOURCE REFERENCE KEY ---"):
        if marker in draft:
            draft = draft.split(marker, 1)[0]
    return draft.strip() or None


def validate_answer(answer, facts, case):
    """Check citation IDs and source routing; NOT the medical correctness."""
    issues = []
    known = {
        item["ref"]
        for group in ("patient_reports", "medical_references", "unverified_history")
        for item in facts.get(group, [])
    }
    cited = set(re.findall(r"\[([A-Za-z]\d+)\]", answer))

    unknown = cited - known
    if unknown:
        issues.append("Unknown citations: " + ", ".join(sorted(unknown)))

    relevant_patients = {
        p["ref"] for p in facts["patient_reports"]
        if p["file"] == case["patient_file"]
    }
    relevant_medical = {
        m["ref"] for m in facts["medical_references"]
        if case["reference_topic"] in (m.get("title") or "").casefold()
    }

    if not (cited & relevant_patients):
        issues.append("Did not cite the expected patient report")
    if not (cited & relevant_medical):
        issues.append("Did not cite the relevant MedlinePlus topic")
    if not re.search(r"\b" + re.escape(case["answer_keyword"]) + r"\b", answer, re.I):
        issues.append("Target symptom not mentioned")
    if re.search(r"\b(?:120\s*/\s*80|130\s*/\s*90)\s*(?:mmhg)?\b", answer, re.I):
        issues.append("Unexpected blood pressure figure (review)")

    return issues


def example_facts():
    return {
        "patient_id": "demo_patient",
        "patient_reports": [
            {
                "ref": "P1", "file": "demo_report.pdf", "page": 1,
                "text": "SYNTHETIC MEDICAL REPORT - TEST ONLY",
            },
            {
                "ref": "P2", "file": "demo_followup_2026-10-10.pdf", "page": 1,
                "text": "SYNTHETIC MEDICAL REPORT - TEST ONLY Symptoms: Cough",
            },
            {
                "ref": "P3", "file": "demo_third_2026-10-15.pdf", "page": 1,
                "text": "SYNTHETIC MEDICAL REPORT - TEST ONLY",
            },
        ],
        "medical_references": [
            {
                "ref": "M1", "title": "Cough",
                "url": "https://medlineplus.gov/cough.html",
                "text": "General information about cough.",
            },
        ],
        "unverified_history": [],
    }


def self_test():
    case = CASES[0]
    facts = example_facts()
    assert not validate_evidence(facts, case), validate_evidence(facts, case)

    good = "Patient report mentions cough [P2]. General cough information [M1]."
    assert not validate_answer(good, facts, case), validate_answer(good, facts, case)
    assert extract_answer(
        ANSWER_MARKER + "\n" + good + "\n" + SUCCESS_MARKER
    ) == good
    assert extract_unverified_draft(
        DRAFT_MARKER + "\n" + good
    ) == good

    bad = "Cough is noted in Reference 2. The general source is Reference 1."
    issues = validate_answer(bad, facts, case)
    assert any("patient report" in issue for issue in issues)
    assert any("MedlinePlus" in issue for issue in issues)

    invalid = validate_answer("Cough [P99] [M1].", facts, case)
    assert any("Unknown citations" in issue for issue in invalid)

    altered = json.loads(json.dumps(facts))
    altered["medical_references"][0]["url"] = "https://example.com/health"
    assert any("MedlinePlus URL" in issue for issue in validate_evidence(altered, case))

    assert extract_unverified_draft("no draft printed") is None

    print("SELF-TEST: PASS (evidence isolation, citation checks, draft capture)")
    print("Groq requests: 0")


def run_evaluation(live):
    # Import here so --self-test needs no embedding or Groq dependencies.
    from dual_rag import prepare, run as dual_rag_run

    outcomes = []
    requests_attempted = 0

    for case in CASES:
        print("\n" + "=" * 55)
        print("TEST:", case["id"])
        print("Question:", case["question"])
        started = time.perf_counter()
        outcome = {
            "case": case["id"],
            "question": case["question"],
            "status": "ERROR",
            "issues": [],
            "latency_seconds": None,
            "answer": None,
        }
        try:
            # Preflight evidence verification, without LLM.
            facts = prepare(case["question"])
            evidence_issues = validate_evidence(facts, case)

            if evidence_issues:
                outcome["status"] = "REVIEW"
                outcome["issues"] = evidence_issues
            elif not live:
                outcome["status"] = "PASS"
            else:
                # Keep using the existing dual_rag.run() Groq prompt and
                # validation; capture output so failed answers are visible.
                buffer = io.StringIO()
                requests_attempted += 1  # upper bound; run() may fail before API call
                try:
                    with contextlib.redirect_stdout(buffer):
                        dual_rag_run(case["question"], live=True)
                except Exception as exc:
                    captured = buffer.getvalue()
                    print("\n--- FAILED DUAL-RAG OUTPUT ---")
                    print(captured[-7000:] or "(No output captured)")
                    draft = extract_unverified_draft(captured)
                    if draft:
                        outcome["answer"] = draft
                        outcome["issues"] = validate_answer(draft, facts, case)
                        outcome["status"] = "REVIEW"
                    else:
                        outcome["status"] = "ERROR"
                        outcome["issues"].append(
                            "No model draft captured; check that dual_rag.py "
                            "prints the GROQ DRAFT marker before validation"
                        )
                    outcome["issues"].insert(
                        0, f"Pipeline error: {type(exc).__name__}: {exc}"
                    )
                else:
                    captured = buffer.getvalue()
                    answer = extract_answer(captured)
                    outcome["answer"] = answer
                    answer_issues = validate_answer(answer, facts, case)
                    outcome["issues"] = answer_issues
                    outcome["status"] = "PASS" if not answer_issues else "REVIEW"
                    print("\nLLM answer:\n" + answer)

        except Exception as exc:
            outcome["status"] = "ERROR"
            outcome["issues"].append(f"{type(exc).__name__}: {exc}")

        outcome["latency_seconds"] = round(time.perf_counter() - started, 2)
        print("Result:", outcome["status"])
        for issue in outcome["issues"]:
            print(" -", issue)
        print("Wall time:", outcome["latency_seconds"], "seconds")
        outcomes.append(outcome)

    count_pass = sum(item["status"] == "PASS" for item in outcomes)
    average = round(
        sum(item["latency_seconds"] for item in outcomes) / len(outcomes), 2
    )
    print("\n--- DUAL-RAG EVALUATION SUMMARY ---")
    print("Mode:", "GROQ LIVE" if live else "LOCAL VERIFICATION")
    print("Total tests:", len(outcomes))
    print("Passed automated checks:", count_pass)
    print("Review / errors:", len(outcomes) - count_pass)
    print("Average wall time:", average, "seconds")
    print("Groq attempts (upper bound):", requests_attempted)
    print("Note: Automated PASS does not establish clinical accuracy or full grounding.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "mode": "live" if live else "verify",
        "created_at_utc": stamp,
        "patient_id": "demo_patient",
        "results": outcomes,
        "passed": count_pass,
        "total": len(outcomes),
        "average_wall_seconds": average,
        "limitation": "Synthetic fixtures, source routing and citation syntax only",
    }
    path = OUT_DIR / f"dual_rag_{'live' if live else 'verify'}_{stamp}.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Results saved:", path)
    return 0 if count_pass == len(outcomes) else 1


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--verify", action="store_true")
    group.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    return run_evaluation(live=args.live)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("EVALUATION ERROR:", type(exc).__name__, str(exc))
        sys.exit(1)

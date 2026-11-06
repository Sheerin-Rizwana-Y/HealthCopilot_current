
import json
import sys
from pathlib import Path

import faiss

from retrieve import retrieve
from report_timeline import load_reports
from compare_symptoms import normalize


ROOT = Path(__file__).resolve().parents[2]

PATIENT_ID = "demo_patient"

INDEX_DIR = (
    ROOT / "rag-llm" / "embeddings"
    / "patients" / PATIENT_ID
)

EXPECTED_REPORTS = {
    "demo_report.pdf": {
        "date": None,
        "symptoms": {"dizziness", "tiredness"}
    },
    "demo_followup_2026-10-10.pdf": {
        "date": "2026-10-10",
        "symptoms": {"cough", "headache"}
    },
    "demo_third_2026-10-15.pdf": {
        "date": "2026-10-15",
        "symptoms": {"dizziness", "nausea"}
    }
}

TEST_QUERIES = [
    {
        "name": "Original report retrieval",
        "query": "Symptoms: Dizziness and tiredness",
        "expected_file": "demo_report.pdf"
    },
    {
        "name": "Follow-up report retrieval",
        "query": "Symptoms: Cough and headache",
        "expected_file": "demo_followup_2026-10-10.pdf"
    },
    {
        "name": "Third report retrieval",
        "query": "Symptoms: Dizziness and nausea",
        "expected_file": "demo_third_2026-10-15.pdf"
    }
]


def verify_index():
    print("\n[1] VERIFYING FAISS INDEX")

    metadata_path = INDEX_DIR / "metadata.json"
    index_path = INDEX_DIR / "index.faiss"

    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)

    if not index_path.exists():
        raise FileNotFoundError(index_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if metadata["patient_id"] != PATIENT_ID:
        raise AssertionError("Patient ID mismatch")

    # Only read FAISS indexes built by our own pipeline.
    index = faiss.read_index(str(index_path))
    chunks = metadata["chunks"]

    indexed_files = {
        chunk["source_file"]
        for chunk in chunks
    }

    print("FAISS vectors:", index.ntotal)
    print("Embedding dimensions:", index.d)
    print("Indexed documents:", len(indexed_files))

    for filename in sorted(indexed_files):
        print(" -", filename)

    assert index.ntotal == len(chunks), (
        "FAISS vector/chunk count mismatch"
    )

    assert index.d == 384, (
        "Unexpected embedding dimension"
    )

    assert indexed_files == set(EXPECTED_REPORTS), (
        "Indexed document list does not match "
        "the three expected demo reports"
    )

    assert index.ntotal == 3, (
        "Expected exactly three demo vectors"
    )

    assert all(
        chunk["patient_id"] == PATIENT_ID
        for chunk in chunks
    ), "Cross-patient chunk detected"

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids)), (
        "Duplicate chunk identifiers detected"
    )

    print("INDEX VERIFICATION: PASS")


def verify_report_metadata():
    print("\n[2] VERIFYING REPORT METADATA")

    reports = load_reports(PATIENT_ID)

    assert len(reports) == 3, (
        "Expected three extracted medical reports"
    )

    for report in reports:
        filename = report["source_file"]

        assert filename in EXPECTED_REPORTS, (
            f"Unexpected report: {filename}"
        )

        expected = EXPECTED_REPORTS[filename]

        assert report["report_date"] == expected["date"], (
            f"Incorrect date for {filename}"
        )

        extracted = {
            normalize(item["name"])
            for item in report["symptoms"]
        }

        assert extracted == expected["symptoms"], (
            f"Incorrect extracted symptoms in {filename}"
        )

        print(
            f"PASS: {filename} "
            f"| Date: {report['report_date'] or 'UNKNOWN'}"
        )

    print("REPORT METADATA VERIFICATION: PASS")


def verify_semantic_retrieval():
    print("\n[3] TESTING SEMANTIC RETRIEVAL")

    passed = 0

    for test in TEST_QUERIES:
        print("\nTest:", test["name"])

        results = retrieve(
            patient_id=PATIENT_ID,
            query=test["query"],
            top_k=1
        )

        if not results:
            print("FAIL: No results")
            continue

        top_result = results[0]

        actual_file = top_result["source_file"]
        expected_file = test["expected_file"]

        print("Expected:", expected_file)
        print("Retrieved:", actual_file)
        print(
            "Similarity:",
            round(top_result["score"], 4)
        )

        if actual_file == expected_file:
            print("Result: PASS")
            passed += 1
        else:
            print("Result: FAIL - incorrect top result")

    print("\n--- SEMANTIC RETRIEVAL SUMMARY ---")
    print("Tests:", len(TEST_QUERIES))
    print("Passed:", passed)

    if passed != len(TEST_QUERIES):
        raise AssertionError(
            "One or more top-1 retrieval tests failed"
        )


def main():
    print("\n===================================")
    print(" CARECOPILOT MULTI-REPORT EVALUATION")
    print("===================================")

    verify_index()
    verify_report_metadata()
    verify_semantic_retrieval()

    print("\n===================================")
    print(" FINAL VERIFICATION: PASS")
    print("===================================")
    print("Reports verified: 3")
    print("FAISS vectors verified: 3")
    print("Top-1 retrieval tests passed: 3/3")
    print("Patient:", PATIENT_ID)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print("\nFINAL VERIFICATION: FAILED")
        print(type(error).__name__, str(error))
        sys.exit(1)

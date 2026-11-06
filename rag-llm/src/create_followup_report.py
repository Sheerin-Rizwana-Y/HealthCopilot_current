
from pathlib import Path
import pymupdf as fitz

BASE = Path(__file__).resolve().parents[1]

OUTPUT = (
    BASE / "sample_reports"
    / "demo_followup_2026-10-10.pdf"
)


def create_report():
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT.exists():
        print("Synthetic report already exists:", OUTPUT)
        return

    lines = [
        "SYNTHETIC MEDICAL REPORT - TEST ONLY",
        "Patient: Demo Patient",
        "Report Date: 2026-10-10",
        "Symptoms: Cough and headache.",
        "Heart Rate: 88 bpm",
        "SpO2: 99 percent",
        "Temperature: 36.9 C",
        "Note: This is fictional test data."
    ]

    doc = fitz.open()
    page = doc.new_page()

    for i, line in enumerate(lines):
        page.insert_text(
            (50, 60 + i * 25),
            line,
            fontsize=11
        )

    doc.save(OUTPUT)
    doc.close()

    print("\n--- SYNTHETIC REPORT CREATED ---")
    print("File:", OUTPUT)
    print("Report date: 2026-10-10")
    print("Symptoms: Cough, headache")


if __name__ == "__main__":
    create_report()


from pathlib import Path
import pymupdf

BASE = Path(__file__).resolve().parents[1]

OUTPUT = (
    BASE / "sample_reports"
    / "demo_third_2026-10-15.pdf"
)


def create_report():
    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT.exists():
        print("Report already exists:", OUTPUT)
        return

    lines = [
        "SYNTHETIC MEDICAL REPORT - TEST ONLY",
        "Patient: Demo Patient",
        "Report Date: 2026-10-15",
        "Symptoms: Dizziness and nausea.",
        "Heart Rate: 92 bpm",
        "SpO2: 97 percent",
        "Temperature: 37.1 C",
        "Note: This is fictional test data."
    ]

    doc = pymupdf.open()

    try:
        page = doc.new_page()

        for i, line in enumerate(lines):
            page.insert_text(
                (50, 60 + i * 25),
                line,
                fontsize=11
            )

        doc.save(str(OUTPUT))

    finally:
        doc.close()

    print("\n--- THIRD SYNTHETIC REPORT CREATED ---")
    print("File:", OUTPUT)
    print("Patient: demo_patient")
    print("Report date: 2026-10-15")
    print("Symptoms: Dizziness, nausea")


if __name__ == "__main__":
    create_report()

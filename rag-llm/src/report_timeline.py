
import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path

from symptom_records import load_symptoms
from compare_symptoms import (
    extract_report_symptoms,
    normalize,
)


BASE = (
    Path(__file__).resolve().parents[1]
    / "knowledge_base"
    / "patients"
)

DATE_PATTERN = re.compile(
    r"^[ \t]*Report Date[ \t]*:[ \t]*"
    r"(\d{4}-\d{2}-\d{2})[ \t]*$",
    flags=re.IGNORECASE | re.MULTILINE
)


def extract_report_date(pages):
    dates = set()

    for page in pages:
        for match in DATE_PATTERN.finditer(page["text"]):
            value = match.group(1)

            # Validate the actual calendar date.
            date.fromisoformat(value)
            dates.add(value)

    if len(dates) > 1:
        raise ValueError(
            "Conflicting report dates found in one PDF"
        )

    return next(iter(dates)) if dates else None


def load_reports(patient_id):
    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,64}", patient_id
    ):
        raise ValueError("Invalid patient ID")

    if patient_id != "demo_patient":
        raise ValueError(
            "Use only demo_patient during development"
        )

    folder = BASE / patient_id / "extracted"
    files = sorted(folder.glob("report_*.json"))

    if not files:
        raise FileNotFoundError(
            "No extracted reports available"
        )

    reports = []

    for file in files:
        data = json.loads(
            file.read_text(encoding="utf-8")
        )

        if data["patient_id"] != patient_id:
            raise ValueError("Patient ID mismatch")

        report_date = extract_report_date(
            data["pages"]
        )

        symptoms = []

        for page in data["pages"]:
            extracted = extract_report_symptoms(
                page["text"]
            )

            for symptom in extracted:
                symptoms.append({
                    "name": symptom,
                    "normalized": normalize(symptom),
                    "page": page["page"]
                })

        reports.append({
            "source_file": data["original_filename"],
            "report_date": report_date,
            "symptoms": symptoms
        })

    return sorted(
        reports,
        key=lambda r: (
            r["report_date"] is None,
            r["report_date"] or "",
            r["source_file"]
        )
    )


def build_timeline(patient_id):
    reports = load_reports(patient_id)
    records = load_symptoms(patient_id)

    print("\n--- CARECOPILOT REPORT TIMELINE ---")
    print("Patient:", patient_id)
    print("Total reports:", len(reports))

    for report in reports:
        print("\nReport:", report["source_file"])
        print(
            "Report date:",
            report["report_date"] or "UNKNOWN"
        )
        print(
            "Documented symptoms:",
            ", ".join(
                s["name"] for s in report["symptoms"]
            ) or "None extracted"
        )

    print("\n--- HISTORICAL SYMPTOM COMPARISON ---")

    for record in records:
        if record.get("patient_id") != patient_id:
            raise ValueError(
                "Caregiver patient ID mismatch"
            )

        symptom = record["symptom"]
        onset = datetime.fromisoformat(
            record["onset_at"]
        )

        if onset.tzinfo is None:
            raise ValueError(
                "Caregiver onset timezone missing"
            )

        print("\nCaregiver symptom:", symptom)
        print("Severity:", record["severity"])
        print("Onset:", record["onset_at"])

        matches = []

        for report in reports:
            for documented in report["symptoms"]:
                if (
                    documented["normalized"]
                    == normalize(symptom)
                ):
                    matches.append({
                        "file": report["source_file"],
                        "page": documented["page"],
                        "date": report["report_date"]
                    })

        if not matches:
            print(
                "No matching symptom found in "
                "the ingested reports."
            )
            continue

        for match in matches:
            print(
                f"\nSource: {match['file']}, "
                f"page {match['page']}"
            )

            report_date = match["date"]

            if report_date is None:
                print("Report date: UNKNOWN")
                print(
                    "Chronology: Cannot determine "
                    "when the report was written."
                )
                continue

            report_day = date.fromisoformat(
                report_date
            )
            onset_day = onset.date()

            print("Report date:", report_date)

            if report_day < onset_day:
                print(
                    "Chronology: Report is dated "
                    "BEFORE the caregiver symptom onset."
                )

            elif report_day > onset_day:
                print(
                    "Chronology: Report is dated "
                    "AFTER the caregiver symptom onset."
                )

            else:
                print(
                    "Chronology: Same calendar date; "
                    "exact order is unknown."
                )

    print("\n--- IMPORTANT LIMITATIONS ---")
    print(
        "A report date is not necessarily the "
        "date a symptom first occurred."
    )
    print(
        "Textual matching does not prove symptom "
        "recurrence, progression, or diagnosis."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patient-id",
        required=True
    )

    args = parser.parse_args()

    build_timeline(args.patient_id)

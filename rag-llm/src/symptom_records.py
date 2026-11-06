
import argparse
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

BASE = (
    Path(__file__).resolve().parents[1]
    / "knowledge_base"
    / "patients"
)


def get_path(patient_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", patient_id):
        raise ValueError("Invalid patient ID")

    # Synthetic data only during development
    if patient_id != "demo_patient":
        raise ValueError("Use demo_patient for this test")

    return BASE / patient_id / "symptoms.json"


def load_symptoms(patient_id):
    path = get_path(patient_id)

    if not path.exists():
        return []

    records = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise ValueError("Invalid symptom record format")

    return records


def add_symptom(patient_id, symptom, severity, onset):
    path = get_path(patient_id)

    symptom = symptom.strip()
    if not symptom or len(symptom) > 120:
        raise ValueError("Invalid symptom name")

    try:
        onset_dt = datetime.fromisoformat(onset)
    except ValueError:
        raise ValueError("Onset must be an ISO 8601 datetime")

    if onset_dt.tzinfo is None or onset_dt.utcoffset() is None:
        raise ValueError("Onset must include a timezone")

    now = datetime.now(timezone.utc)

    if onset_dt > now:
        raise ValueError("Onset cannot be in the future")

    records = load_symptoms(patient_id)

    record = {
        "record_id": str(uuid.uuid4()),
        "patient_id": patient_id,
        "symptom": symptom,
        "severity": severity,
        "onset_at": onset_dt.isoformat(),
        "recorded_at": now.isoformat(),
        "source": "synthetic_caregiver_entry"
    }

    records.append(record)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Write a temporary file before replacing the original
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temp_path.replace(path)

    print("\n--- SYMPTOM SAVED ---")
    print("Patient:", patient_id)
    print("Symptom:", symptom)
    print("Severity:", severity)
    print("Onset:", record["onset_at"])
    print("Total records:", len(records))


def list_symptoms(patient_id):
    records = load_symptoms(patient_id)

    print("\n--- CARECOPILOT SYMPTOM HISTORY ---")
    print("Patient:", patient_id)
    print("Total records:", len(records))

    for i, record in enumerate(records, 1):
        print(f"\nRecord {i}")
        print("Symptom:", record["symptom"])
        print("Severity:", record["severity"])
        print("Onset:", record["onset_at"])
        print("Recorded:", record["recorded_at"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("--patient-id", required=True)
    add_parser.add_argument("--symptom", required=True)
    add_parser.add_argument(
        "--severity",
        choices=["mild", "moderate", "severe"],
        required=True
    )
    add_parser.add_argument("--onset", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--patient-id", required=True)

    args = parser.parse_args()

    if args.command == "add":
        add_symptom(
            args.patient_id,
            args.symptom,
            args.severity,
            args.onset
        )
    elif args.command == "list":
        list_symptoms(args.patient_id)

"""CareCopilot Phase 3: synthetic structured patient-history store.

Development fixture only. No API calls and no patient authorization.
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

PATIENT_ID = "demo_patient"
BASE = Path(__file__).resolve().parents[1] / "knowledge_base" / "patients"
CATEGORIES = {"condition", "diagnosis", "medication"}
SOURCE = "synthetic_caregiver_entry"

# Fictional, caregiver-reported examples. These are NOT facts from the PDFs.
DEMO_ENTRIES = [
    {
        "category": "condition",
        "name": "Seasonal allergies",
        "details": "Fictional caregiver-reported condition; not clinically verified.",
        "reported_on": "2026-10-12",
    },
    {
        "category": "diagnosis",
        "name": "Migraine",
        "details": "Fictional reported diagnosis; no confirming document supplied.",
        "reported_on": "2026-10-12",
    },
    {
        "category": "medication",
        "name": "Cetirizine",
        "details": "Fictional medication entry; dose and schedule unknown.",
        "reported_on": "2026-10-12",
    },
]


def history_path(patient_id):
    if patient_id != PATIENT_ID:
        raise ValueError("Only demo_patient is permitted in this prototype test.")
    return BASE / patient_id / "medical_history.json"


def normalized(value):
    return " ".join(value.casefold().split())


def record_id(patient_id, category, name):
    key = f"{patient_id}|{category}|{normalized(name)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def check_entry(entry, patient_id):
    if not isinstance(entry, dict):
        raise ValueError("Invalid medical-history entry.")
    if entry.get("patient_id") != patient_id:
        raise ValueError("Cross-patient history entry detected.")
    if entry.get("category") not in CATEGORIES:
        raise ValueError("Invalid history category.")
    name = entry.get("name")
    if not isinstance(name, str) or not name.strip() or len(name) > 120:
        raise ValueError("Invalid history entry name.")
    if not isinstance(entry.get("details"), str) or len(entry["details"]) > 400:
        raise ValueError("Invalid entry details.")
    if entry.get("source") != SOURCE or entry.get("verification_status") != "unverified":
        raise ValueError("Only unverified synthetic history entries are allowed.")
    if entry.get("record_id") != record_id(patient_id, entry["category"], name):
        raise ValueError("Invalid history record ID.")
    reported_on = entry.get("reported_on")
    if not isinstance(reported_on, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", reported_on):
        raise ValueError("Reported date must use YYYY-MM-DD.")
    if date.fromisoformat(reported_on) > date.today():
        raise ValueError("Reported date cannot be in the future.")


def load_history(patient_id):
    path = history_path(patient_id)
    if not path.exists():
        return {"schema_version": 1, "patient_id": patient_id, "entries": []}

    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or data.get("patient_id") != patient_id
        or not isinstance(data.get("entries"), list)
    ):
        raise ValueError("Invalid patient medical-history file.")

    ids = set()
    for entry in data["entries"]:
        check_entry(entry, patient_id)
        if entry["record_id"] in ids:
            raise ValueError("Duplicate history record detected.")
        ids.add(entry["record_id"])
    return data


def save_history(patient_id, data):
    path = history_path(patient_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement of one JSON file on the same filesystem.
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".medical_history_", suffix=".tmp", delete=False
        ) as handle:
            temp_name = handle.name
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def seed_demo(patient_id):
    data = load_history(patient_id)
    ids = {entry["record_id"] for entry in data["entries"]}
    added = 0
    for item in DEMO_ENTRIES:
        identity = record_id(patient_id, item["category"], item["name"])
        if identity in ids:
            continue
        entry = {
            "record_id": identity,
            "patient_id": patient_id,
            **item,
            "source": SOURCE,
            "verification_status": "unverified",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        check_entry(entry, patient_id)
        data["entries"].append(entry)
        ids.add(identity)
        added += 1

    if added:
        save_history(patient_id, data)

    print("\n--- CARECOPILOT HISTORY SEED ---")
    print("Patient:", patient_id)
    print("New entries:", added)
    print("Total entries:", len(data["entries"]))
    print("Saved:", history_path(patient_id))


def show_history(patient_id):
    data = load_history(patient_id)
    print("\n--- CARECOPILOT MEDICAL HISTORY ---")
    print("Patient:", patient_id)
    print("Total entries:", len(data["entries"]))
    for entry in data["entries"]:
        print(f"\n{entry['category'].upper()}: {entry['name']}")
        print("Reported on:", entry["reported_on"])
        print("Verification:", entry["verification_status"])
        print("Source:", entry["source"])
        print("Notes:", entry["details"])


def verify_history(patient_id):
    data = load_history(patient_id)
    indexed = {
        (entry["category"], normalized(entry["name"]))
        for entry in data["entries"]
    }
    expected = {(item["category"], normalized(item["name"])) for item in DEMO_ENTRIES}
    missing = expected - indexed
    if missing:
        raise AssertionError(f"Missing demo history entries: {sorted(missing)}")
    print("\n--- CARECOPILOT HISTORY VERIFICATION ---")
    print("Schema / patient isolation: PASS")
    print("Duplicate IDs: NONE")
    print("Three synthetic fixtures: PRESENT")
    print("Total entries:", len(data["entries"]))
    print("HISTORY VERIFICATION: PASS")


def main():
    parser = argparse.ArgumentParser(description="Synthetic medical history store")
    parser.add_argument("command", choices=["seed-demo", "show", "verify"])
    parser.add_argument("--patient-id", default=PATIENT_ID)
    args = parser.parse_args()
    if args.command == "seed-demo":
        seed_demo(args.patient_id)
    elif args.command == "show":
        show_history(args.patient_id)
    else:
        verify_history(args.patient_id)


if __name__ == "__main__":
    main()

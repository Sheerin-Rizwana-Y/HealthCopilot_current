from __future__ import annotations

from datetime import datetime, timezone

from ..auth import hash_password
from ..config import settings
from ..database import db


def seed_demo_data() -> None:
    if not settings.seed_demo_data:
        return
    if len(settings.jwt_secret_key) < 32:
        raise RuntimeError("Set JWT_SECRET_KEY to a random string of at least 32 characters in .env")
    if not settings.bootstrap_admin_password or settings.bootstrap_admin_password.startswith("ChangeThis"):
        raise RuntimeError("Set BOOTSTRAP_ADMIN_PASSWORD to a private password in .env before starting the API")

    now = datetime.now(timezone.utc)
    db.patients.update_one(
        {"patient_id": "demo_patient"},
        {"$setOnInsert": {
            "patient_id": "demo_patient",
            "display_name": "Demo Patient",
            "synthetic": True,
            "created_at": now,
            "status": "demo",
            "note": "Fictional test profile; not a real patient.",
        }},
        upsert=True,
    )
    db.users.update_one(
        {"username": settings.bootstrap_admin_username},
        {"$setOnInsert": {
            "username": settings.bootstrap_admin_username,
            "password_hash": hash_password(settings.bootstrap_admin_password),
            "role": "admin",
            "patient_ids": ["demo_patient"],
            "is_active": True,
            "created_at": now,
        }},
        upsert=True,
    )

    if db.vitals.count_documents({"patient_id": "demo_patient"}) == 0:
        db.vitals.insert_one({
            "patient_id": "demo_patient",
            "heart_rate_bpm": 82,
            "spo2_percent": 98.0,
            "temperature_c": 36.8,
            "fall_detected": False,
            "medicine_box_open": False,
            "source": "synthetic_demo_seed",
            "recorded_at": now,
            "synthetic": True,
        })

    if db.symptoms.count_documents({"patient_id": "demo_patient"}) == 0:
        db.symptoms.insert_many([
            {"patient_id": "demo_patient", "name": "dizziness", "severity": "moderate", "onset_at": datetime(2026, 10, 18, 9, tzinfo=timezone.utc), "notes": "Synthetic caregiver-reported demo entry.", "created_at": now, "source": "caregiver_reported", "verification_status": "unverified", "synthetic": True},
            {"patient_id": "demo_patient", "name": "cough", "severity": "mild", "onset_at": datetime(2026, 10, 18, 11, tzinfo=timezone.utc), "notes": "Synthetic caregiver-reported demo entry.", "created_at": now, "source": "caregiver_reported", "verification_status": "unverified", "synthetic": True},
        ])

    if db.history.count_documents({"patient_id": "demo_patient"}) == 0:
        db.history.insert_many([
            {"patient_id": "demo_patient", "category": "condition", "name": "Seasonal allergies", "details": "Fictional caregiver-reported condition; not clinically verified.", "reported_on": "2026-10-12", "created_at": now, "source": "synthetic_demo_seed", "verification_status": "unverified", "synthetic": True},
            {"patient_id": "demo_patient", "category": "diagnosis", "name": "Migraine", "details": "Fictional reported diagnosis; no confirming document supplied.", "reported_on": "2026-10-12", "created_at": now, "source": "synthetic_demo_seed", "verification_status": "unverified", "synthetic": True},
            {"patient_id": "demo_patient", "category": "medication", "name": "Cetirizine", "details": "Fictional medication entry; dose and schedule unknown.", "reported_on": "2026-10-12", "created_at": now, "source": "synthetic_demo_seed", "verification_status": "unverified", "synthetic": True},
        ])

    if db.medications.count_documents({"patient_id": "demo_patient"}) == 0:
        db.medications.insert_one({
            "patient_id": "demo_patient", "name": "Cetirizine", "dosage_text": "Not recorded", "schedule_text": "Not recorded",
            "notes": "Synthetic demo record only; no dose or schedule is inferred.", "created_at": now, "source": "synthetic_demo_seed",
            "verification_status": "unverified", "synthetic": True, "active": True,
        })

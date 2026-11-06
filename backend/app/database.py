from __future__ import annotations

from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ServerSelectionTimeoutError

from .config import settings

client = MongoClient(
    settings.mongo_uri,
    serverSelectionTimeoutMS=3500,
    connectTimeoutMS=3500,
    tz_aware=True,
)
db = client[settings.mongo_db]


def init_database() -> None:
    """Ping MongoDB and create query indexes; fail clearly if DB is unavailable."""
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as exc:
        raise RuntimeError(
            "MongoDB is not reachable. Start MongoDB with 'docker compose up -d mongodb' "
            "or configure MONGO_URI in the project-root .env file."
        ) from exc

    db.users.create_index([("username", ASCENDING)], unique=True)
    db.patients.create_index([("patient_id", ASCENDING)], unique=True)
    db.vitals.create_index([("patient_id", ASCENDING), ("recorded_at", DESCENDING)])
    db.symptoms.create_index([("patient_id", ASCENDING), ("onset_at", DESCENDING)])
    db.reports.create_index([("patient_id", ASCENDING), ("created_at", DESCENDING)])
    db.alerts.create_index([("patient_id", ASCENDING), ("created_at", DESCENDING)])
    db.history.create_index([("patient_id", ASCENDING), ("created_at", DESCENDING)])
    db.medications.create_index([("patient_id", ASCENDING), ("created_at", DESCENDING)])
    db.medication_events.create_index([("patient_id", ASCENDING), ("recorded_at", DESCENDING)])
    db.emergency_events.create_index([("patient_id", ASCENDING), ("status", ASCENDING), ("updated_at", DESCENDING)])
    db.emergency_events.create_index([("event_id", ASCENDING)], unique=True)
    # At most one unresolved emergency event per patient, even with concurrent sensor packets.
    db.emergency_events.create_index(
        [("patient_id", ASCENDING), ("active_key", ASCENDING)],
        unique=True,
        partialFilterExpression={"status": "active"},
    )
    db.monitoring_state.create_index([("patient_id", ASCENDING)], unique=True)
    db.emergency_contacts.create_index([("patient_id", ASCENDING), ("priority", ASCENDING)])
    db.emergency_contacts.create_index([("contact_id", ASCENDING)], unique=True)
    db.notification_jobs.create_index([("idempotency_key", ASCENDING)], unique=True)
    db.notification_jobs.create_index([("status", ASCENDING), ("next_attempt_at", ASCENDING)])


def close_database() -> None:
    client.close()

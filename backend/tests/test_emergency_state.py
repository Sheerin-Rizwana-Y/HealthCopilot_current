from __future__ import annotations

import copy
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.services import emergency, notification_service
from pymongo.errors import DuplicateKeyError


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def sort(self, key, direction=None):
        if isinstance(key, list):
            for field, order in reversed(key):
                self.rows.sort(key=lambda row: row.get(field) or datetime.min.replace(tzinfo=timezone.utc), reverse=order < 0)
        else:
            self.rows.sort(key=lambda row: row.get(key) or datetime.min.replace(tzinfo=timezone.utc), reverse=(direction or 1) < 0)
        return self

    def limit(self, count):
        self.rows = self.rows[:count]
        return self

    def __iter__(self):
        return iter(self.rows)


def matches(document, query):
    for key, expected in query.items():
        actual = document.get(key)
        if isinstance(expected, dict) and any(str(k).startswith("$") for k in expected):
            for operator, value in expected.items():
                if operator == "$in" and actual not in value: return False
                if operator == "$gt" and not (actual is not None and actual > value): return False
                if operator == "$gte" and not (actual is not None and actual >= value): return False
                if operator == "$lt" and not (actual is not None and actual < value): return False
                if operator == "$lte" and not (actual is not None and actual <= value): return False
                if operator == "$ne" and actual == value: return False
        elif actual != expected:
            return False
    return True


class FakeCollection:
    def __init__(self):
        self.rows = []
        self.counter = 0

    def find_one(self, query, projection=None, sort=None):
        rows = [copy.deepcopy(row) for row in self.rows if matches(row, query)]
        if sort:
            FakeCursor(rows).sort(sort)
        return rows[0] if rows else None

    def find(self, query=None):
        query = query or {}
        return FakeCursor([copy.deepcopy(row) for row in self.rows if matches(row, query)])

    def insert_one(self, document):
        if document.get("idempotency_key") and any(row.get("idempotency_key") == document["idempotency_key"] for row in self.rows):
            raise DuplicateKeyError("duplicate idempotency key")
        self.counter += 1
        row = copy.deepcopy(document)
        row.setdefault("_id", self.counter)
        self.rows.append(row)
        return type("InsertResult", (), {"inserted_id": row["_id"]})()

    def _apply(self, row, update, inserting=False):
        for key, value in update.get("$setOnInsert", {}).items():
            if inserting: row[key] = copy.deepcopy(value)
        for key, value in update.get("$set", {}).items():
            row[key] = copy.deepcopy(value)
        for key, value in update.get("$push", {}).items():
            row.setdefault(key, []).append(copy.deepcopy(value))

    def update_one(self, query, update, upsert=False):
        for row in self.rows:
            if matches(row, query):
                self._apply(row, update)
                return type("UpdateResult", (), {"matched_count": 1, "modified_count": 1})()
        if upsert:
            seed = {key: value for key, value in query.items() if not isinstance(value, dict)}
            self._apply(seed, update, inserting=True)
            self.insert_one(seed)
            return type("UpdateResult", (), {"matched_count": 0, "modified_count": 1, "upserted_id": self.counter})()
        return type("UpdateResult", (), {"matched_count": 0, "modified_count": 0})()

    def update_many(self, query, update):
        count = 0
        for row in self.rows:
            if matches(row, query):
                self._apply(row, update)
                count += 1
        return type("UpdateResult", (), {"matched_count": count, "modified_count": count})()


class FakeDB:
    def __init__(self):
        for name in ("vitals", "monitoring_state", "emergency_events", "emergency_contacts", "notification_jobs"):
            setattr(self, name, FakeCollection())


class EmergencyStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.original_emergency_db = emergency.db
        self.original_notification_db = notification_service.db
        self.fake_db = FakeDB()
        emergency.db = self.fake_db
        notification_service.db = self.fake_db
        self.original_settings = {
            key: getattr(settings, key) for key in (
                "inactivity_threshold_seconds", "inactivity_response_timeout_seconds",
                "local_alarm_active_seconds", "local_alarm_snooze_seconds",
                "warning_alarm_active_seconds", "sensor_stale_after_seconds",
                "vital_persistence_seconds", "twilio_enabled", "twilio_account_sid", "twilio_auth_token",
                "twilio_from_number", "emergency_sms_enabled", "emergency_calls_enabled", "emergency_contact_encryption_key",
            )
        }
        settings.inactivity_threshold_seconds = 5
        settings.inactivity_response_timeout_seconds = 5
        settings.local_alarm_active_seconds = 5
        settings.local_alarm_snooze_seconds = 10
        settings.warning_alarm_active_seconds = 5
        settings.sensor_stale_after_seconds = 60
        settings.vital_persistence_seconds = 1
        settings.twilio_enabled = False
        settings.twilio_account_sid = ""
        settings.twilio_auth_token = ""
        settings.twilio_from_number = ""
        settings.emergency_sms_enabled = False
        settings.emergency_calls_enabled = False
        settings.emergency_contact_encryption_key = ""

    def tearDown(self):
        emergency.db = self.original_emergency_db
        notification_service.db = self.original_notification_db
        for key, value in self.original_settings.items():
            setattr(settings, key, value)

    def packet(self, patient_id="p1", **overrides):
        reading = {
            "patient_id": patient_id,
            "recorded_at": datetime.now(timezone.utc),
            "heart_rate_bpm": 82,
            "spo2_percent": 98,
            "temperature_c": 36.8,
            "movement_detected": True,
            "motion_event": "normal_motion",
            "signal_quality": "good",
            "fall_detected": False,
            "source": "simulator",
        }
        reading.update(overrides)
        self.fake_db.vitals.insert_one(reading)
        return reading

    def test_unknown_signal_quality_never_verifies_green(self):
        reading = self.packet(signal_quality="unknown")
        emergency.on_reading("p1", reading)
        state = self.fake_db.monitoring_state.find_one({"patient_id": "p1"})
        self.assertFalse(state["normal_verified"])
        self.assertEqual(emergency.emergency_status("p1")["monitoring_status"], "monitoring_unverified")

    def test_fall_creates_one_critical_event_for_repeated_packets(self):
        first = self.packet(fall_detected=True, movement_detected=False, motion_event="fall_detected")
        emergency.on_reading("p1", first)
        second = self.packet(fall_detected=True, movement_detected=False, motion_event="fall_detected")
        emergency.on_reading("p1", second)
        events = list(self.fake_db.emergency_events.find({"patient_id": "p1", "status": "active"}))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["severity"], "critical")
        self.assertEqual(events[0]["event_type"], "fall_detected")

    def test_inactivity_without_response_escalates_to_critical(self):
        now = datetime.now(timezone.utc)
        self.fake_db.monitoring_state.insert_one({
            "patient_id": "p1", "latest_reading": {"heart_rate_bpm": 82, "spo2_percent": 98, "temperature_c": 36.8, "signal_quality": "good"},
            "last_reading_received_at": now, "movement_known": True, "movement_detected": False,
            "still_since": now - timedelta(seconds=10), "vital_severity": "normal", "vital_reasons": [],
            "normal_verified": False, "updated_at": now,
        })
        emergency.monitor_once()
        event = self.fake_db.emergency_events.find_one({"patient_id": "p1", "status": "active"})
        self.assertEqual(event["event_type"], "inactivity_checkin")
        self.assertEqual(event["severity"], "warning")
        self.fake_db.emergency_events.update_one({"event_id": event["event_id"]}, {"$set": {"response_deadline": now - timedelta(seconds=1)}})
        emergency.monitor_once()
        event = self.fake_db.emergency_events.find_one({"event_id": event["event_id"]})
        self.assertEqual(event["event_type"], "inactivity_no_response")
        self.assertEqual(event["severity"], "critical")
        self.assertEqual(event["local_alarm_state"], "active")

    def test_critical_alarm_snoozes_and_rearms_without_resetting_notification_retry_generation(self):
        reading = self.packet(fall_detected=True, movement_detected=False, motion_event="fall_detected")
        emergency.on_reading("p1", reading)
        event = self.fake_db.emergency_events.find_one({"patient_id": "p1", "status": "active"})
        original_generation = event["notification_generation"]
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        self.fake_db.emergency_events.update_one({"event_id": event["event_id"]}, {"$set": {"alarm_until": past}})
        emergency.monitor_once()
        event = self.fake_db.emergency_events.find_one({"event_id": event["event_id"]})
        self.assertEqual(event["state"], "snoozed")
        self.fake_db.emergency_events.update_one({"event_id": event["event_id"]}, {"$set": {"snooze_until": past}})
        emergency.monitor_once()
        event = self.fake_db.emergency_events.find_one({"event_id": event["event_id"]})
        self.assertEqual(event["state"], "critical")
        self.assertEqual(event["local_alarm_state"], "active")
        self.assertEqual(event["notification_generation"], original_generation)

    def test_new_fall_during_snooze_reactivates_and_starts_new_notification_generation(self):
        reading = self.packet(fall_detected=True, movement_detected=False, motion_event="fall_detected")
        emergency.on_reading("p1", reading)
        event = self.fake_db.emergency_events.find_one({"patient_id": "p1", "status": "active"})
        self.fake_db.emergency_events.update_one({"event_id": event["event_id"]}, {"$set": {
            "state": "snoozed", "local_alarm_state": "snoozed", "snooze_until": datetime.now(timezone.utc) + timedelta(seconds=30),
        }})
        new_fall = self.packet(fall_detected=True, movement_detected=False, motion_event="fall_detected")
        emergency.on_reading("p1", new_fall)
        event = self.fake_db.emergency_events.find_one({"event_id": event["event_id"]})
        self.assertEqual(event["state"], "critical")
        self.assertEqual(event["local_alarm_state"], "active")
        self.assertEqual(event["notification_generation"], 2)

    def test_notification_jobs_are_idempotent_and_do_not_store_phone_numbers(self):
        settings.twilio_enabled = True
        settings.twilio_account_sid = "AC-test"
        settings.twilio_auth_token = "test-token"
        settings.twilio_from_number = "+15550000000"
        settings.emergency_sms_enabled = True
        settings.emergency_calls_enabled = True
        self.fake_db.emergency_contacts.insert_one({"patient_id": "p1", "contact_id": "primary", "priority": 1, "phone_encrypted": "encrypted-one"})
        self.fake_db.emergency_contacts.insert_one({"patient_id": "p1", "contact_id": "secondary", "priority": 2, "phone_encrypted": "encrypted-two"})
        event = {"event_id": "event-test", "patient_id": "p1", "severity": "critical", "event_type": "fall_detected", "created_at": datetime.now(timezone.utc), "latest_readings": {"heart_rate_bpm": 82}}
        self.fake_db.emergency_events.insert_one({**event, "status": "active"})
        notification_service.queue_event_notifications(event, critical=True, generation=1)
        notification_service.queue_event_notifications(event, critical=True, generation=1)
        jobs = list(self.fake_db.notification_jobs.find({"event_id": "event-test"}))
        self.assertEqual(len(jobs), 3)  # primary SMS + primary call + secondary SMS
        self.assertTrue(all("to_number" not in job and "phone" not in job for job in jobs))
        self.assertEqual(self.fake_db.emergency_events.find_one({"event_id": "event-test"})["notification_status"], "queued")

    def test_new_critical_vital_with_no_previous_measurement_reactivates_during_snooze(self):
        first = self.packet(heart_rate_bpm=None, spo2_percent=None, temperature_c=None, fall_detected=True, movement_detected=False, motion_event="fall_detected")
        emergency.on_reading("p1", first)
        event = self.fake_db.emergency_events.find_one({"patient_id": "p1", "status": "active"})
        self.fake_db.emergency_events.update_one({"event_id": event["event_id"]}, {"$set": {
            "state": "snoozed", "local_alarm_state": "snoozed", "snooze_until": datetime.now(timezone.utc) + timedelta(seconds=30),
        }})
        new_critical = self.packet(heart_rate_bpm=35, spo2_percent=86, fall_detected=False, movement_detected=True, motion_event="normal_motion")
        emergency.on_reading("p1", new_critical)
        event = self.fake_db.emergency_events.find_one({"event_id": event["event_id"]})
        self.assertEqual(event["state"], "critical")
        self.assertEqual(event["local_alarm_state"], "active")

    def test_person_response_does_not_resolve_event_or_cancel_critical_retry_jobs(self):
        reading = self.packet(fall_detected=True, movement_detected=False, motion_event="fall_detected")
        emergency.on_reading("p1", reading)
        event = self.fake_db.emergency_events.find_one({"patient_id": "p1", "status": "active"})
        # Simulate a critical event that supports a person response, then verify it remains active.
        self.fake_db.emergency_events.update_one({"event_id": event["event_id"]}, {"$set": {"event_type": "inactivity_no_response"}})
        result = emergency.acknowledge_event("p1", event["event_id"], "monitored_person", person_response=True)
        self.assertEqual(result["status"], "active")
        self.assertTrue(result["person_responded"])
        self.assertEqual(result["state"], "snoozed")


if __name__ == "__main__":
    unittest.main(verbosity=2)

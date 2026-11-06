from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.schemas import EmergencyContactInput, EmergencyContactsUpdate, DeviceIngestRequest
from app.services.alerts import build_rule_alerts
from app.services.contact_crypto import decrypt_phone, encrypt_phone
from app.services.emergency import classify_vitals, detect_movement
from cryptography.fernet import Fernet


class EmergencyRuleTests(unittest.TestCase):
    def test_normal_complete_vitals_classify_normal(self):
        result = classify_vitals({"heart_rate_bpm": 82, "spo2_percent": 98, "temperature_c": 36.8, "signal_quality": "good"})
        self.assertEqual(result["severity"], "normal")

    def test_critical_heart_rate_is_critical(self):
        result = classify_vitals({"heart_rate_bpm": settings.critical_hr_low - 1, "spo2_percent": 98, "temperature_c": 36.8})
        self.assertEqual(result["severity"], "critical")

    def test_warning_vitals_are_not_immediately_critical(self):
        result = classify_vitals({"heart_rate_bpm": settings.warning_hr_high + 1, "spo2_percent": 98, "temperature_c": 36.8})
        self.assertEqual(result["severity"], "warning")

    def test_large_heart_rate_change_is_detected_as_trend_warning(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        baseline = {"heart_rate_bpm": 80, "spo2_percent": 98, "temperature_c": 36.8, "signal_quality": "good", "recorded_at": now - timedelta(seconds=60)}
        current = {"heart_rate_bpm": 80 + settings.hr_trend_change_bpm, "spo2_percent": 98, "temperature_c": 36.8, "signal_quality": "good", "recorded_at": now}
        result = classify_vitals(current, baseline)
        self.assertEqual(result["severity"], "warning")
        self.assertIn("hr_trend", result["signature"])

    def test_poor_signal_is_not_classified_as_normal_or_critical_from_values_alone(self):
        result = classify_vitals({"heart_rate_bpm": 30, "spo2_percent": 80, "temperature_c": 36.8, "signal_quality": "poor"})
        self.assertEqual(result["severity"], "unknown")
        self.assertIn("poor", result["signature"])

    def test_missing_movement_is_unknown_not_inactive(self):
        self.assertIsNone(detect_movement({"heart_rate_bpm": 80}))

    def test_explicit_stillness_is_detected(self):
        self.assertFalse(detect_movement({"motion_event": "still"}))
        self.assertFalse(detect_movement({"movement_detected": False}))

    def test_explicit_movement_is_detected(self):
        self.assertTrue(detect_movement({"motion_event": "normal_motion"}))
        self.assertTrue(detect_movement({"movement_detected": True}))

    def test_poor_signal_creates_quality_warning_not_vital_threshold_alerts(self):
        alerts = build_rule_alerts("demo_patient", {"heart_rate_bpm": 30, "spo2_percent": 80, "signal_quality": "poor"})
        codes = {item["code"] for item in alerts}
        self.assertIn("sensor_signal_unreliable", codes)
        self.assertNotIn("heart_rate_outside_demo_range", codes)
        self.assertNotIn("spo2_below_demo_threshold", codes)

    def test_fall_remains_independent_even_with_poor_signal(self):
        alerts = build_rule_alerts("demo_patient", {"fall_detected": True, "signal_quality": "poor"})
        codes = {item["code"] for item in alerts}
        self.assertIn("possible_fall", codes)
        self.assertIn("sensor_signal_unreliable", codes)

    def test_emergency_contact_numbers_are_encrypted_at_rest(self):
        old_key = settings.emergency_contact_encryption_key
        settings.emergency_contact_encryption_key = Fernet.generate_key().decode()
        try:
            token = encrypt_phone("+15551234567")
            self.assertNotEqual(token, "+15551234567")
            self.assertEqual(decrypt_phone(token), "+15551234567")
        finally:
            settings.emergency_contact_encryption_key = old_key

    def test_emergency_contact_requires_e164_format(self):
        with self.assertRaises(Exception):
            EmergencyContactInput(name="Test", phone="555-1234", priority=1)
        valid = EmergencyContactInput(name="Demo Contact", phone="+15551234567", priority=1)
        self.assertEqual(valid.priority, 1)

    def test_duplicate_contact_priorities_are_rejected(self):
        with self.assertRaises(Exception):
            EmergencyContactsUpdate(contacts=[
                {"name": "A", "phone": "+15551234567", "priority": 1},
                {"name": "B", "phone": "+15551234568", "priority": 1},
            ])

    def test_device_payload_accepts_additive_motion_fields(self):
        payload = DeviceIngestRequest(patient_id="demo_patient", device_id="esp32-test", movement_detected=False, motion_event="still")
        self.assertFalse(payload.movement_detected)

    def test_device_payload_can_report_person_checkin_without_vital_packet(self):
        payload = DeviceIngestRequest(patient_id="demo_patient", device_id="esp32-test", person_response=True)
        self.assertTrue(payload.person_response)


if __name__ == "__main__":
    unittest.main(verbosity=2)

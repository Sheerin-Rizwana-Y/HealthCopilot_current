from __future__ import annotations

import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.auth import hash_password, verify_password
from app.schemas import VitalCreate
from app.services.alerts import build_rule_alerts


class AuthTests(unittest.TestCase):
    def test_password_hash_round_trip(self):
        stored = hash_password("a-long-demo-password")
        self.assertNotEqual(stored, "a-long-demo-password")
        self.assertTrue(verify_password("a-long-demo-password", stored))
        self.assertFalse(verify_password("wrong-password", stored))


class ValidationTests(unittest.TestCase):
    def test_vital_payload_accepts_sensible_demo_values(self):
        item = VitalCreate(heart_rate_bpm=82, spo2_percent=98, temperature_c=36.8)
        self.assertEqual(item.heart_rate_bpm, 82)

    def test_vital_payload_rejects_out_of_range_spo2(self):
        with self.assertRaises(Exception):
            VitalCreate(heart_rate_bpm=82, spo2_percent=120, temperature_c=36.8)


class AlertTests(unittest.TestCase):
    def test_possible_fall_creates_independent_alert(self):
        alerts = build_rule_alerts("demo_patient", {"fall_detected": True})
        self.assertTrue(any(item["code"] == "possible_fall" for item in alerts))
        self.assertEqual(alerts[0]["source"], "independent_rule_engine")

    def test_normal_demo_values_do_not_trigger_rule_alert(self):
        alerts = build_rule_alerts("demo_patient", {
            "heart_rate_bpm": 82,
            "spo2_percent": 98,
            "temperature_c": 36.8,
            "fall_detected": False,
        })
        self.assertEqual(alerts, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

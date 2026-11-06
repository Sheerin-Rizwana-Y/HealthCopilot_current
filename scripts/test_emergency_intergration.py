"""End-to-end emergency API test using a newly created synthetic patient.

Requires the backend, MongoDB, and a valid project-root .env. Twilio calls/SMS
remain disabled unless deliberately configured. The script never prints secrets.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

BASE = os.getenv("LIFORA_API_URL", "http://127.0.0.1:8000").rstrip("/")
USERNAME = os.getenv("LIFORA_TEST_USERNAME") or os.getenv("BOOTSTRAP_ADMIN_USERNAME", "caregiver@example.com")
PASSWORD = os.getenv("LIFORA_TEST_PASSWORD") or os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")
DEVICE_TOKEN = os.getenv("DEVICE_INGEST_TOKEN", "").strip()


def request(path: str, data=None, *, token: str | None = None, device: bool = False, method: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if device:
        headers["X-Device-Token"] = DEVICE_TOKEN
    verb = method or ("POST" if data is not None else "GET")
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(BASE + path, data=body, headers=headers, method=verb)
    with urllib.request.urlopen(req, timeout=20) as response:
        text = response.read().decode("utf-8")
        return json.loads(text) if text else {}


def send_reading(patient_id: str, packet: dict):
    return request("/api/v1/devices/ingest", {
        "patient_id": patient_id,
        "device_id": "lifora-integration-test",
        "signal_quality": "good",
        **packet,
    }, device=True)


def current_status(patient_id: str, token: str):
    return request(f"/api/v1/patients/{patient_id}/emergency", token=token)


def assert_state(patient_id: str, token: str, expected: str, label: str):
    result = current_status(patient_id, token)
    actual = result.get("monitoring_status")
    assert actual == expected, f"{label}: expected {expected}, got {actual}; status={result}"
    print(f"PASS: {label} -> {actual}")
    return result


def resolve_active(patient_id: str, token: str):
    result = current_status(patient_id, token)
    for event in result.get("active_events", []):
        request(f"/api/v1/patients/{patient_id}/emergency-events/{event['event_id']}/resolve", {
            "safety_confirmed": True,
            "notes": "Synthetic integration-test event; demo safety state verified.",
        }, token=token)


def main() -> int:
    if not PASSWORD:
        raise SystemExit("BOOTSTRAP_ADMIN_PASSWORD is missing from the project-root .env file")
    if not DEVICE_TOKEN:
        raise SystemExit("DEVICE_INGEST_TOKEN is missing from the project-root .env file")
    print("[1] API health")
    health = request("/api/v1/health")
    assert health.get("status") == "ok"
    print("PASS: API health")
    print("[2] Authenticate test administrator")
    login = request("/api/v1/auth/login", {"username": USERNAME, "password": PASSWORD})
    token = login["access_token"]
    assert login.get("user", {}).get("role") == "admin", "The integration test requires an admin account to create a synthetic patient."
    print("PASS: authenticated")

    patient_id = "emergency_test_" + uuid.uuid4().hex[:8]
    request("/api/v1/patients", {"patient_id": patient_id, "display_name": "Synthetic Emergency Test", "synthetic": True}, token=token)
    print("Created synthetic test profile:", patient_id)

    print("[3] Normal verified state")
    send_reading(patient_id, {"heart_rate_bpm": 82, "spo2_percent": 98, "temperature_c": 36.8, "movement_detected": True, "motion_event": "normal_motion"})
    assert_state(patient_id, token, "normal_verified", "normal readings plus movement")

    print("[4] Poor sensor quality is not normal")
    send_reading(patient_id, {"heart_rate_bpm": 30, "spo2_percent": 80, "temperature_c": 36.8, "movement_detected": True, "motion_event": "normal_motion", "signal_quality": "poor"})
    assert_state(patient_id, token, "monitoring_unverified", "poor-quality sensor values")
    status = current_status(patient_id, token)
    assert not any(event.get("severity") == "critical" for event in status.get("active_events", [])), "Poor signal alone must not create a vital-based critical event."
    print("PASS: poor signal did not trigger a vital-based critical event")

    print("[5] Critical vitals create an emergency event")
    send_reading(patient_id, {"heart_rate_bpm": 35, "spo2_percent": 86, "temperature_c": 36.8, "movement_detected": True, "motion_event": "normal_motion"})
    status = assert_state(patient_id, token, "critical", "critical vital thresholds")
    event = status["active_events"][0]
    assert event["severity"] == "critical" and event["local_alarm_state"] == "active"
    event_id = event["event_id"]
    print("PASS: critical event and local alarm state recorded")

    print("[6] Acknowledgment does not resolve the event")
    request(f"/api/v1/patients/{patient_id}/emergency-events/{event_id}/acknowledge", token=token, method="POST")
    status = assert_state(patient_id, token, "acknowledged_unresolved", "caregiver acknowledgment")
    assert status["active_events"][0]["status"] == "active"
    print("PASS: acknowledged event remains unresolved")

    print("[7] Worsening critical readings reactivate the alert")
    send_reading(patient_id, {"heart_rate_bpm": 25, "spo2_percent": 80, "temperature_c": 36.8, "movement_detected": True, "motion_event": "normal_motion"})
    status = assert_state(patient_id, token, "critical", "worsening after acknowledgment")
    assert status["active_events"][0]["local_alarm_state"] == "active"
    print("PASS: critical alarm reactivated")

    print("[8] Safety confirmation plus normal readings allow resolution")
    send_reading(patient_id, {"heart_rate_bpm": 82, "spo2_percent": 98, "temperature_c": 36.8, "movement_detected": True, "motion_event": "normal_motion"})
    resolve_active(patient_id, token)
    assert_state(patient_id, token, "normal_verified", "resolved event with verified normal demo readings")

    print("[9] Fall event is immediate and repeated packets are deduplicated")
    fall_packet = {"heart_rate_bpm": 82, "spo2_percent": 98, "temperature_c": 36.8, "movement_detected": False, "motion_event": "fall_detected", "fall_detected": True}
    send_reading(patient_id, fall_packet)
    first = current_status(patient_id, token)
    assert first["monitoring_status"] == "critical"
    send_reading(patient_id, fall_packet)
    second = current_status(patient_id, token)
    assert len(second["active_events"]) == 1, f"Expected one unresolved event, got {len(second['active_events'])}"
    assert second["active_events"][0]["event_type"] == "fall_detected"
    print("PASS: fall triggers immediately and repeated packets do not create duplicate emergency events")

    print("[10] Close the synthetic test event")
    send_reading(patient_id, {"heart_rate_bpm": 82, "spo2_percent": 98, "temperature_c": 36.8, "movement_detected": True, "motion_event": "normal_motion"})
    resolve_active(patient_id, token)
    print("PASS: integration scenarios completed")
    print("Note: created profile remains in MongoDB for audit; it is synthetic. Twilio delivery is not tested by this script.")
    print("EMERGENCY INTEGRATION TEST: PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read().decode("utf-8", errors="replace"))
        raise SystemExit(1)
    except urllib.error.URLError as exc:
        print("Could not reach Lifora API:", exc.reason)
        raise SystemExit(1)

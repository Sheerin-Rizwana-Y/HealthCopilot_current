"""Small local API smoke test using the Python standard library."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

BASE = os.getenv("LIFORA_API_URL", "http://127.0.0.1:8000").rstrip("/")
USERNAME = os.getenv("BOOTSTRAP_ADMIN_USERNAME", "caregiver@example.com")
PASSWORD = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")


def request(path: str, data=None, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(data).encode("utf-8") if data is not None else None,
        headers=headers,
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        body = response.read().decode("utf-8")
        return json.loads(body) if body else {}


def main():
    print("[1] API health")
    print(request("/api/v1/health"))
    if not PASSWORD:
        raise SystemExit("Set BOOTSTRAP_ADMIN_PASSWORD in .env before running the authenticated smoke test")
    print("[2] Login")
    login = request("/api/v1/auth/login", {"username": USERNAME, "password": PASSWORD})
    token = login["access_token"]
    print("Login: PASS")
    print("[3] Read patient list")
    patients = request("/api/v1/patients", token=token)
    assert any(p.get("patient_id") == "demo_patient" for p in patients), "demo_patient missing"
    print("Patient list: PASS; patients =", len(patients))
    print("[4] Read system status")
    status = request("/api/v1/system/status", token=token)
    assert status.get("database") == "connected"
    assert "emergency_monitoring" in status
    print("System status and emergency-monitor config: PASS")
    print("[5] Read emergency status for synthetic demo patient")
    emergency = request("/api/v1/patients/demo_patient/emergency", token=token)
    assert emergency.get("patient_id") == "demo_patient"
    assert "monitoring_status" in emergency
    print("Emergency status API: PASS; state =", emergency["monitoring_status"])
    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()

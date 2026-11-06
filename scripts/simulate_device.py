"""Send one synthetic sensor packet to the Lifora device-ingest API."""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv
    ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--patient-id", default="demo_patient")
    parser.add_argument("--fall", action="store_true", help="Simulate a possible fall to exercise the alert path")
    args = parser.parse_args()

    token = os.getenv("DEVICE_INGEST_TOKEN", "").strip()
    if not token:
        raise SystemExit("DEVICE_INGEST_TOKEN is missing from the project-root .env file")

    packet = {
        "patient_id": args.patient_id,
        "device_id": "esp32-simulator-01",
        "heart_rate_bpm": 82,
        "spo2_percent": 98.0,
        "temperature_c": 36.8,
        "fall_detected": args.fall,
        "medicine_box_open": False,
        "motion_event": "simulated_fall" if args.fall else "normal_motion",
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/api/v1/devices/ingest",
        data=json.dumps(packet).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Device-Token": token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            print("HTTP", response.status)
            print(json.dumps(json.loads(response.read().decode("utf-8")), indent=2))
            return 0
    except urllib.error.HTTPError as exc:
        print("HTTP", exc.code, exc.read().decode("utf-8", errors="replace"))
        return 1
    except urllib.error.URLError as exc:
        print("Could not reach Lifora API:", exc.reason)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

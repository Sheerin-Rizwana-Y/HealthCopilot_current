from __future__ import annotations

from datetime import datetime, timezone
from ..config import settings


def build_rule_alerts(patient_id: str, reading: dict) -> list[dict]:
    """Deterministic prototype rules; these thresholds are not clinically validated."""
    now = datetime.now(timezone.utc)
    alerts: list[dict] = []

    if reading.get("fall_detected"):
        alerts.append({
            "patient_id": patient_id,
            "code": "possible_fall",
            "severity": "high",
            "title": "Possible fall event",
            "message": "The existing motion/fall input indicated a possible fall. Check the person and follow your established emergency plan; do not wait for an AI explanation.",
            "source": "independent_rule_engine",
            "created_at": now,
            "acknowledged": False,
            "clinical_validation": "not_validated_prototype_rule",
        })

    if reading.get("signal_quality") == "poor":
        alerts.append({
            "patient_id": patient_id,
            "code": "sensor_signal_unreliable",
            "severity": "medium",
            "title": "Sensor signal is unreliable",
            "message": "One or more vital readings are marked poor quality. Verify sensor placement and signal before interpreting values; poor signal is not treated as normal.",
            "source": "independent_rule_engine",
            "created_at": now,
            "acknowledged": False,
            "clinical_validation": "not_validated_prototype_rule",
        })
        return alerts

    hr = reading.get("heart_rate_bpm")
    if hr is not None and (hr < settings.demo_hr_min or hr > settings.demo_hr_max):
        alerts.append({
            "patient_id": patient_id,
            "code": "heart_rate_outside_demo_range",
            "severity": "medium",
            "title": "Heart-rate value outside demo range",
            "message": f"Recorded heart rate: {hr} bpm. This is a configurable software test threshold, not a diagnosis or validated clinical alarm.",
            "source": "independent_rule_engine",
            "created_at": now,
            "acknowledged": False,
            "clinical_validation": "not_validated_prototype_rule",
        })

    spo2 = reading.get("spo2_percent")
    if spo2 is not None and spo2 < settings.demo_spo2_low:
        alerts.append({
            "patient_id": patient_id,
            "code": "spo2_below_demo_threshold",
            "severity": "high",
            "title": "SpO₂ below demo threshold",
            "message": f"Recorded SpO₂: {spo2}%. Verify sensor placement and follow clinician-approved escalation guidance. The prototype threshold requires clinical review.",
            "source": "independent_rule_engine",
            "created_at": now,
            "acknowledged": False,
            "clinical_validation": "not_validated_prototype_rule",
        })

    temperature = reading.get("temperature_c")
    if temperature is not None and temperature >= settings.demo_temperature_high_c:
        alerts.append({
            "patient_id": patient_id,
            "code": "temperature_above_demo_threshold",
            "severity": "medium",
            "title": "Temperature above demo threshold",
            "message": f"Recorded temperature: {temperature} °C. This configurable prototype threshold is not clinically validated.",
            "source": "independent_rule_engine",
            "created_at": now,
            "acknowledged": False,
            "clinical_validation": "not_validated_prototype_rule",
        })

    return alerts

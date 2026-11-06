"""Event-driven emergency monitoring and escalation for Lifora.

This module is an additive rule layer. It consumes the existing `fall_detected`
flag and sensor readings; it does not load, retrain, or change any ML model.
Thresholds are configurable prototype defaults, not clinically validated limits.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo.errors import DuplicateKeyError

from ..config import settings
from ..database import db
from .notification_service import cancel_event_jobs, dispatch_due_jobs, queue_event_notifications

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dt(value: Any, fallback: datetime | None = None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return fallback or utcnow()


def classify_vitals(reading: dict | None, baseline: dict | None = None) -> dict:
    """Classify thresholds and coarse trends using configurable demo settings only."""
    reading = reading or {}
    baseline = baseline or {}
    if baseline and baseline.get("signal_quality") != "good":
        baseline = {}
    if reading.get("signal_quality") == "poor":
        return {"severity": "unknown", "reasons": ["Sensor signal quality is marked poor; readings are not treated as verified normal."], "signature": "poor_signal"}

    critical: list[str] = []
    warning: list[str] = []
    warning_codes: list[str] = []
    hr = reading.get("heart_rate_bpm")
    spo2 = reading.get("spo2_percent")
    temp = reading.get("temperature_c")

    if hr is not None:
        if hr < settings.critical_hr_low:
            critical.append(f"heart rate below configured critical test threshold ({hr} bpm)")
        elif hr > settings.critical_hr_high:
            critical.append(f"heart rate above configured critical test threshold ({hr} bpm)")
        elif hr < settings.warning_hr_low:
            warning.append(f"heart rate outside configured warning test range ({hr} bpm)")
            warning_codes.append("hr_low")
        elif hr > settings.warning_hr_high:
            warning.append(f"heart rate outside configured warning test range ({hr} bpm)")
            warning_codes.append("hr_high")

    if spo2 is not None:
        if spo2 < settings.critical_spo2_low:
            critical.append(f"SpO₂ below configured critical test threshold ({spo2}%)")
        elif spo2 < settings.warning_spo2_low:
            warning.append(f"SpO₂ below configured warning test threshold ({spo2}%)")
            warning_codes.append("spo2_low")

    if temp is not None:
        if temp >= settings.critical_temp_high_c:
            critical.append(f"temperature above configured critical test threshold ({temp} °C)")
        elif temp >= settings.warning_temp_high_c:
            warning.append(f"temperature above configured warning test threshold ({temp} °C)")
            warning_codes.append("temperature_high")

    current_at = _dt(reading.get("recorded_at"))
    baseline_at = _dt(baseline.get("recorded_at"), current_at - timedelta(seconds=settings.trend_window_seconds)) if baseline else None
    baseline_is_recent = bool(baseline and current_at - baseline_at <= timedelta(seconds=settings.trend_window_seconds) and current_at >= baseline_at)
    if baseline_is_recent:
        old_hr = baseline.get("heart_rate_bpm")
        if hr is not None and old_hr is not None and abs(float(hr) - float(old_hr)) >= settings.hr_trend_change_bpm:
            warning.append(f"heart rate changed by {abs(float(hr) - float(old_hr)):.0f} bpm within the configured trend window")
            warning_codes.append("hr_trend")
        old_spo2 = baseline.get("spo2_percent")
        if spo2 is not None and old_spo2 is not None and float(old_spo2) - float(spo2) >= settings.spo2_trend_drop_percent:
            warning.append(f"SpO₂ decreased by {float(old_spo2) - float(spo2):.1f} percentage points within the configured trend window")
            warning_codes.append("spo2_trend_down")
        old_temp = baseline.get("temperature_c")
        if temp is not None and old_temp is not None and float(temp) - float(old_temp) >= settings.temp_trend_rise_c:
            warning.append(f"temperature rose by {float(temp) - float(old_temp):.1f} °C within the configured trend window")
            warning_codes.append("temperature_trend_up")

    if critical:
        return {"severity": "critical", "reasons": critical, "signature": "|".join(sorted(critical))}
    if warning:
        return {"severity": "warning", "reasons": warning, "signature": "|".join(sorted(set(warning_codes)))}
    if all(reading.get(key) is None for key in ("heart_rate_bpm", "spo2_percent", "temperature_c")):
        return {"severity": "unknown", "reasons": ["No vital measurements are available; normal status cannot be verified."], "signature": "missing_vitals"}
    return {"severity": "normal", "reasons": [], "signature": "normal"}

def detect_movement(reading: dict | None) -> bool | None:
    """Return True/False only when the packet actually provides movement evidence."""
    reading = reading or {}
    if reading.get("movement_detected") is not None:
        return bool(reading["movement_detected"])
    level = reading.get("movement_level")
    if level is not None:
        return float(level) >= settings.movement_threshold
    event = str(reading.get("motion_event") or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if event in {"normal_motion", "moving", "motion", "walking", "activity", "movement_detected"}:
        return True
    if event in {"still", "no_motion", "inactive", "stationary", "motionless", "no_movement"}:
        return False
    return None


def _readings_complete_for_green(reading: dict, movement: bool | None, vitals: dict) -> bool:
    return (
        movement is True
        and reading.get("signal_quality") == "good"
        and all(reading.get(key) is not None for key in ("heart_rate_bpm", "spo2_percent", "temperature_c"))
        and vitals["severity"] == "normal"
    )


def _event_id() -> str:
    return uuid.uuid4().hex


def _active_event(patient_id: str) -> dict | None:
    return db.emergency_events.find_one({"patient_id": patient_id, "status": "active"}, sort=[("updated_at", -1)])


def _append_timeline(event_id: str, kind: str, note: str, at: datetime | None = None) -> None:
    db.emergency_events.update_one(
        {"event_id": event_id},
        {"$push": {"timeline": {"at": at or utcnow(), "kind": kind, "note": note[:300]}}, "$set": {"updated_at": at or utcnow()}},
    )


def _create_event(patient_id: str, event_type: str, severity: str, reasons: list[str], reading: dict, *, response_timeout: bool = False) -> dict:
    now = utcnow()
    critical = severity == "critical"
    record = {
        "event_id": _event_id(),
        "patient_id": patient_id,
        "active_key": patient_id,
        "event_type": event_type,
        "trigger_events": [event_type],
        "inactivity_checkin_pending": event_type == "inactivity_checkin",
        "severity": severity,
        "state": "critical" if critical else "warning",
        "status": "active",
        "local_alarm_state": "active",
        "alarm_started_at": now,
        "alarm_until": now + timedelta(seconds=settings.local_alarm_active_seconds if critical else settings.warning_alarm_active_seconds),
        "snooze_until": None,
        "created_at": now,
        "updated_at": now,
        "acknowledged": False,
        "acknowledged_at": None,
        "acknowledged_by": None,
        "person_responded": False,
        "response_deadline": now + timedelta(seconds=settings.inactivity_response_timeout_seconds) if response_timeout else None,
        "resolved_at": None,
        "resolution_note": None,
        "reasons": reasons[:8],
        "latest_readings": _public_readings(reading),
        "notification_generation": 1 if critical else 0,
        "notification_status": "local_checkin_only" if event_type == "inactivity_checkin" else "pending_configuration_check",
        "timeline": [{"at": now, "kind": "created", "note": "; ".join(reasons)[:300] or event_type}],
        "clinical_validation": "prototype thresholds; not clinically validated",
    }
    try:
        db.emergency_events.insert_one(record)
    except DuplicateKeyError:
        # Concurrent readings may race to create the first active event. Reuse
        # the winner instead of creating duplicate alarms for one patient.
        existing = _active_event(patient_id)
        if not existing:
            raise
        if severity == "critical" and existing.get("severity") != "critical":
            return _promote_to_critical(existing, event_type, reasons, reading, "Concurrent critical input upgraded the active patient event.")
        return existing
    if critical or event_type in {"persistent_vital_abnormality", "sensor_data_missing"}:
        queue_event_notifications(record, critical=critical, generation=record["notification_generation"])
    elif event_type == "inactivity_checkin":
        _append_timeline(record["event_id"], "check_in_requested", "Local check-in requested; no medical conclusion is inferred.", now)
    return record


def _public_readings(reading: dict | None) -> dict:
    reading = reading or {}
    return {
        key: reading.get(key)
        for key in ("heart_rate_bpm", "spo2_percent", "temperature_c", "fall_detected", "movement_detected", "movement_level", "signal_quality", "recorded_at", "source")
        if reading.get(key) is not None
    }


def _worsened(previous: dict | None, current: dict | None) -> bool:
    """Acknowledge/snooze does not suppress a genuinely worsening measurement."""
    previous = previous or {}
    current = current or {}
    if current.get("fall_detected") is True:
        return True
    old_hr, new_hr = previous.get("heart_rate_bpm"), current.get("heart_rate_bpm")
    if new_hr is not None and new_hr < settings.critical_hr_low and (old_hr is None or old_hr >= settings.critical_hr_low):
        return True
    if new_hr is not None and new_hr > settings.critical_hr_high and (old_hr is None or old_hr <= settings.critical_hr_high):
        return True
    if old_hr is not None and new_hr is not None:
        if (old_hr < settings.warning_hr_low and new_hr <= old_hr - 5) or (old_hr > settings.warning_hr_high and new_hr >= old_hr + 5):
            return True
        if old_hr >= settings.warning_hr_low and old_hr <= settings.warning_hr_high and (new_hr < settings.critical_hr_low or new_hr > settings.critical_hr_high):
            return True
    old_spo2, new_spo2 = previous.get("spo2_percent"), current.get("spo2_percent")
    if new_spo2 is not None and new_spo2 < settings.critical_spo2_low and (old_spo2 is None or old_spo2 >= settings.critical_spo2_low):
        return True
    if old_spo2 is not None and new_spo2 is not None and (new_spo2 <= old_spo2 - 2 or (old_spo2 >= settings.critical_spo2_low and new_spo2 < settings.critical_spo2_low)):
        return True
    old_temp, new_temp = previous.get("temperature_c"), current.get("temperature_c")
    if new_temp is not None and new_temp >= settings.critical_temp_high_c and (old_temp is None or old_temp < settings.critical_temp_high_c):
        return True
    if old_temp is not None and new_temp is not None and (new_temp >= old_temp + 0.5 or (old_temp < settings.critical_temp_high_c <= new_temp)):
        return True
    return False


def _promote_to_critical(event: dict, event_type: str, reasons: list[str], reading: dict, note: str) -> dict:
    now = utcnow()
    generation = int(event.get("notification_generation", 0)) + 1
    combined_reasons = list(dict.fromkeys([*(event.get("reasons") or []), *reasons]))[:8]
    trigger_events = list(dict.fromkeys([*(event.get("trigger_events") or [event.get("event_type", "unknown")]), event_type]))
    db.emergency_events.update_one(
        {"event_id": event["event_id"], "status": "active"},
        {"$set": {
            "event_type": event_type,
            "severity": "critical",
            "state": "critical",
            "local_alarm_state": "active",
            "alarm_started_at": now,
            "alarm_until": now + timedelta(seconds=settings.local_alarm_active_seconds),
            "snooze_until": None,
            "response_deadline": None,
            "acknowledged": False,
            "acknowledged_at": None,
            "acknowledged_by": None,
            "person_responded": False,
            "reasons": combined_reasons,
            "trigger_events": trigger_events,
            "inactivity_checkin_pending": event_type in {"inactivity_checkin", "inactivity_no_response"},
            "latest_readings": _public_readings(reading),
            "notification_generation": generation,
            "updated_at": now,
        }, "$push": {"timeline": {"at": now, "kind": "escalated", "note": note[:300]}}},
    )
    cancel_event_jobs(event["event_id"], reason="event_escalated")
    refreshed = db.emergency_events.find_one({"event_id": event["event_id"]}) or event
    queue_event_notifications(refreshed, critical=True, generation=generation)
    return refreshed


def _update_existing(event: dict, reading: dict, vitals: dict, *, movement: bool | None = None) -> None:
    now = utcnow()
    updates: dict[str, Any] = {"updated_at": now, "latest_readings": _public_readings(reading)}
    timeline_entry = None
    if vitals.get("reasons"):
        updates["current_vital_reasons"] = vitals["reasons"][:8]
    else:
        updates["current_vital_reasons"] = []
    if movement is True:
        updates["last_movement_at"] = now
    if event.get("event_type") == "sensor_data_missing" and not event.get("sensor_recovered_at"):
        updates["sensor_recovered_at"] = now
        timeline_entry = {"at": now, "kind": "sensor_data_resumed", "note": "A new sensor packet arrived; the sensor-loss event remains open until reviewed."}
    elif event.get("event_type") == "persistent_vital_abnormality" and vitals.get("severity") == "normal" and not event.get("condition_cleared_at"):
        updates["condition_cleared_at"] = now
        timeline_entry = {"at": now, "kind": "vitals_back_in_demo_range", "note": "The latest available values are back within configured prototype ranges; the event remains open until reviewed."}
    elif event.get("event_type") == "inactivity_checkin" and movement is True and not event.get("person_responded") and not event.get("movement_resumed_at"):
        updates["movement_resumed_at"] = now
        timeline_entry = {"at": now, "kind": "movement_resumed", "note": "Movement was detected again, but a person check-in or caregiver review is still required."}
    update = {"$set": updates}
    if timeline_entry:
        update["$push"] = {"timeline": timeline_entry}
    db.emergency_events.update_one({"event_id": event["event_id"], "status": "active"}, update)


def _handle_condition(patient_id: str, state: dict, reading: dict, previous_reading: dict | None = None, vitals_override: dict | None = None) -> None:
    now = utcnow()
    vitals = vitals_override or classify_vitals(reading)
    movement = detect_movement(reading)
    fall = reading.get("fall_detected") is True
    active = _active_event(patient_id)

    if fall:
        if not active:
            _create_event(patient_id, "fall_detected", "critical", ["Existing fall-detection input reported a fall event."], reading)
            return
        if active.get("state") in {"snoozed", "acknowledged"} or active.get("severity") != "critical":
            _promote_to_critical(active, "fall_detected", ["Existing fall-detection input reported a fall event."], reading, "Fall event received; critical alarm reactivated regardless of snooze.")
            return
        _update_existing(active, reading, vitals, movement=movement)
        return

    if vitals["severity"] == "critical":
        if not active:
            _create_event(patient_id, "critical_vitals", "critical", vitals["reasons"], reading)
            return
        if active.get("severity") != "critical":
            _promote_to_critical(active, "critical_vitals", vitals["reasons"], reading, "Vital readings reached the configured critical test range.")
            return
        if active.get("state") in {"snoozed", "acknowledged"} and _worsened(previous_reading, reading):
            _promote_to_critical(active, "critical_vitals", vitals["reasons"], reading, "New or worsening critical readings reactivated the alert.")
            return
        _update_existing(active, reading, vitals, movement=movement)
        return

    if active:
        _update_existing(active, reading, vitals, movement=movement)
        return

    abnormal_since = state.get("abnormal_since")
    if vitals["severity"] == "warning" and abnormal_since and now - _dt(abnormal_since) >= timedelta(seconds=settings.vital_persistence_seconds):
        _create_event(patient_id, "persistent_vital_abnormality", "warning", vitals["reasons"], reading)


def on_reading(patient_id: str, reading: dict) -> None:
    """Persist monitoring context and evaluate a new reading without blocking on SMS/calls."""
    now = utcnow()
    reading = dict(reading)
    recorded_at = _dt(reading.get("recorded_at"), now)
    movement = detect_movement(reading)
    existing = db.monitoring_state.find_one({"patient_id": patient_id}) or {}
    previous_reading = existing.get("latest_reading") or {}
    baseline = db.vitals.find_one(
        {"patient_id": patient_id, "recorded_at": {"$gte": recorded_at - timedelta(seconds=settings.trend_window_seconds), "$lt": recorded_at}},
        sort=[("recorded_at", 1)],
    ) or previous_reading
    vitals = classify_vitals(reading, baseline)

    last_movement_at = existing.get("last_movement_at")
    still_since = existing.get("still_since")
    movement_for_state = movement if movement is not None else existing.get("movement_detected")
    movement_known = movement is not None or bool(existing.get("movement_known", False))
    if movement is True:
        last_movement_at = now
        still_since = None
    elif movement is False:
        still_since = _dt(still_since, now) if still_since else now
    # A vital-only packet preserves the last explicit movement state but never
    # counts as new movement and never establishes a verified-green state.

    old_abnormal_since = existing.get("abnormal_since")
    old_signature = existing.get("abnormal_signature")
    if vitals["severity"] == "warning":
        abnormal_since = _dt(old_abnormal_since, now) if old_signature == vitals["signature"] and old_abnormal_since else now
        abnormal_signature = vitals["signature"]
    else:
        abnormal_since = None
        abnormal_signature = None

    state_record = {
        "patient_id": patient_id,
        "latest_reading": reading,
        "previous_reading": previous_reading,
        "last_reading_at": recorded_at,
        "last_reading_received_at": now,
        "movement_known": movement_known,
        "movement_detected": movement_for_state,
        "last_movement_at": last_movement_at,
        "still_since": still_since,
        "vital_severity": vitals["severity"],
        "vital_reasons": vitals["reasons"],
        "abnormal_since": abnormal_since,
        "abnormal_signature": abnormal_signature,
        "normal_verified": _readings_complete_for_green(reading, movement, vitals),
        "updated_at": now,
    }
    db.monitoring_state.update_one({"patient_id": patient_id}, {"$set": state_record}, upsert=True)
    _handle_condition(patient_id, state_record, reading, previous_reading, vitals_override=vitals)


def _process_inactivity(state: dict, now: datetime) -> None:
    patient_id = state.get("patient_id")
    if not patient_id or not state.get("still_since"):
        return
    if state.get("movement_detected") is not False:
        return
    if now - _dt(state["still_since"]) < timedelta(seconds=settings.inactivity_threshold_seconds):
        return

    reading = state.get("latest_reading") or {}
    vitals = classify_vitals(reading)
    active = _active_event(patient_id)
    if vitals["severity"] == "critical":
        if not active:
            _create_event(patient_id, "inactivity_with_critical_vitals", "critical", ["Prolonged inactivity coincided with readings in the configured critical test range.", *vitals["reasons"]], reading)
        elif active.get("severity") != "critical":
            _promote_to_critical(active, "inactivity_with_critical_vitals", ["Prolonged inactivity coincided with readings in the configured critical test range.", *vitals["reasons"]], reading, "Inactivity plus critical readings triggered immediate escalation.")
        return

    if not active:
        reasons = ["Movement has remained below the configured threshold for the inactivity interval."]
        if vitals["severity"] == "normal":
            reasons.append("Available vital readings are within configured prototype ranges; a person check-in is still required.")
        else:
            reasons.append("Vital readings are missing or not verified normal; no assumption of safety is made.")
        _create_event(patient_id, "inactivity_checkin", "warning", reasons, reading, response_timeout=True)
        return

    # A stale sensor means movement can no longer be trusted, so do not infer inactivity from it.
    if active.get("event_type") == "sensor_data_missing" or active.get("person_responded"):
        return
    # If a different warning is already open, attach the inactivity check-in to
    # that event rather than creating a second active event for the same patient.
    if active.get("severity") == "warning" and not active.get("inactivity_checkin_pending") and active.get("event_type") != "inactivity_checkin":
        now = utcnow()
        combined = list(dict.fromkeys([*(active.get("reasons") or []), "Movement has remained below the configured threshold; a person check-in is required."]))[:8]
        db.emergency_events.update_one({"event_id": active["event_id"], "status": "active"}, {"$set": {
            "inactivity_checkin_pending": True,
            "response_deadline": now + timedelta(seconds=settings.inactivity_response_timeout_seconds),
            "state": "warning", "local_alarm_state": "active", "alarm_started_at": now,
            "alarm_until": now + timedelta(seconds=settings.warning_alarm_active_seconds),
            "acknowledged": False, "acknowledged_at": None, "acknowledged_by": None,
            "reasons": combined, "updated_at": now,
            "trigger_events": list(dict.fromkeys([*(active.get("trigger_events") or [active.get("event_type")]), "inactivity_checkin"])),
        }, "$push": {"timeline": {"at": now, "kind": "inactivity_checkin_requested", "note": "Prolonged inactivity was added to the existing warning; a person response is required before the timeout."}}})
        return

    # A pending inactivity check-in must escalate if no response arrives in time.
    if (active.get("event_type") == "inactivity_checkin" or active.get("inactivity_checkin_pending")) and not active.get("person_responded"):
        deadline = active.get("response_deadline")
        if deadline and now >= _dt(deadline):
            _promote_to_critical(active, "inactivity_no_response", ["The monitored person did not respond before the configured check-in deadline.", *vitals.get("reasons", [])], reading, "No response to inactivity check-in; escalation timeout elapsed.")


def _advance_alarm_cycle(event: dict, now: datetime) -> None:
    if event.get("status") != "active":
        return
    state = event.get("state")
    alarm_state = event.get("local_alarm_state")
    if state == "critical" and alarm_state == "active" and event.get("alarm_until") and now >= _dt(event["alarm_until"]):
        until = now + timedelta(seconds=settings.local_alarm_snooze_seconds)
        db.emergency_events.update_one({"event_id": event["event_id"], "status": "active"}, {"$set": {
            "state": "snoozed", "local_alarm_state": "snoozed", "snooze_until": until, "updated_at": now,
        }, "$push": {"timeline": {"at": now, "kind": "local_alarm_snoozed", "note": "Local sound snoozed; sensor monitoring and escalation continue."}}})
        return
    if state == "snoozed" and event.get("snooze_until") and now >= _dt(event["snooze_until"]):
        # Only the local sound cycle restarts here. Notification retries keep their
        # own independent schedule and attempt counts; a new/worsening event uses
        # _promote_to_critical() to start a new notification generation.
        db.emergency_events.update_one({"event_id": event["event_id"], "status": "active"}, {"$set": {
            "state": "critical", "local_alarm_state": "active", "alarm_started_at": now,
            "alarm_until": now + timedelta(seconds=settings.local_alarm_active_seconds), "snooze_until": None,
            "updated_at": now,
        }, "$push": {"timeline": {"at": now, "kind": "alarm_reactivated", "note": "Critical event remains unresolved after snooze; local alarm reactivated. Notification retries continue on their separate schedule."}}})
        return
    if state == "warning" and alarm_state == "active" and event.get("alarm_until") and now >= _dt(event["alarm_until"]):
        db.emergency_events.update_one({"event_id": event["event_id"], "status": "active"}, {"$set": {"local_alarm_state": "silent", "updated_at": now}, "$push": {"timeline": {"at": now, "kind": "warning_sound_ended", "note": "Warning sound ended; unresolved warning remains visible and monitoring continues."}}})


def monitor_once() -> None:
    """Run one non-blocking scheduler cycle; safe to call in tests or from lifespan."""
    now = utcnow()
    for state in db.monitoring_state.find({}):
        try:
            reading = state.get("latest_reading") or {}
            vitals = {"severity": state.get("vital_severity", "unknown"), "reasons": state.get("vital_reasons", []), "signature": state.get("abnormal_signature", "")}
            patient_id = state.get("patient_id")
            active = _active_event(patient_id) if patient_id else None
            last_received = state.get("last_reading_received_at")
            if patient_id and last_received and now - _dt(last_received) >= timedelta(seconds=settings.sensor_stale_after_seconds) and not active:
                _create_event(patient_id, "sensor_data_missing", "warning", [f"No new sensor packet has arrived for at least {settings.sensor_stale_after_seconds} seconds; normal status cannot be verified."], reading)
                active = _active_event(patient_id)
            # Persistently abnormal latest values can create a warning even if no new packet arrives.
            abnormal_since = state.get("abnormal_since")
            if not active and vitals["severity"] == "warning" and abnormal_since and now - _dt(abnormal_since) >= timedelta(seconds=settings.vital_persistence_seconds):
                _create_event(patient_id, "persistent_vital_abnormality", "warning", vitals["reasons"], reading)
            _process_inactivity(state, now)
            active = _active_event(patient_id) if patient_id else None
            if active:
                _advance_alarm_cycle(active, now)
        except Exception as exc:  # don't leak readings, contacts or provider details to logs
            logger.error("Emergency monitor cycle failed (%s)", type(exc).__name__)
    try:
        dispatch_due_jobs()
    except Exception as exc:
        logger.error("Notification dispatch cycle failed (%s)", type(exc).__name__)


async def emergency_monitor_loop() -> None:
    while True:
        await asyncio.sleep(max(1, settings.emergency_monitor_interval_seconds))
        await asyncio.to_thread(monitor_once)


def emergency_status(patient_id: str) -> dict:
    state = db.monitoring_state.find_one({"patient_id": patient_id}) or {}
    active_events = list(db.emergency_events.find({"patient_id": patient_id, "status": "active"}).sort("created_at", -1).limit(10))
    latest = state.get("latest_reading") or {}
    vital_class = classify_vitals(latest)
    if active_events:
        primary = active_events[0]
        display_status = "critical" if primary.get("severity") == "critical" else "warning"
        if primary.get("state") == "acknowledged":
            display_status = "acknowledged_unresolved"
    elif state.get("last_reading_received_at") and utcnow() - _dt(state.get("last_reading_received_at")) >= timedelta(seconds=settings.sensor_stale_after_seconds):
        display_status = "sensor_data_stale"
    elif state.get("normal_verified") is True and vital_class["severity"] == "normal":
        display_status = "normal_verified"
    elif state:
        display_status = "monitoring_unverified"
    else:
        display_status = "no_sensor_data"

    return {
        "patient_id": patient_id,
        "monitoring_status": display_status,
        "active_events": [_public_event(event) for event in active_events],
        "latest_monitoring": {
            "last_reading_at": state.get("last_reading_at"),
            "last_reading_received_at": state.get("last_reading_received_at"),
            "last_movement_at": state.get("last_movement_at"),
            "still_since": state.get("still_since"),
            "movement_known": state.get("movement_known", False),
            "movement_detected": state.get("movement_detected"),
            "vital_severity": state.get("vital_severity", "unknown"),
            "vital_reasons": state.get("vital_reasons", []),
            "normal_verified": state.get("normal_verified", False),
            "signal_quality": latest.get("signal_quality", "unknown"),
        },
        "configuration": {
            "inactivity_threshold_seconds": settings.inactivity_threshold_seconds,
            "sensor_stale_after_seconds": settings.sensor_stale_after_seconds,
            "trend_window_seconds": settings.trend_window_seconds,
            "inactivity_response_timeout_seconds": settings.inactivity_response_timeout_seconds,
            "local_alarm_active_seconds": settings.local_alarm_active_seconds,
            "local_alarm_snooze_seconds": settings.local_alarm_snooze_seconds,
            "warning_alarm_active_seconds": settings.warning_alarm_active_seconds,
            "vital_persistence_seconds": settings.vital_persistence_seconds,
            "twilio_enabled": settings.twilio_enabled,
            "twilio_credentials_configured": bool(settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_from_number),
            "sms_enabled": settings.twilio_enabled and settings.emergency_sms_enabled,
            "calls_enabled": settings.twilio_enabled and settings.emergency_calls_enabled,
            "contact_encryption_configured": bool(settings.emergency_contact_encryption_key),
        },
    }


def _public_event(event: dict) -> dict:
    keys = (
        "event_id", "patient_id", "event_type", "severity", "state", "status", "local_alarm_state",
        "alarm_started_at", "alarm_until", "snooze_until", "created_at", "updated_at", "acknowledged",
        "acknowledged_at", "acknowledged_by", "person_responded", "response_deadline", "resolved_at",
        "resolution_note", "reasons", "trigger_events", "inactivity_checkin_pending", "latest_readings", "notification_generation", "notification_status",
        "timeline", "clinical_validation", "current_vital_reasons", "sensor_recovered_at", "condition_cleared_at", "movement_resumed_at",
    )
    return {key: event.get(key) for key in keys if key in event}


def acknowledge_event(patient_id: str, event_id: str, actor: str, *, person_response: bool = False) -> dict | None:
    now = utcnow()
    event = db.emergency_events.find_one({"patient_id": patient_id, "event_id": event_id, "status": "active"})
    if not event:
        return None
    if person_response:
        critical_event = event.get("severity") == "critical"
        db.emergency_events.update_one({"event_id": event_id}, {"$set": {
            "person_responded": True, "person_responded_at": now,
            "state": "snoozed" if critical_event else "acknowledged",
            "local_alarm_state": "snoozed" if critical_event else "silent",
            "snooze_until": now + timedelta(seconds=settings.local_alarm_snooze_seconds) if critical_event else None,
            "acknowledged": True, "acknowledged_at": now,
            "acknowledged_by": "monitored_person", "updated_at": now,
        }, "$push": {"timeline": {"at": now, "kind": "person_response", "note": "The monitored person indicated they are responsive; this does not prove the emergency is resolved. Critical local sound is snoozed, while notification retries and monitoring continue."}}})
    else:
        db.emergency_events.update_one({"event_id": event_id}, {"$set": {
            "state": "acknowledged", "local_alarm_state": "silent", "acknowledged": True,
            "acknowledged_at": now, "acknowledged_by": actor, "updated_at": now,
        }, "$push": {"timeline": {"at": now, "kind": "caregiver_acknowledged", "note": "An authorized caregiver acknowledged the alert; monitoring continues."}}})
    if not person_response:
        cancel_event_jobs(event_id, reason="event_acknowledged")
    return _public_event(db.emergency_events.find_one({"event_id": event_id}) or event)


def resolve_event(patient_id: str, event_id: str, actor: str, note: str) -> dict | None:
    event = db.emergency_events.find_one({"patient_id": patient_id, "event_id": event_id, "status": "active"})
    if not event:
        return None
    state = db.monitoring_state.find_one({"patient_id": patient_id}) or {}
    current_vitals = classify_vitals(state.get("latest_reading") or {})
    if current_vitals["severity"] == "critical":
        raise ValueError("Latest readings remain in a configured critical test range; do not resolve automatically. Verify the person and review the readings first.")
    now = utcnow()
    db.emergency_events.update_one({"event_id": event_id, "status": "active"}, {"$set": {
        "status": "resolved", "state": "resolved", "local_alarm_state": "silent", "resolved_at": now,
        "resolved_by": actor, "resolution_note": note[:300], "updated_at": now,
    }, "$push": {"timeline": {"at": now, "kind": "resolved", "note": f"Authorized caregiver confirmed safety: {note[:250]}"}}})
    cancel_event_jobs(event_id, reason="event_resolved")
    return _public_event(db.emergency_events.find_one({"event_id": event_id}) or event)

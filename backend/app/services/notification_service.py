"""Durable, optional Twilio SMS/voice dispatch. No credentials or phone numbers are logged."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape

from pymongo.errors import DuplicateKeyError

from ..config import settings
from ..database import db
from .contact_crypto import decrypt_phone

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _provider_ready() -> bool:
    return bool(settings.twilio_enabled and settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_from_number)


def _message_for(event: dict) -> str:
    readings = event.get("latest_readings") or {}
    pieces = []
    if readings.get("heart_rate_bpm") is not None:
        pieces.append(f"heart rate {readings['heart_rate_bpm']} beats per minute")
    if readings.get("spo2_percent") is not None:
        pieces.append(f"oxygen saturation {readings['spo2_percent']} percent")
    if readings.get("temperature_c") is not None:
        pieces.append(f"temperature {readings['temperature_c']} Celsius")
    reading_text = "; available readings: " + ", ".join(pieces) if pieces else "; no reliable vital readings are available"
    timestamp = event.get("created_at") or _now()
    return (
        f"Lifora {event.get('severity', 'warning')} alert. Event: {event.get('event_type', 'health monitoring alert')}. "
        f"Detected at {timestamp.isoformat()}{reading_text}. Please check the monitored person and follow your established emergency plan. "
        "This automated message is not a diagnosis."
    )[:1200]


def queue_event_notifications(event: dict, *, critical: bool, generation: int) -> None:
    """Create idempotent jobs. Actual network calls happen only in the background worker."""
    event_id = event.get("event_id")
    patient_id = event.get("patient_id")
    if not event_id or not patient_id:
        return
    contacts = list(db.emergency_contacts.find({"patient_id": patient_id}).sort("priority", 1).limit(3))
    if not contacts:
        db.emergency_events.update_one({"event_id": event_id}, {"$set": {"notification_status": "no_emergency_contacts"}})
        return
    if not _provider_ready():
        db.emergency_events.update_one({"event_id": event_id}, {"$set": {"notification_status": "provider_not_configured"}})
        return

    sms_enabled = settings.emergency_sms_enabled
    calls_enabled = settings.emergency_calls_enabled
    created_any = False
    for contact in contacts:
        contact_id = contact.get("contact_id", str(contact.get("_id", "contact")))
        priority = int(contact.get("priority", 99))
        channels = []
        if sms_enabled:
            channels.append("sms")
        # Initial automated call goes to the primary contact. Secondary calls are
        # queued by the worker if primary call retries are exhausted.
        if critical and calls_enabled and priority == 1:
            channels.append("call")
        if not critical and priority > 1:
            continue
        for channel in channels:
            key = f"{event_id}:{generation}:{channel}:{contact_id}"
            job = {
                "job_id": uuid.uuid4().hex,
                "idempotency_key": key,
                "event_id": event_id,
                "patient_id": patient_id,
                "contact_id": contact_id,
                "contact_priority": priority,
                "channel": channel,
                "severity": "critical" if critical else "warning",
                "generation": generation,
                "status": "pending",
                "attempts": 0,
                "created_at": _now(),
                "next_attempt_at": _now(),
                "last_error_code": None,
            }
            try:
                db.notification_jobs.insert_one(job)
                created_any = True
            except DuplicateKeyError:
                created_any = True
    db.emergency_events.update_one({"event_id": event_id}, {"$set": {"notification_status": "queued" if created_any else "no_channels_enabled"}})


def cancel_event_jobs(event_id: str, *, reason: str) -> None:
    db.notification_jobs.update_many(
        {"event_id": event_id, "status": {"$in": ["pending", "sent", "failed"]}},
        {"$set": {"status": "cancelled", "cancel_reason": reason, "updated_at": _now()}},
    )


def _send(job: dict) -> None:
    from twilio.rest import Client

    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    if job.get("channel") == "sms":
        client.messages.create(from_=settings.twilio_from_number, to=job["to_number"], body=job["message"][:1500])
        return
    if job.get("channel") == "call":
        voice_text = escape(job["message"])
        twiml = f"<Response><Say>{voice_text}</Say></Response>"
        client.calls.create(from_=settings.twilio_from_number, to=job["to_number"], twiml=twiml)
        return
    raise ValueError("Unsupported notification channel")


def _secondary_call_after_primary_exhausted(job: dict) -> None:
    if job.get("channel") != "call" or job.get("contact_priority") != 1 or job.get("severity") != "critical":
        return
    event = db.emergency_events.find_one({"event_id": job.get("event_id"), "status": "active"})
    if not event or (event.get("state") == "acknowledged" and event.get("acknowledged_by") != "monitored_person"):
        return
    secondary_contacts = list(db.emergency_contacts.find({"patient_id": job.get("patient_id"), "priority": {"$gt": 1}}).sort("priority", 1).limit(2))
    for contact in secondary_contacts:
        contact_id = contact.get("contact_id", str(contact.get("_id", "contact")))
        key = f"{job.get('event_id')}:{job.get('generation')}:call:{contact_id}:secondary"
        try:
            db.notification_jobs.insert_one({
                "job_id": uuid.uuid4().hex, "idempotency_key": key, "event_id": job.get("event_id"),
                "patient_id": job.get("patient_id"), "contact_id": contact_id,
                "contact_priority": int(contact.get("priority", 99)), "channel": "call",
                "severity": "critical", "generation": job.get("generation"), "status": "pending", "attempts": 0,
                "created_at": _now(), "next_attempt_at": _now(), "last_error_code": None,
            })
        except DuplicateKeyError:
            pass


def _record_attempt(event_id: str, channel: str, attempt: int, *, accepted: bool, error_code: str | None = None) -> None:
    now = _now()
    if accepted:
        note = f"{channel.upper()} attempt {attempt}: request accepted by provider; final carrier delivery is not guaranteed."
        status = "provider_accepted"
    else:
        note = f"{channel.upper()} attempt {attempt} failed ({error_code or 'provider_error'}); retry policy remains active if attempts remain."
        status = "delivery_retrying_or_failed"
    db.emergency_events.update_one(
        {"event_id": event_id},
        {"$set": {"notification_status": status, "updated_at": now}, "$push": {"timeline": {"at": now, "kind": "notification_attempt", "note": note}}},
    )


def dispatch_due_jobs() -> None:
    """Send due jobs and schedule bounded retries; errors never contain recipient details in logs."""
    now = _now()
    if not _provider_ready():
        return
    due = db.notification_jobs.find({"status": {"$in": ["pending", "sent", "failed"]}, "next_attempt_at": {"$lte": now}}).sort("created_at", 1).limit(10)
    for job in due:
        event = db.emergency_events.find_one({"event_id": job.get("event_id")})
        if not event or event.get("status") != "active" or (event.get("state") == "acknowledged" and event.get("acknowledged_by") != "monitored_person"):
            db.notification_jobs.update_one({"job_id": job["job_id"]}, {"$set": {"status": "cancelled", "updated_at": now}})
            continue
        channel = job.get("channel")
        retry_seconds = settings.call_retry_seconds if channel == "call" else settings.sms_retry_seconds
        max_attempts = settings.max_call_attempts if channel == "call" else settings.max_sms_attempts
        attempts = int(job.get("attempts", 0))
        if attempts >= max_attempts:
            db.notification_jobs.update_one({"job_id": job["job_id"]}, {"$set": {"status": "exhausted", "updated_at": now}})
            _secondary_call_after_primary_exhausted(job)
            continue
        try:
            contact = db.emergency_contacts.find_one({"contact_id": job.get("contact_id"), "patient_id": job.get("patient_id")})
            if not contact or not contact.get("phone_encrypted"):
                raise RuntimeError("Emergency contact is unavailable")
            recipient = decrypt_phone(contact["phone_encrypted"])
            _send({**job, "to_number": recipient, "message": _message_for(event)})
            attempts += 1
            status = "exhausted" if attempts >= max_attempts else "sent"
            next_at = now + timedelta(seconds=retry_seconds) if status == "sent" else None
            db.notification_jobs.update_one({"job_id": job["job_id"]}, {"$set": {
                "status": status, "attempts": attempts, "last_attempt_at": now, "next_attempt_at": next_at,
                "last_error_code": None, "updated_at": now,
            }})
            _record_attempt(job.get("event_id"), channel, attempts, accepted=True)
            if status == "exhausted":
                _secondary_call_after_primary_exhausted({**job, "attempts": attempts, "status": status})
        except Exception as exc:
            attempts += 1
            status = "exhausted" if attempts >= max_attempts else "failed"
            next_at = now + timedelta(seconds=retry_seconds) if status == "failed" else None
            db.notification_jobs.update_one({"job_id": job["job_id"]}, {"$set": {
                "status": status, "attempts": attempts, "last_attempt_at": now, "next_attempt_at": next_at,
                "last_error_code": type(exc).__name__, "updated_at": now,
            }})
            _record_attempt(job.get("event_id"), channel, attempts, accepted=False, error_code=type(exc).__name__)
            logger.warning("Notification dispatch failed (channel=%s, error=%s)", channel, type(exc).__name__)
            if status == "exhausted":
                _secondary_call_after_primary_exhausted({**job, "attempts": attempts, "status": status})

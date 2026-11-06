from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import secrets
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pymongo import DESCENDING
from pymongo.errors import DuplicateKeyError

from .auth import create_access_token, verify_password
from .config import settings
from .database import close_database, db, init_database
from .dependencies import get_current_user, require_patient_access
from .schemas import (
    DeviceIngestRequest,
    EmergencyContactsUpdate,
    EmergencyResolveRequest,
    ExplainRequest,
    HistoryCreate,
    LoginRequest,
    MedicationCreate,
    MedicationEventCreate,
    PatientCreate,
    PersonResponseRequest,
    SymptomCreate,
    VitalCreate,
)
from .serialization import serialize_document
from .services.alerts import build_rule_alerts
from .services.contact_crypto import decrypt_phone, decrypt_value, encrypt_phone, encrypt_value
from .services.emergency import acknowledge_event, emergency_monitor_loop, emergency_status, on_reading, resolve_event
from .services.ml_service import model_status
from .services.rag_service import answer_question, index_report, rag_status
from .services.seed import seed_demo_data

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_database()
    seed_demo_data()
    monitor_task = asyncio.create_task(emergency_monitor_loop(), name="lifora-emergency-monitor")
    try:
        yield
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        close_database()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "Lifora final-year project prototype: caregiver dashboard, MongoDB-backed records, "
        "local patient-report retrieval, separate trusted-reference retrieval, and optional Groq explanations. "
        "Not a medical device or diagnostic system."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Device-Token"],
)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_patient_or_404(patient_id: str, user: dict) -> dict:
    require_patient_access(patient_id, user)
    patient = db.patients.find_one({"patient_id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


def store_vital(patient_id: str, reading: dict, source_override: str | None = None) -> dict:
    record = dict(reading)
    record["patient_id"] = patient_id
    record["recorded_at"] = record.get("recorded_at") or now_utc()
    if source_override:
        record["source"] = source_override
    record.setdefault("source", "caregiver_manual")
    record.setdefault("fall_detected", False)
    patient_record = db.patients.find_one({"patient_id": patient_id}, {"synthetic": 1}) or {}
    record["synthetic"] = bool(patient_record.get("synthetic")) or record.get("source") in {"simulator", "synthetic_demo_seed"}
    inserted = db.vitals.insert_one(record)
    record["id"] = str(inserted.inserted_id)

    alerts = build_rule_alerts(patient_id, record)
    if alerts:
        db.alerts.insert_many(alerts)

    try:
        # Additive emergency layer consumes the existing fall flag and available sensor data.
        # The ML model itself is not loaded, retrained, or modified here.
        on_reading(patient_id, record)
        emergency_processing_status = "processed"
    except Exception as exc:
        # Keep the raw reading stored, but persist a sanitized failure marker so it is not silent.
        emergency_processing_status = "failed"
        logger.error("Emergency event processing failed (%s)", type(exc).__name__)
    record["emergency_processing_status"] = emergency_processing_status
    db.vitals.update_one({"_id": inserted.inserted_id}, {"$set": {"emergency_processing_status": emergency_processing_status}})

    if record.get("medicine_box_open") is True:
        db.medication_events.insert_one({
            "patient_id": patient_id,
            "event": "box_opened",
            "notes": "Recorded by a device reading; does not prove a dose was taken.",
            "recorded_at": record["recorded_at"],
            "source": record.get("source", "unknown"),
        })
    return record


def public_report(document: dict) -> dict:
    result = serialize_document(document)
    if result is not None:
        result.pop("local_path", None)
        result.pop("index_output_tail", None)
    return result


def safe_filename(filename: str | None) -> str:
    base = Path(filename or "report.pdf").name
    base = re.sub(r"[^A-Za-z0-9._-]", "_", base)[:120]
    if not base.lower().endswith(".pdf"):
        base += ".pdf"
    return base


@app.get("/api/v1/health")
def health_check():
    return {"status": "ok", "service": "lifora-api", "timestamp": now_utc().isoformat()}


@app.post("/api/v1/auth/login")
def login(payload: LoginRequest):
    user = db.users.find_one({"username": payload.username, "is_active": True})
    if not user or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    token = create_access_token(user["username"])
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "username": user["username"],
            "role": user.get("role", "caregiver"),
            "patient_ids": user.get("patient_ids", []),
        },
    }


@app.get("/api/v1/auth/me")
def who_am_i(user: dict = Depends(get_current_user)):
    return {
        "username": user["username"],
        "role": user.get("role", "caregiver"),
        "patient_ids": user.get("patient_ids", []),
    }


@app.get("/api/v1/patients")
def list_patients(user: dict = Depends(get_current_user)):
    query = {} if user.get("role") == "admin" else {"patient_id": {"$in": user.get("patient_ids", [])}}
    return [serialize_document(x) for x in db.patients.find(query).sort("created_at", DESCENDING)]


@app.post("/api/v1/patients", status_code=201)
def create_patient(payload: PatientCreate, user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Only an admin can create patient profiles")
    record = {
        "patient_id": payload.patient_id,
        "display_name": payload.display_name,
        "synthetic": payload.synthetic,
        "created_at": now_utc(),
        "status": "active",
    }
    try:
        db.patients.insert_one(record)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="Patient ID already exists") from exc
    return serialize_document(record)


@app.get("/api/v1/patients/{patient_id}")
def get_patient(patient_id: str, user: dict = Depends(get_current_user)):
    patient = get_patient_or_404(patient_id, user)
    return serialize_document(patient)


@app.get("/api/v1/patients/{patient_id}/dashboard")
def dashboard_summary(patient_id: str, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    latest = db.vitals.find_one({"patient_id": patient_id}, sort=[("recorded_at", DESCENDING)])
    recent_alerts = list(db.alerts.find({"patient_id": patient_id, "acknowledged": False}).sort("created_at", DESCENDING).limit(8))
    recent_symptoms = list(db.symptoms.find({"patient_id": patient_id}).sort("onset_at", DESCENDING).limit(5))
    return {
        "patient_id": patient_id,
        "latest_vitals": serialize_document(latest),
        "unacknowledged_alert_count": db.alerts.count_documents({"patient_id": patient_id, "acknowledged": False}),
        "symptom_count": db.symptoms.count_documents({"patient_id": patient_id}),
        "report_count": db.reports.count_documents({"patient_id": patient_id}),
        "history_count": db.history.count_documents({"patient_id": patient_id}),
        "recent_alerts": [serialize_document(x) for x in recent_alerts],
        "recent_symptoms": [serialize_document(x) for x in recent_symptoms],
    }


@app.get("/api/v1/patients/{patient_id}/vitals")
def list_vitals(patient_id: str, limit: int = Query(default=50, ge=1, le=250), user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    return [serialize_document(x) for x in db.vitals.find({"patient_id": patient_id}).sort("recorded_at", DESCENDING).limit(limit)]


@app.post("/api/v1/patients/{patient_id}/vitals", status_code=201)
def create_vital(patient_id: str, payload: VitalCreate, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    return serialize_document(store_vital(patient_id, payload.model_dump()))


@app.post("/api/v1/devices/ingest", status_code=201)
def ingest_device_reading(
    payload: DeviceIngestRequest,
    x_device_token: str | None = Header(default=None, alias="X-Device-Token"),
):
    expected = settings.device_ingest_token
    if not expected or not x_device_token or not secrets.compare_digest(x_device_token, expected):
        raise HTTPException(status_code=401, detail="Invalid device token")
    if not db.patients.find_one({"patient_id": payload.patient_id}):
        raise HTTPException(status_code=404, detail="Device patient is not registered")
    person_response_recorded = False
    if payload.person_response:
        checkin = db.emergency_events.find_one({
            "patient_id": payload.patient_id,
            "status": "active",
            "event_type": {"$in": ["inactivity_checkin", "inactivity_no_response", "persistent_vital_abnormality", "sensor_data_missing"]},
        }, sort=[("created_at", DESCENDING)])
        if checkin:
            acknowledge_event(payload.patient_id, checkin["event_id"], "esp32_person_button", person_response=True)
            person_response_recorded = True
        elif all(value is None for value in (payload.heart_rate_bpm, payload.spo2_percent, payload.temperature_c)) and not payload.fall_detected and payload.medicine_box_open is None and not payload.motion_event and payload.movement_detected is None and payload.movement_level is None:
            raise HTTPException(status_code=409, detail="No active inactivity check-in is waiting for a person response.")

    reading = payload.model_dump(exclude={"patient_id", "device_id", "person_response"})
    has_sensor_data = (
        any(reading.get(key) is not None for key in ("heart_rate_bpm", "spo2_percent", "temperature_c", "medicine_box_open", "motion_event", "movement_detected", "movement_level"))
        or reading.get("fall_detected") is True
    )
    if not has_sensor_data:
        return {"person_response": "recorded", "message": "Person check-in recorded; the emergency event remains unresolved until reviewed."}
    reading["device_id"] = payload.device_id
    reading["source"] = "esp32"
    result = store_vital(payload.patient_id, reading, source_override="esp32")
    response_status = "recorded" if person_response_recorded else ("no_active_checkin" if payload.person_response else None)
    return {"reading": serialize_document(result), "person_response": response_status, "message": "Reading accepted; emergency rules run independently of AI explanations."}


@app.get("/api/v1/devices/alert-state")
def device_alert_state(
    patient_id: str = Query(min_length=1, max_length=64),
    x_device_token: str | None = Header(default=None, alias="X-Device-Token"),
):
    expected = settings.device_ingest_token
    if not expected or not x_device_token or not secrets.compare_digest(x_device_token, expected):
        raise HTTPException(status_code=401, detail="Invalid device token")
    if not db.patients.find_one({"patient_id": patient_id}):
        raise HTTPException(status_code=404, detail="Device patient is not registered")
    result = emergency_status(patient_id)
    events = result["active_events"]
    primary = events[0] if events else None
    state = result["monitoring_status"]
    alarm_active = bool(primary and primary.get("local_alarm_state") == "active" and primary.get("state") in {"critical", "warning"})
    if primary and primary.get("severity") == "critical":
        led_color = "red"
        alarm_pattern = "critical" if alarm_active else "silent"
    elif primary:
        led_color = "amber"
        alarm_pattern = "warning" if alarm_active else "silent"
    elif state == "normal_verified":
        led_color = "green"
        alarm_pattern = "silent"
    else:
        # Missing or unreliable data is never represented as normal/green.
        led_color = "amber"
        alarm_pattern = "silent"
    return {
        "patient_id": patient_id,
        "monitoring_status": state,
        "led_color": led_color,
        "alarm_active": alarm_active,
        "alarm_pattern": alarm_pattern,
        "event_id": primary.get("event_id") if primary else None,
        "local_alarm_state": primary.get("local_alarm_state") if primary else "silent",
        "snooze_until": primary.get("snooze_until") if primary else None,
        "poll_interval_seconds": max(2, settings.emergency_monitor_interval_seconds),
        "note": "Adapter endpoint only; the existing ESP32 firmware must poll this endpoint and drive its own buzzer/LED pins.",
    }


@app.get("/api/v1/patients/{patient_id}/emergency")
def get_emergency_status(patient_id: str, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    return emergency_status(patient_id)


@app.get("/api/v1/patients/{patient_id}/emergency-events")
def list_emergency_events(
    patient_id: str,
    active_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    user: dict = Depends(get_current_user),
):
    get_patient_or_404(patient_id, user)
    query = {"patient_id": patient_id}
    if active_only:
        query["status"] = "active"
    rows = list(db.emergency_events.find(query).sort("created_at", DESCENDING).limit(limit))
    return [serialize_document(row) for row in rows]


@app.get("/api/v1/patients/{patient_id}/emergency-contacts")
def get_emergency_contacts(patient_id: str, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    try:
        rows = list(db.emergency_contacts.find({"patient_id": patient_id}).sort("priority", 1))
        return [{
            "contact_id": row.get("contact_id"),
            "name": decrypt_value(row["name_encrypted"]) if row.get("name_encrypted") else "",
            "relationship": decrypt_value(row["relationship_encrypted"]) if row.get("relationship_encrypted") else "caregiver",
            "priority": row.get("priority", 1),
            "phone": decrypt_phone(row["phone_encrypted"]) if row.get("phone_encrypted") else "",
        } for row in rows]
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.put("/api/v1/patients/{patient_id}/emergency-contacts")
def update_emergency_contacts(patient_id: str, payload: EmergencyContactsUpdate, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    try:
        prepared = []
        for contact in payload.contacts:
            item = contact.model_dump()
            phone = item.pop("phone")
            name = item.pop("name")
            relationship = item.pop("relationship")
            item.update({
                "patient_id": patient_id,
                "contact_id": uuid.uuid4().hex,
                "name_encrypted": encrypt_value(name),
                "relationship_encrypted": encrypt_value(relationship),
                "phone_encrypted": encrypt_phone(phone),
                "updated_at": now_utc(),
                "updated_by": user["username"],
            })
            prepared.append(item)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    db.emergency_contacts.delete_many({"patient_id": patient_id})
    if prepared:
        db.emergency_contacts.insert_many(prepared)
    return {"status": "saved", "contact_count": len(prepared), "contacts": [{
        "contact_id": item["contact_id"], "priority": item["priority"],
    } for item in prepared], "note": "Contact names, relationships and phone numbers are encrypted in MongoDB and are not returned by this write endpoint."}


@app.post("/api/v1/patients/{patient_id}/emergency-events/{event_id}/respond")
def person_responds_to_emergency(patient_id: str, event_id: str, payload: PersonResponseRequest, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    if not payload.person_is_responsive:
        raise HTTPException(status_code=422, detail="A response must explicitly confirm responsiveness.")
    active_event = db.emergency_events.find_one({"patient_id": patient_id, "event_id": event_id, "status": "active"})
    if not active_event:
        raise HTTPException(status_code=404, detail="Active emergency event not found")
    if active_event.get("event_type") not in {"inactivity_checkin", "inactivity_no_response", "persistent_vital_abnormality", "sensor_data_missing"}:
        raise HTTPException(status_code=409, detail="A person response can only acknowledge an inactivity or warning event.")
    result = acknowledge_event(patient_id, event_id, "monitored_person", person_response=True)
    if not result:
        raise HTTPException(status_code=404, detail="Active emergency event not found")
    return {"status": "acknowledged_not_resolved", "event": result, "note": "Responsiveness does not by itself prove that an emergency is resolved."}


@app.post("/api/v1/patients/{patient_id}/emergency-events/{event_id}/acknowledge")
def caregiver_acknowledges_emergency(patient_id: str, event_id: str, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    result = acknowledge_event(patient_id, event_id, user["username"])
    if not result:
        raise HTTPException(status_code=404, detail="Active emergency event not found")
    return {"status": "acknowledged_not_resolved", "event": result}


@app.post("/api/v1/patients/{patient_id}/emergency-events/{event_id}/resolve")
def resolve_emergency(patient_id: str, event_id: str, payload: EmergencyResolveRequest, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    if not payload.safety_confirmed:
        raise HTTPException(status_code=422, detail="Safety must be explicitly confirmed before resolving an event.")
    try:
        result = resolve_event(patient_id, event_id, user["username"], payload.notes)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result:
        raise HTTPException(status_code=404, detail="Active emergency event not found")
    return {"status": "resolved", "event": result}


@app.get("/api/v1/patients/{patient_id}/symptoms")
def list_symptoms(patient_id: str, limit: int = Query(default=100, ge=1, le=500), user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    return [serialize_document(x) for x in db.symptoms.find({"patient_id": patient_id}).sort("onset_at", DESCENDING).limit(limit)]


@app.post("/api/v1/patients/{patient_id}/symptoms", status_code=201)
def create_symptom(patient_id: str, payload: SymptomCreate, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    record = payload.model_dump()
    record.update({
        "patient_id": patient_id,
        "created_at": now_utc(),
        "source": "caregiver_reported",
        "verification_status": "unverified",
        "synthetic": db.patients.find_one({"patient_id": patient_id}).get("synthetic", False),
    })
    result = db.symptoms.insert_one(record)
    record["id"] = str(result.inserted_id)
    return serialize_document(record)


@app.get("/api/v1/patients/{patient_id}/history")
def list_history(patient_id: str, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    return [serialize_document(x) for x in db.history.find({"patient_id": patient_id}).sort("created_at", DESCENDING)]


@app.post("/api/v1/patients/{patient_id}/history", status_code=201)
def create_history(patient_id: str, payload: HistoryCreate, user: dict = Depends(get_current_user)):
    patient = get_patient_or_404(patient_id, user)
    record = payload.model_dump()
    if hasattr(record.get("reported_on"), "isoformat"):
        record["reported_on"] = record["reported_on"].isoformat()
    record.update({
        "patient_id": patient_id,
        "created_at": now_utc(),
        "source": "caregiver_entered",
        "verification_status": "unverified",
        "synthetic": patient.get("synthetic", False),
    })
    result = db.history.insert_one(record)
    record["id"] = str(result.inserted_id)
    return serialize_document(record)


@app.get("/api/v1/patients/{patient_id}/reports")
def list_reports(patient_id: str, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    return [public_report(x) for x in db.reports.find({"patient_id": patient_id}).sort("created_at", DESCENDING)]


@app.post("/api/v1/patients/{patient_id}/reports", status_code=201)
def upload_report(patient_id: str, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    patient = get_patient_or_404(patient_id, user)
    filename = safe_filename(file.filename)
    if not filename.lower().endswith(".pdf") or file.content_type not in {"application/pdf", "application/octet-stream", None}:
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported")

    content = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"PDF exceeds {settings.max_upload_mb} MB limit")
    if not content.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="Uploaded file does not have a PDF signature")

    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass or len(document) == 0:
                raise HTTPException(status_code=422, detail="Encrypted or empty PDFs are not supported")
            extracted_text = "\n".join(page.get_text("text", sort=True) for page in document)
            page_count = len(document)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="PDF could not be parsed") from exc

    if not extracted_text.strip():
        raise HTTPException(status_code=422, detail="No selectable text found. OCR for scanned PDFs is not enabled in this version.")

    if not settings.allow_real_report_uploads:
        if patient_id != "demo_patient" or "SYNTHETIC MEDICAL REPORT - TEST ONLY" not in extracted_text or "Patient: Demo Patient" not in extracted_text:
            raise HTTPException(
                status_code=403,
                detail="Synthetic-only upload mode is enabled. Set ALLOW_REAL_REPORT_UPLOADS=true only after local-storage, access-control, backup and privacy review.",
            )

    digest = hashlib.sha256(content).hexdigest()
    existing = db.reports.find_one({"patient_id": patient_id, "sha256": digest})
    if existing:
        if existing.get("index_status") == "indexed":
            return {"report": public_report(existing), "message": "This exact PDF was already uploaded and indexed; duplicate skipped."}
        existing_path = Path(existing.get("local_path", ""))
        if existing_path.is_file():
            try:
                retry_result = index_report(existing_path, patient_id)
                db.reports.update_one(
                    {"report_id": existing["report_id"]},
                    {"$set": {"index_status": "indexed", "indexed_at": now_utc(), "index_output_tail": retry_result.get("output", "")[-2500:]}, "$unset": {"index_error": ""}},
                )
                existing["index_status"] = "indexed"
                existing["indexed_at"] = now_utc()
                existing.pop("index_error", None)
                return {"report": public_report(existing), "message": "Previously uploaded report was re-indexed successfully."}
            except Exception as exc:
                db.reports.update_one({"report_id": existing["report_id"]}, {"$set": {"index_status": "failed", "index_error": str(exc)[:1800]}})
                existing["index_status"] = "failed"
                existing["index_error"] = str(exc)[:1800]
                return {"report": public_report(existing), "message": "Duplicate report found; re-indexing failed. Inspect index_error."}
        return {"report": public_report(existing), "message": "Duplicate report record exists, but its local file is missing."}

    upload_id = uuid.uuid4().hex
    destination_dir = settings.upload_dir / patient_id / upload_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    # Keep the original, sanitized basename so RAG citations remain readable.
    destination = destination_dir / filename
    destination.write_bytes(content)

    record = {
        "report_id": uuid.uuid4().hex,
        "patient_id": patient_id,
        "original_filename": filename,
        "local_path": str(destination),
        "sha256": digest,
        "page_count": page_count,
        "byte_size": len(content),
        "created_at": now_utc(),
        "index_status": "indexing",
        "synthetic": patient.get("synthetic", False),
    }
    db.reports.insert_one(record)

    try:
        result = index_report(destination, patient_id)
        db.reports.update_one({"report_id": record["report_id"]}, {"$set": {"index_status": "indexed", "indexed_at": now_utc(), "index_output_tail": result.get("output", "")[-2500:]}})
        record["index_status"] = "indexed"
        record["indexed_at"] = now_utc()
        record["index_output_tail"] = result.get("output", "")[-2500:]
    except Exception as exc:
        db.reports.update_one({"report_id": record["report_id"]}, {"$set": {"index_status": "failed", "index_error": str(exc)[:1800]}})
        record["index_status"] = "failed"
        record["index_error"] = str(exc)[:1800]
    return {"report": public_report(record), "message": "Report stored locally. Check index_status before relying on retrieval."}


@app.get("/api/v1/patients/{patient_id}/reports/{report_id}/download")
def download_report(patient_id: str, report_id: str, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    record = db.reports.find_one({"patient_id": patient_id, "report_id": report_id})
    if not record:
        raise HTTPException(status_code=404, detail="Report not found")
    path = Path(record.get("local_path", ""))
    try:
        resolved = path.resolve()
        allowed_root = settings.upload_dir.resolve()
        if not resolved.is_relative_to(allowed_root) or not resolved.is_file():
            raise HTTPException(status_code=404, detail="Report file is unavailable")
    except OSError as exc:
        raise HTTPException(status_code=404, detail="Report file is unavailable") from exc
    return FileResponse(str(resolved), media_type="application/pdf", filename=record.get("original_filename", "report.pdf"))


@app.get("/api/v1/patients/{patient_id}/alerts")
def list_alerts(patient_id: str, limit: int = Query(default=100, ge=1, le=500), user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    return [serialize_document(x) for x in db.alerts.find({"patient_id": patient_id}).sort("created_at", DESCENDING).limit(limit)]


@app.post("/api/v1/patients/{patient_id}/alerts/{alert_id}/acknowledge")
def acknowledge_alert(patient_id: str, alert_id: str, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    result = db.alerts.update_one(
        {"patient_id": patient_id, "$or": [{"alert_id": alert_id}, {"_id": _object_id_or_none(alert_id)}]},
        {"$set": {"acknowledged": True, "acknowledged_at": now_utc(), "acknowledged_by": user["username"]}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"status": "acknowledged", "alert_id": alert_id}


def _object_id_or_none(value: str):
    from bson import ObjectId
    try:
        return ObjectId(value)
    except Exception:
        return None


@app.get("/api/v1/patients/{patient_id}/medications")
def list_medications(patient_id: str, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    return [serialize_document(x) for x in db.medications.find({"patient_id": patient_id, "active": {"$ne": False}}).sort("created_at", DESCENDING)]


@app.post("/api/v1/patients/{patient_id}/medications", status_code=201)
def create_medication(patient_id: str, payload: MedicationCreate, user: dict = Depends(get_current_user)):
    patient = get_patient_or_404(patient_id, user)
    record = payload.model_dump()
    record.update({
        "patient_id": patient_id,
        "created_at": now_utc(),
        "source": "caregiver_entered",
        "verification_status": "unverified",
        "synthetic": patient.get("synthetic", False),
        "active": True,
    })
    result = db.medications.insert_one(record)
    record["id"] = str(result.inserted_id)
    return serialize_document(record)


@app.post("/api/v1/patients/{patient_id}/medications/{medication_id}/events", status_code=201)
def log_medication_event(patient_id: str, medication_id: str, payload: MedicationEventCreate, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    medication = db.medications.find_one({"patient_id": patient_id, "$or": [{"medication_id": medication_id}, {"_id": _object_id_or_none(medication_id)}]})
    if not medication:
        raise HTTPException(status_code=404, detail="Medication entry not found")
    record = payload.model_dump()
    record.update({"patient_id": patient_id, "medication_id": medication_id, "created_by": user["username"]})
    result = db.medication_events.insert_one(record)
    record["id"] = str(result.inserted_id)
    if payload.event == "dose_marked_taken":
        record["note"] = "Caregiver marked the dose as taken; this is not sensor-verified."
    return serialize_document(record)


@app.get("/api/v1/patients/{patient_id}/medication-events")
def list_medication_events(patient_id: str, limit: int = Query(default=100, ge=1, le=500), user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    return [serialize_document(x) for x in db.medication_events.find({"patient_id": patient_id}).sort("recorded_at", DESCENDING).limit(limit)]


@app.post("/api/v1/patients/{patient_id}/explain")
def explain_patient_context(patient_id: str, payload: ExplainRequest, user: dict = Depends(get_current_user)):
    get_patient_or_404(patient_id, user)
    symptoms = list(db.symptoms.find({"patient_id": patient_id}).sort("onset_at", DESCENDING).limit(10))
    history = list(db.history.find({"patient_id": patient_id}).sort("created_at", DESCENDING).limit(10))
    try:
        result = answer_question(
            patient_id=patient_id,
            question=payload.question,
            top_k=payload.top_k,
            symptom_records=symptoms,
            history_records=history,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (FileNotFoundError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=f"RAG component is not ready: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Explanation could not pass its evidence/citation checks: {str(exc)[:500]}") from exc
    return result


@app.get("/api/v1/system/status")
def system_status(user: dict = Depends(get_current_user)):
    return {
        "service": "lifora",
        "database": "connected",
        "rag": rag_status(),
        "ml": model_status(),
        "emergency_monitoring": {
            "status": "running" if settings.emergency_monitor_interval_seconds >= 1 else "disabled",
            "inactivity_threshold_seconds": settings.inactivity_threshold_seconds,
            "response_timeout_seconds": settings.inactivity_response_timeout_seconds,
            "sensor_stale_after_seconds": settings.sensor_stale_after_seconds,
            "trend_window_seconds": settings.trend_window_seconds,
            "local_alarm_seconds": settings.local_alarm_active_seconds,
            "local_snooze_seconds": settings.local_alarm_snooze_seconds,
            "contact_encryption_configured": bool(settings.emergency_contact_encryption_key),
            "twilio_enabled": settings.twilio_enabled,
            "twilio_credentials_configured": bool(settings.twilio_account_sid and settings.twilio_auth_token and settings.twilio_from_number),
            "sms_enabled": settings.twilio_enabled and settings.emergency_sms_enabled,
            "calls_enabled": settings.twilio_enabled and settings.emergency_calls_enabled,
            "note": "Thresholds are prototype settings and require clinical validation; notifications are opt-in.",
        },
        "privacy": {
            "real_report_uploads_enabled": settings.allow_real_report_uploads,
            "cloud_llm_enabled": settings.allow_cloud_llm,
            "real_data_to_cloud_llm_enabled": settings.allow_real_data_to_cloud_llm,
            "note": "Keep real-data cloud calls disabled until privacy, consent, security and applicable legal requirements are reviewed.",
        },
    }

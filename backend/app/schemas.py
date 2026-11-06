from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoginRequest(StrictModel):
    username: str = Field(min_length=3, max_length=120)
    password: str = Field(min_length=1, max_length=200)


class PatientCreate(StrictModel):
    patient_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    display_name: str = Field(min_length=1, max_length=120)
    synthetic: bool = False


class VitalCreate(StrictModel):
    heart_rate_bpm: int | None = Field(default=None, ge=20, le=250)
    spo2_percent: float | None = Field(default=None, ge=50, le=100)
    temperature_c: float | None = Field(default=None, ge=25, le=45)
    fall_detected: bool = False
    medicine_box_open: bool | None = None
    motion_event: str | None = Field(default=None, max_length=80)
    movement_detected: bool | None = None
    movement_level: float | None = Field(default=None, ge=0, le=10)
    signal_quality: Literal["good", "poor", "unknown"] = "unknown"
    source: Literal["caregiver_manual", "esp32", "simulator"] = "caregiver_manual"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def require_measurement_or_event(self):
        if all(v is None for v in (self.heart_rate_bpm, self.spo2_percent, self.temperature_c)) and not self.fall_detected and self.medicine_box_open is None and not self.motion_event and self.movement_detected is None and self.movement_level is None:
            raise ValueError("Provide at least one measurement or device event")
        return self


class SymptomCreate(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    severity: Literal["mild", "moderate", "severe"]
    onset_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = Field(default="", max_length=500)


class HistoryCreate(StrictModel):
    category: Literal["condition", "diagnosis", "medication", "allergy", "other"]
    name: str = Field(min_length=1, max_length=120)
    details: str = Field(default="", max_length=500)
    reported_on: date = Field(default_factory=date.today)


class MedicationCreate(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    dosage_text: str = Field(default="Not recorded", max_length=120)
    schedule_text: str = Field(default="Not recorded", max_length=200)
    notes: str = Field(default="", max_length=400)


class MedicationEventCreate(StrictModel):
    event: Literal["box_opened", "dose_marked_taken", "dose_missed", "refill"]
    notes: str = Field(default="", max_length=300)
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExplainRequest(StrictModel):
    question: str = Field(min_length=4, max_length=500)
    top_k: int = Field(default=3, ge=1, le=8)


class DeviceIngestRequest(StrictModel):
    patient_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    person_response: bool = False
    device_id: str = Field(min_length=1, max_length=80)
    heart_rate_bpm: int | None = Field(default=None, ge=20, le=250)
    spo2_percent: float | None = Field(default=None, ge=50, le=100)
    temperature_c: float | None = Field(default=None, ge=25, le=45)
    fall_detected: bool = False
    medicine_box_open: bool | None = None
    motion_event: str | None = Field(default=None, max_length=80)
    movement_detected: bool | None = None
    movement_level: float | None = Field(default=None, ge=0, le=10)
    signal_quality: Literal["good", "poor", "unknown"] = "unknown"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def require_measurement_or_event(self):
        if all(v is None for v in (self.heart_rate_bpm, self.spo2_percent, self.temperature_c)) and not self.fall_detected and self.medicine_box_open is None and not self.motion_event and self.movement_detected is None and self.movement_level is None and not self.person_response:
            raise ValueError("Provide at least one measurement, device event, or person response")
        return self


class EmergencyContactInput(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    relationship: str = Field(default="caregiver", max_length=50)
    phone: str = Field(min_length=9, max_length=16, pattern=r"^\+[1-9][0-9]{7,14}$")
    priority: int = Field(default=1, ge=1, le=3)


class EmergencyContactsUpdate(StrictModel):
    contacts: list[EmergencyContactInput] = Field(max_length=3)

    @model_validator(mode="after")
    def unique_priorities(self):
        priorities = [item.priority for item in self.contacts]
        if len(priorities) != len(set(priorities)):
            raise ValueError("Each emergency contact must have a unique priority")
        if priorities and 1 not in priorities:
            raise ValueError("The primary emergency contact must have priority 1")
        return self


class PersonResponseRequest(StrictModel):
    person_is_responsive: bool = True


class EmergencyResolveRequest(StrictModel):
    safety_confirmed: bool
    notes: str = Field(min_length=3, max_length=300)

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


class Settings:
    app_name = os.getenv("APP_NAME", "Lifora API")
    app_env = os.getenv("APP_ENV", "development")
    api_prefix = os.getenv("API_PREFIX", "/api/v1")
    mongo_uri = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
    mongo_db = os.getenv("MONGO_DB", "lifora_db")
    jwt_secret_key = os.getenv("JWT_SECRET_KEY", "")
    access_token_expire_minutes = _int("ACCESS_TOKEN_EXPIRE_MINUTES", 720)
    bootstrap_admin_username = os.getenv("BOOTSTRAP_ADMIN_USERNAME", "caregiver@example.com")
    bootstrap_admin_password = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")
    seed_demo_data = _bool("SEED_DEMO_DATA", True)
    cors_origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
    device_ingest_token = os.getenv("DEVICE_INGEST_TOKEN", "")
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    allow_cloud_llm = _bool("ALLOW_CLOUD_LLM", False)
    allow_real_data_to_cloud_llm = _bool("ALLOW_REAL_DATA_TO_CLOUD_LLM", False)
    allow_real_report_uploads = _bool("ALLOW_REAL_REPORT_UPLOADS", False)
    max_upload_mb = _int("MAX_UPLOAD_MB", 10)
    demo_hr_min = _int("DEMO_HR_MIN", 40)
    demo_hr_max = _int("DEMO_HR_MAX", 130)
    demo_spo2_low = _float("DEMO_SPO2_LOW", 90.0)
    demo_temperature_high_c = _float("DEMO_TEMPERATURE_HIGH_C", 39.5)

    # Emergency-monitoring thresholds are configurable prototype defaults only.
    emergency_monitor_interval_seconds = max(1, _int("EMERGENCY_MONITOR_INTERVAL_SECONDS", 5))
    sensor_stale_after_seconds = max(10, _int("SENSOR_STALE_AFTER_SECONDS", 60))
    inactivity_threshold_seconds = max(5, _int("INACTIVITY_THRESHOLD_SECONDS", 120))
    inactivity_response_timeout_seconds = max(5, _int("INACTIVITY_RESPONSE_TIMEOUT_SECONDS", 30))
    local_alarm_active_seconds = max(5, _int("LOCAL_ALARM_ACTIVE_SECONDS", 300))
    local_alarm_snooze_seconds = max(5, _int("LOCAL_ALARM_SNOOZE_SECONDS", 900))
    warning_alarm_active_seconds = max(5, _int("WARNING_ALARM_ACTIVE_SECONDS", 300))
    vital_persistence_seconds = max(1, _int("VITAL_PERSISTENCE_SECONDS", 30))
    trend_window_seconds = max(30, _int("TREND_WINDOW_SECONDS", 300))
    hr_trend_change_bpm = max(1, _int("HR_TREND_CHANGE_BPM", 20))
    spo2_trend_drop_percent = max(0.5, _float("SPO2_TREND_DROP_PERCENT", 3.0))
    temp_trend_rise_c = max(0.1, _float("TEMP_TREND_RISE_C", 1.0))
    movement_threshold = _float("MOVEMENT_THRESHOLD", 0.15)
    warning_hr_low = _int("WARNING_HR_LOW", 50)
    warning_hr_high = _int("WARNING_HR_HIGH", 120)
    critical_hr_low = _int("CRITICAL_HR_LOW", 40)
    critical_hr_high = _int("CRITICAL_HR_HIGH", 140)
    warning_spo2_low = _float("WARNING_SPO2_LOW", 92.0)
    critical_spo2_low = _float("CRITICAL_SPO2_LOW", 88.0)
    warning_temp_high_c = _float("WARNING_TEMP_HIGH_C", 38.0)
    critical_temp_high_c = _float("CRITICAL_TEMP_HIGH_C", 40.0)

    # Twilio is opt-in and disabled by default. Never commit these values.
    twilio_enabled = _bool("TWILIO_ENABLED", False)
    twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    twilio_from_number = os.getenv("TWILIO_FROM_NUMBER", "").strip()
    emergency_contact_encryption_key = os.getenv("EMERGENCY_CONTACT_ENCRYPTION_KEY", "").strip()
    emergency_sms_enabled = _bool("EMERGENCY_SMS_ENABLED", False)
    emergency_calls_enabled = _bool("EMERGENCY_CALLS_ENABLED", False)
    sms_retry_seconds = max(30, _int("SMS_RETRY_SECONDS", 300))
    call_retry_seconds = max(60, _int("CALL_RETRY_SECONDS", 900))
    max_sms_attempts = max(1, _int("MAX_SMS_ATTEMPTS", 5))
    max_call_attempts = max(1, _int("MAX_CALL_ATTEMPTS", 3))
    upload_dir = PROJECT_ROOT / "data" / "uploads"
    rag_src_dir = PROJECT_ROOT / "rag-llm" / "src"
    rag_patient_dir = PROJECT_ROOT / "rag-llm" / "knowledge_base" / "patients"


settings = Settings()

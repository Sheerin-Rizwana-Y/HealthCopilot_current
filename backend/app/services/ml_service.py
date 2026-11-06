from __future__ import annotations

from ..config import PROJECT_ROOT


def model_status() -> dict:
    """Do not guess feature schemas or activate unvalidated experimental models."""
    model_dir = PROJECT_ROOT / "models"
    artifacts = sorted(str(p.relative_to(PROJECT_ROOT)) for p in model_dir.glob("**/*") if p.is_file()) if model_dir.exists() else []
    return {
        "fall_detection": {
            "status": "not_connected",
            "reason": "The artifact path and exact feature-preprocessing schema must be confirmed before inference is wired into the API.",
        },
        "vitals_model": {
            "status": "disabled",
            "reason": "The current experimental model metrics are not sufficient for clinical alerting; not used to drive alerts.",
        },
        "discovered_model_files": artifacts,
        "note": "RAG/LLM explanations do not override independent rule alerts. Model integration is intentionally fail-closed until the exact artifacts and input schemas are confirmed.",
    }

"""Cloud FAMACHA scoring service (Model B).

This is the SECOND model of the anemia pipeline. It only runs after the
conjunctiva model (Model A, ``conjunctiva_roboflow.py``) has found a valid
conjunctiva. It runs the cloud FAMACHA model and never uses any local model.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import requests

DEFAULT_MODEL_ID = "prueba-3-hajoz-zq7mv/1"
DEFAULT_INPUT_MODE = "full"
DEFAULT_API_URL = "https://serverless.roboflow.com"

# ---------------------------------------------------------------------------
# CLASS MAPPING (Maps dataset labels to FAMACHA scores 1 to 5)
# ---------------------------------------------------------------------------
CLASS_TO_SCORE: dict[str, int] = {
    # Nivel 1 to 5 (Standard Latin America / Roboflow FAMACHA labels)
    "nivel1": 1,
    "nivel2": 2,
    "nivel3": 3,
    "nivel4": 4,
    "nivel5": 5,
    "nivel 1": 1,
    "nivel 2": 2,
    "nivel 3": 3,
    "nivel 4": 4,
    "nivel 5": 5,
    # Standard FAMACHA 1-5 naming
    "famacha_1": 1,
    "famacha_2": 2,
    "famacha_3": 3,
    "famacha_4": 4,
    "famacha_5": 5,
    "famacha1": 1,
    "famacha2": 2,
    "famacha3": 3,
    "famacha4": 4,
    "famacha5": 5,
    "famacha 1": 1,
    "famacha 2": 2,
    "famacha 3": 3,
    "famacha 4": 4,
    "famacha 5": 5,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "score 1": 1,
    "score 2": 2,
    "score 3": 3,
    "score 4": 4,
    "score 5": 5,
}

VALID_INPUT_MODES = ("crop", "full")


class FamachaRoboflowError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def model_id() -> str:
    return os.getenv("FAMACHA_ROBOFLOW_MODEL_ID", "").strip() or DEFAULT_MODEL_ID


def api_url() -> str:
    return (
        os.getenv("FAMACHA_ROBOFLOW_API_URL", DEFAULT_API_URL)
        .strip()
        .rstrip("/")
    )


def _api_key() -> str:
    return (
        os.getenv("FAMACHA_ROBOFLOW_API_KEY", "").strip()
        or os.getenv("CONJUNCTIVA_ROBOFLOW_API_KEY", "").strip()
        or os.getenv("ROBOFLOW_API_KEY", "").strip()
    )


def input_mode() -> str:
    mode = os.getenv("FAMACHA_INPUT_MODE", "").strip().lower()
    return mode if mode in VALID_INPUT_MODES else DEFAULT_INPUT_MODE


def _normalise(name: Any) -> str:
    return str(name).strip().lower()


def missing_configuration() -> list[str]:
    """Human-readable list of what still has to be set. Empty list = ready."""
    missing: list[str] = []
    if not model_id():
        missing.append("FAMACHA_ROBOFLOW_MODEL_ID")
    if not _api_key():
        missing.append("FAMACHA_ROBOFLOW_API_KEY (or CONJUNCTIVA_ROBOFLOW_API_KEY)")
    if not CLASS_TO_SCORE:
        missing.append("CLASS_TO_SCORE class mapping in famacha_roboflow.py")
    return missing


def famacha_is_configured() -> bool:
    return not missing_configuration()


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _to_dict(value: Any) -> Any:
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except Exception:
            return value
    return value


def parse_top_prediction(raw: Any) -> tuple[str, float]:
    """Return (class_name, confidence 0..1) from a Roboflow response."""
    raw = _to_dict(raw)
    if isinstance(raw, list):
        raw = _to_dict(raw[0]) if raw else {}
    if not isinstance(raw, dict):
        raise FamachaRoboflowError("Scoring model returned an unsupported response type.")

    candidates: list[tuple[str, float]] = []

    top = raw.get("top")
    top_conf = raw.get("confidence")
    if isinstance(top, str) and isinstance(top_conf, (int, float)):
        candidates.append((top, float(top_conf)))

    preds = raw.get("predictions")
    if isinstance(preds, list):
        for item in preds:
            item = _to_dict(item)
            if not isinstance(item, dict):
                continue
            name = item.get("class", item.get("class_name"))
            conf = item.get("confidence")
            if name is not None and isinstance(conf, (int, float)):
                candidates.append((str(name), float(conf)))
    elif isinstance(preds, dict):
        for name, item in preds.items():
            item = _to_dict(item)
            conf = item.get("confidence") if isinstance(item, dict) else None
            if isinstance(conf, (int, float)):
                candidates.append((str(name), float(conf)))

    if not candidates:
        raise FamachaRoboflowError("FAMACHA model returned no prediction.")

    name, conf = max(candidates, key=lambda c: c[1])
    if not 0.0 <= conf <= 1.0:
        raise FamachaRoboflowError(f"FAMACHA model returned an invalid confidence: {conf}")
    return name, conf


def map_class_to_score(class_name: str) -> int:
    table = {_normalise(k): v for k, v in CLASS_TO_SCORE.items()}
    key = _normalise(class_name)
    if key not in table:
        raise FamachaRoboflowError(
            f"FAMACHA model returned an unknown class '{class_name}'. "
            "No score was produced."
        )
    return int(table[key])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_famacha_model(image_bytes: bytes, full_image_bytes: bytes | None = None) -> dict[str, Any]:
    """Run Model B on image bytes and return a validated result.

    Uses cloud FAMACHA scoring and never returns a score from a local model.
    """
    missing = missing_configuration()
    if missing:
        raise FamachaRoboflowError(
            "FAMACHA model is not configured. Missing: " + ", ".join(missing)
        )

    key = _api_key()
    target_model = model_id()
    endpoint = f"{api_url()}/{target_model}"

    raw_result = None
    last_error = None

    # Helper for REST inference with confidence threshold
    def _call_rest(data: bytes, min_conf: float = 0.05) -> dict[str, Any] | None:
        try:
            resp = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}"},
                params={"confidence": min_conf},
                files={"file": ("image.jpg", data, "image/jpeg")},
                timeout=15,
            )
            if resp.status_code == 200:
                js = resp.json()
                if isinstance(js, dict) and js.get("predictions"):
                    return js
        except Exception as e:
            pass
        return None

    # 1. Primary: Try REST with low confidence (0.05) on the given image bytes
    raw_result = _call_rest(image_bytes, min_conf=0.05)

    # 2. If empty and full image bytes available, try on full image bytes
    if not raw_result and full_image_bytes and full_image_bytes != image_bytes:
        raw_result = _call_rest(full_image_bytes, min_conf=0.05)

    # 3. Fallback: Try official InferenceHTTPClient if REST had no predictions
    if not raw_result:
        temp_path: str | None = None
        try:
            from inference_sdk import InferenceConfiguration, InferenceHTTPClient

            client = InferenceHTTPClient(api_url=api_url(), api_key=key).configure(
                InferenceConfiguration(api_key_transport="header")
            )
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(image_bytes)
                temp_path = tmp.name
            raw_result = client.infer(temp_path, model_id=target_model)
        except Exception as exc:
            last_error = exc
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    if not raw_result:
        raise FamachaRoboflowError(
            f"FAMACHA inference produced no predictions: {last_error or 'no matches found'}"
        )

    class_name, confidence = parse_top_prediction(raw_result)
    score = map_class_to_score(class_name)

    return {
        "model_id": target_model,
        "provider": "FAMACHA Scoring Model",
        "class_name": class_name,
        "confidence": confidence,
        "score": score,
        "input_mode": input_mode(),
    }

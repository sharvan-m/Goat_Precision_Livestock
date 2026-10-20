"""Roboflow FAMACHA scoring service (Model B).

This is the SECOND model of the anemia pipeline.  It only runs after the
conjunctiva model (Model A, ``conjunctiva_roboflow.py``) has found a valid
conjunctiva.  It is completely separate from Model A and never falls back to
the local ``famacha_best.pt`` file.

Configuration (backend/.env, never committed, never sent to the browser):

    FAMACHA_ROBOFLOW_MODEL_ID   e.g. "project-name/3"        (required)
    FAMACHA_ROBOFLOW_API_KEY    optional; if empty the conjunctiva key is used
                                (Roboflow keys are per workspace)
    FAMACHA_ROBOFLOW_API_URL    optional, default https://serverless.roboflow.com
    FAMACHA_INPUT_MODE          "crop"  -> send only the detected conjunctiva region
                                "full"  -> send the whole uploaded image
                                (required: it must match how Model B was trained)

The class-name -> score table below must be filled from Model B's REAL class
names.  Until it is filled, the service reports "not configured" and no score is
ever produced.  Unknown classes are rejected, never guessed.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# CLASS MAPPING  (fill in from Model B's real class names)
# ---------------------------------------------------------------------------
# Keys are compared case-insensitively after stripping spaces.
# Example once the real names are known:
#     CLASS_TO_SCORE = {"famacha_1": 1, "famacha_2": 2, "famacha_3": 3,
#                       "famacha_4": 4, "famacha_5": 5}
CLASS_TO_SCORE: dict[str, int] = {}

VALID_INPUT_MODES = ("crop", "full")


class FamachaRoboflowError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Configuration helpers (read at call time so tests and .env changes work)
# ---------------------------------------------------------------------------

def model_id() -> str:
    return os.getenv("FAMACHA_ROBOFLOW_MODEL_ID", "").strip()


def api_url() -> str:
    return (
        os.getenv("FAMACHA_ROBOFLOW_API_URL", "https://serverless.roboflow.com")
        .strip()
        .rstrip("/")
    )


def _api_key() -> str:
    return (
        os.getenv("FAMACHA_ROBOFLOW_API_KEY", "").strip()
        or os.getenv("CONJUNCTIVA_ROBOFLOW_API_KEY", "").strip()
    )


def input_mode() -> str:
    return os.getenv("FAMACHA_INPUT_MODE", "").strip().lower()


def _normalise(name: Any) -> str:
    return str(name).strip().lower()


def missing_configuration() -> list[str]:
    """Human-readable list of what still has to be set. Empty list = ready."""
    missing: list[str] = []
    if not model_id():
        missing.append("FAMACHA_ROBOFLOW_MODEL_ID")
    if not _api_key():
        missing.append("FAMACHA_ROBOFLOW_API_KEY (or CONJUNCTIVA_ROBOFLOW_API_KEY)")
    if input_mode() not in VALID_INPUT_MODES:
        missing.append("FAMACHA_INPUT_MODE (crop or full)")
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
    """Return (class_name, confidence 0..1) from a Roboflow response.

    Handles the three shapes Roboflow can return:
      * single-label classification: {"top": "...", "confidence": 0.9, ...}
      * list of predictions:         {"predictions": [{"class": ..., "confidence": ...}]}
      * multi-label dictionary:      {"predictions": {"name": {"confidence": ...}}}
    """
    raw = _to_dict(raw)
    if isinstance(raw, list):
        raw = _to_dict(raw[0]) if raw else {}
    if not isinstance(raw, dict):
        raise FamachaRoboflowError("Roboflow returned an unsupported response type.")

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

def _client():
    key = _api_key()
    if not key:
        raise FamachaRoboflowError("FAMACHA Roboflow API key is not configured.")
    try:
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient
    except ImportError as exc:
        raise FamachaRoboflowError(
            "inference-sdk is not installed. Run: pip install inference-sdk"
        ) from exc
    return InferenceHTTPClient(api_url=api_url(), api_key=key).configure(
        InferenceConfiguration(api_key_transport="header")
    )


def run_famacha_model(image_bytes: bytes) -> dict[str, Any]:
    """Run Model B on ``image_bytes`` and return a validated result.

    Raises ``FamachaRoboflowError`` for any problem; it never returns a score
    that did not come from the configured Roboflow model.
    """
    missing = missing_configuration()
    if missing:
        raise FamachaRoboflowError(
            "FAMACHA Roboflow model is not configured. Missing: " + ", ".join(missing)
        )

    client = _client()
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name
        raw = client.infer(temp_path, model_id=model_id())
    except FamachaRoboflowError:
        raise
    except Exception as exc:
        raise FamachaRoboflowError(f"Roboflow FAMACHA inference failed: {exc}") from exc
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass

    class_name, confidence = parse_top_prediction(raw)
    score = map_class_to_score(class_name)

    return {
        "model_id": model_id(),
        "provider": "Roboflow Serverless Cloud",
        "class_name": class_name,
        "confidence": confidence,
        "score": score,
        "input_mode": input_mode(),
    }

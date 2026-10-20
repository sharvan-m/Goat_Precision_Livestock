"""Local FAMACHA scoring service using custom-trained YOLO11 classification weights.

Loads the local model from Anemia/goat_eye_detection/model/famacha_best.pt
and classifies the cropped conjunctiva into FAMACHA scores 1 to 5.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAMACHA_MODEL = PROJECT_ROOT / "Anemia" / "goat_eye_detection" / "model" / "famacha_best.pt"

_cached_model = None
_cached_path: Path | None = None


class FamachaLocalError(RuntimeError):
    pass


def get_model_path() -> Path:
    return DEFAULT_FAMACHA_MODEL


def famacha_model_is_available() -> bool:
    return get_model_path().exists()


def load_famacha_model():
    global _cached_model, _cached_path
    target_path = get_model_path()

    if not target_path.exists():
        raise FamachaLocalError(f"Local FAMACHA model not found at: {target_path}")

    if _cached_model is not None and _cached_path == target_path:
        return _cached_model

    try:
        from ultralytics import YOLO

        model = YOLO(str(target_path))
        _cached_model = model
        _cached_path = target_path
        return model
    except Exception as exc:
        raise FamachaLocalError(f"Failed to load local FAMACHA model from {target_path}: {exc}") from exc


def risk_level(score: int | None) -> str:
    if score is None:
        return "Unavailable"
    if score <= 2:
        return "Lower anemia risk"
    if score == 3:
        return "Borderline / review"
    return "Higher anemia risk"


def run_famacha_scoring(image_input: bytes | np.ndarray) -> dict[str, Any]:
    """Classify conjunctiva image / crop using famacha_best.pt and return score."""
    model = load_famacha_model()

    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise FamachaLocalError("Could not decode image bytes for FAMACHA scoring.")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise FamachaLocalError(f"Unsupported image input type: {type(image_input)}")

    # Ensure image has sufficient size
    h, w = img.shape[:2]
    if h < 8 or w < 8:
        raise FamachaLocalError(f"Image ROI is too small for scoring ({w}x{h}).")

    try:
        results = model(img, verbose=False)
    except Exception as exc:
        raise FamachaLocalError(f"Inference error in famacha_best.pt: {exc}") from exc

    if not results or len(results) == 0:
        raise FamachaLocalError("FAMACHA classifier produced no results.")

    r = results[0]
    probs = getattr(r, "probs", None)
    if probs is None:
        raise FamachaLocalError("Model output missing probabilities.")

    top1 = int(probs.top1)
    top1conf = float(probs.top1conf)
    cls_name = str(model.names.get(top1, f"FAMACHA_{top1 + 1}"))

    # Extract numeric score from class name e.g. "FAMACHA_4" -> 4, "Nivel2" -> 2
    digits = "".join(filter(str.isdigit, cls_name))
    if digits:
        score = int(digits)
    else:
        score = top1 + 1

    # Clamp score to 1-5 range
    score = max(1, min(5, score))

    return {
        "model_id": get_model_path().name,
        "provider": "Local Classifier (famacha_best.pt)",
        "score": score,
        "class_name": cls_name,
        "confidence": top1conf,
        "risk_level": risk_level(score),
    }

"""Roboflow conjunctiva instance-segmentation service.

Uses Roboflow's inference-sdk against the hosted model
``segmentation_conjunctiva-rdlmb/3``.  The API key is read only from the
backend environment and is never returned to the frontend.
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

MODEL_ID = os.getenv("CONJUNCTIVA_ROBOFLOW_MODEL_ID", "segmentation_conjunctiva-rdlmb/3").strip()
API_URL = os.getenv("CONJUNCTIVA_ROBOFLOW_API_URL", "https://serverless.roboflow.com").strip().rstrip("/")
API_KEY = os.getenv("CONJUNCTIVA_ROBOFLOW_API_KEY", "").strip()


class ConjunctivaRoboflowError(RuntimeError):
    pass


def conjunctiva_is_configured() -> bool:
    return bool(API_KEY and MODEL_ID)


def conjunctiva_endpoint() -> str:
    return API_URL


def _client():
    if not API_KEY:
        raise ConjunctivaRoboflowError(
            "CONJUNCTIVA_ROBOFLOW_API_KEY is not configured."
        )
    try:
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    except ImportError as exc:
        raise ConjunctivaRoboflowError(
            "inference-sdk is not installed. Run: pip install inference-sdk"
        ) from exc

    return InferenceHTTPClient(
        api_url=API_URL,
        api_key=API_KEY,
    ).configure(
        InferenceConfiguration(api_key_transport="header")
    )


def _point_list(value: Any) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, float]] = []
    for p in value:
        if isinstance(p, dict) and "x" in p and "y" in p:
            try:
                out.append({"x": float(p["x"]), "y": float(p["y"])})
            except (TypeError, ValueError):
                continue
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            try:
                out.append({"x": float(p[0]), "y": float(p[1])})
            except (TypeError, ValueError):
                continue
    return out


def _normalize_prediction(pred: Any) -> dict[str, Any]:
    if hasattr(pred, "dict"):
        try:
            pred = pred.dict()
        except Exception:
            pass
    if not isinstance(pred, dict):
        try:
            pred = dict(pred)
        except Exception:
            pred = {}

    confidence = pred.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    result: dict[str, Any] = {
        "class": str(pred.get("class", pred.get("class_name", "conjunctiva"))),
        "confidence": confidence,
    }

    for key in ("x", "y", "width", "height"):
        if key in pred:
            try:
                result[key] = float(pred[key])
            except (TypeError, ValueError):
                result[key] = pred[key]

    points = _point_list(
        pred.get("points")
        or pred.get("polygon")
        or pred.get("segmentation")
        or pred.get("mask")
    )
    if points:
        result["points"] = points

    # Preserve other useful model fields without copying secrets/config.
    for key in ("class_id", "detection_id", "parent_id"):
        if key in pred:
            result[key] = pred[key]

    return result


def _annotate(image_bytes: bytes, predictions: list[dict[str, Any]]) -> str | None:
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None

    output = image.copy()
    for pred in predictions:
        pts = pred.get("points") or []
        if len(pts) < 3:
            continue
        polygon = np.asarray(
            [[int(round(p["x"])), int(round(p["y"]))] for p in pts],
            dtype=np.int32,
        ).reshape((-1, 1, 2))
        overlay = output.copy()
        cv2.fillPoly(overlay, [polygon], (0, 255, 0))
        output = cv2.addWeighted(overlay, 0.30, output, 0.70, 0)
        cv2.polylines(output, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)

    ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def run_conjunctiva_workflow(image_bytes: bytes) -> dict[str, Any]:
    """Run ``segmentation_conjunctiva-rdlmb/3`` on uploaded image bytes."""
    client = _client()

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        raw = client.infer(temp_path, model_id=MODEL_ID)
        if hasattr(raw, "dict"):
            try:
                raw = raw.dict()
            except Exception:
                pass
        if not isinstance(raw, dict):
            try:
                raw = dict(raw)
            except Exception as exc:
                raise ConjunctivaRoboflowError(
                    "Roboflow returned an unsupported response type."
                ) from exc

        raw_predictions = raw.get("predictions") or []
        predictions = [_normalize_prediction(p) for p in raw_predictions]

        return {
            "model_id": MODEL_ID,
            "provider": "Roboflow Serverless Cloud",
            "predictions": predictions,
            "prediction_count": len(predictions),
            "annotated_image": _annotate(image_bytes, predictions),
            "raw_image": raw.get("image"),
        }
    except ConjunctivaRoboflowError:
        raise
    except Exception as exc:
        raise ConjunctivaRoboflowError(
            f"Roboflow conjunctiva inference failed: {exc}"
        ) from exc
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass

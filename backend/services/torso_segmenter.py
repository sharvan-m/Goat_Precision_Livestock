"""
Torso and body segmentation service using the deployed Roboflow model:
team3-ohd3p/goat-weight-prediction/2

Provides high-resolution instance segmentation of the caprine Torso and Full Body.
The extracted torso polygon enables biometric volume calculation and Body Condition Score (BCS).
"""

import os
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

try:
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False


_CLIENT_CACHE = {}


def get_torso_client() -> Optional[Any]:
    if not INFERENCE_AVAILABLE:
        return None
    api_key = os.getenv("TORSO_ROBOFLOW_API_KEY") or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        return None
    if "torso_client" not in _CLIENT_CACHE:
        try:
            config = InferenceConfiguration(api_key_transport="both")
            _CLIENT_CACHE["torso_client"] = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key,
            )
        except Exception:
            _CLIENT_CACHE["torso_client"] = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=api_key,
            )
    return _CLIENT_CACHE["torso_client"]


def detect_torso_and_body(
    image_bytes: bytes,
    image_shape: Optional[Tuple[int, int]] = None,
    min_confidence: float = 0.25,
) -> Dict[str, Any]:
    """
    Run torso and body segmentation on an image.

    Args:
        image_bytes: Raw JPEG/PNG image bytes.
        image_shape: Optional (height, width) tuple of the original image.
        min_confidence: Minimum detection confidence threshold.

    Returns:
        Dict containing torso and goat body geometry, masks, and polygons.
    """
    default_result: Dict[str, Any] = {
        "has_torso": False,
        "torso_box": None,
        "torso_mask": None,
        "torso_confidence": 0.0,
        "torso_points": [],
        "has_goat": False,
        "goat_box": None,
        "goat_mask": None,
        "goat_confidence": 0.0,
    }

    client = get_torso_client()
    if client is None:
        return default_result

    model_id = os.getenv("TORSO_ROBOFLOW_MODEL_ID", "goat-weight-prediction/2")

    # Determine image dimensions if not passed
    h, w = (0, 0)
    if image_shape:
        h, w = image_shape[:2]
    else:
        nparr = np.frombuffer(image_bytes, np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if decoded is not None:
            h, w = decoded.shape[:2]

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        temp_path = tmp.name

    try:
        res = client.infer(temp_path, model_id=model_id)
    except Exception as exc:
        # Graceful fallback: do not crash caller if cloud API fails
        return default_result
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    if not isinstance(res, dict):
        return default_result

    predictions = res.get("predictions", [])
    if not predictions:
        return default_result

    torso_pred = None
    goat_pred = None

    for p in predictions:
        cls_name = str(p.get("class", "")).strip().lower()
        conf = float(p.get("confidence", 0.0))
        if conf < min_confidence:
            continue

        if "torso" in cls_name:
            if torso_pred is None or conf > torso_pred.get("confidence", 0):
                torso_pred = p
        elif "goat" in cls_name:
            if goat_pred is None or conf > goat_pred.get("confidence", 0):
                goat_pred = p

    # Process Torso
    if torso_pred and h > 0 and w > 0:
        bx = float(torso_pred.get("x", 0))
        by = float(torso_pred.get("y", 0))
        bw = float(torso_pred.get("width", 0))
        bh = float(torso_pred.get("height", 0))
        x1 = max(0.0, bx - bw / 2.0)
        y1 = max(0.0, by - bh / 2.0)
        x2 = min(float(w), bx + bw / 2.0)
        y2 = min(float(h), by + bh / 2.0)

        points = torso_pred.get("points", [])
        torso_mask = np.zeros((h, w), dtype=np.uint8)
        if points and len(points) >= 3:
            poly_pts = np.array([[pt["x"], pt["y"]] for pt in points], dtype=np.int32)
            cv2.fillPoly(torso_mask, [poly_pts], 1)
        else:
            torso_mask[int(y1):int(y2), int(x1):int(x2)] = 1

        default_result["has_torso"] = True
        default_result["torso_box"] = [x1, y1, x2, y2]
        default_result["torso_mask"] = torso_mask
        default_result["torso_confidence"] = float(torso_pred.get("confidence", 0.0))
        default_result["torso_points"] = points

    # Process Goat Body
    if goat_pred and h > 0 and w > 0:
        bx = float(goat_pred.get("x", 0))
        by = float(goat_pred.get("y", 0))
        bw = float(goat_pred.get("width", 0))
        bh = float(goat_pred.get("height", 0))
        x1 = max(0.0, bx - bw / 2.0)
        y1 = max(0.0, by - bh / 2.0)
        x2 = min(float(w), bx + bw / 2.0)
        y2 = min(float(h), by + bh / 2.0)

        points = goat_pred.get("points", [])
        goat_mask = np.zeros((h, w), dtype=np.uint8)
        if points and len(points) >= 3:
            poly_pts = np.array([[pt["x"], pt["y"]] for pt in points], dtype=np.int32)
            cv2.fillPoly(goat_mask, [poly_pts], 1)
        else:
            goat_mask[int(y1):int(y2), int(x1):int(x2)] = 1

        default_result["has_goat"] = True
        default_result["goat_box"] = [x1, y1, x2, y2]
        default_result["goat_mask"] = goat_mask
        default_result["goat_confidence"] = float(goat_pred.get("confidence", 0.0))

    return default_result

"""Local conjunctiva detection service using custom-trained YOLO11 weights.

Loads the local model from Anemia/goat_eye_detection/model/conjunctiva_best.pt
(or best.pt fallback) and performs bounding-box object detection offline on CPU/GPU.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODEL = PROJECT_ROOT / "Anemia" / "goat_eye_detection" / "model" / "conjunctiva_best.pt"
FALLBACK_LOCAL_MODEL = PROJECT_ROOT / "Anemia" / "goat_eye_detection" / "model" / "best.pt"

_cached_model = None
_cached_path: Path | None = None


class ConjunctivaLocalError(RuntimeError):
    pass


def get_model_path() -> Path:
    if DEFAULT_LOCAL_MODEL.exists():
        return DEFAULT_LOCAL_MODEL
    if FALLBACK_LOCAL_MODEL.exists():
        return FALLBACK_LOCAL_MODEL
    return DEFAULT_LOCAL_MODEL


def local_model_is_available() -> bool:
    return get_model_path().exists()


def load_local_model():
    global _cached_model, _cached_path
    target_path = get_model_path()

    if not target_path.exists():
        raise ConjunctivaLocalError(f"Local conjunctiva model file not found at: {target_path}")

    if _cached_model is not None and _cached_path == target_path:
        return _cached_model

    try:
        from ultralytics import YOLO

        model = YOLO(str(target_path))
        _cached_model = model
        _cached_path = target_path
        return model
    except Exception as exc:
        raise ConjunctivaLocalError(f"Failed to load local YOLO model from {target_path}: {exc}") from exc


def run_local_conjunctiva_detection(
    image_bytes: bytes,
    conf_threshold: float = 0.20,
) -> dict[str, Any]:
    """Run local YOLO11 conjunctiva detector on raw image bytes.

    Renders annotations as crisp green lines and badge matching user model specifications.
    """
    model = load_local_model()

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ConjunctivaLocalError("Could not decode image bytes into OpenCV image.")

    h, w = image.shape[:2]
    t0 = time.perf_counter()

    # Run inference
    results = model(image, conf=conf_threshold, verbose=False)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    predictions: list[dict[str, Any]] = []
    annotated = image.copy()
    overlay = annotated.copy()

    # Green styling for our own developed model (BGR)
    BOX_COLOR = (0, 225, 45)       # Crisp bright green line
    LABEL_BG = (0, 180, 35)        # Solid green label badge
    TEXT_COLOR = (255, 255, 255)   # White text

    if results and len(results) > 0:
        r = results[0]
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)

        if boxes is not None and len(boxes) > 0:
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = model.names.get(cls_id, "conjunctiva")

                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1

                # Check if segmentation mask polygon is available
                poly_points: list[dict[str, float]] = []
                if masks is not None and getattr(masks, "xy", None) is not None and idx < len(masks.xy):
                    raw_pts = masks.xy[idx]
                    poly_points = [{"x": round(float(pt[0]), 1), "y": round(float(pt[1]), 1)} for pt in raw_pts]

                if not poly_points:
                    # 4 corner points representing the bounding box polygon
                    poly_points = [
                        {"x": round(x1, 1), "y": round(y1, 1)},
                        {"x": round(x2, 1), "y": round(y1, 1)},
                        {"x": round(x2, 1), "y": round(y2, 1)},
                        {"x": round(x1, 1), "y": round(y2, 1)},
                    ]

                predictions.append({
                    "class": str(cls_name),
                    "class_id": cls_id,
                    "confidence": round(conf, 4),
                    "x": round(cx, 1),
                    "y": round(cy, 1),
                    "width": round(bw, 1),
                    "height": round(bh, 1),
                    "points": poly_points,
                })

                # Draw green line annotation
                ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

                if masks is not None and len(poly_points) >= 3:
                    contour = np.asarray([[int(round(p["x"])), int(round(p["y"]))] for p in poly_points], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [contour], (0, 210, 50))
                    annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
                    cv2.polylines(annotated, [contour], True, BOX_COLOR, 2, cv2.LINE_AA)
                else:
                    cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), BOX_COLOR, 2)


    # Encode annotated image as base64 data URL
    ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ConjunctivaLocalError("Failed to encode annotated image to JPEG.")

    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

    return {
        "model_id": f"local:{get_model_path().name} (YOLO11n)",
        "provider": "Our Local Model (YOLO11n)",
        "model_type": "bounding_box",
        "predictions": predictions,
        "prediction_count": len(predictions),
        "annotated_image": annotated_b64,
        "raw_image": {"width": w, "height": h},
        "latency_ms": elapsed_ms,
    }

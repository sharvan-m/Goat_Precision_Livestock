"""
GoatVision FAMACHA service.

Pipeline:

    Uploaded goat-eye image
            |
            v
    Roboflow conjunctiva detector
            |
            v
    Conjunctiva ROI / crop
            |
            v
    Local FAMACHA 1-5 model
            |
            v
    FAMACHA score

The existing local FAMACHA model is preserved.

The service tries:
    1. Original image
    2. Roboflow conjunctiva crop
    3. Larger padded conjunctiva crop

This makes the scoring stage much more robust to full-frame images.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from backend.services.roboflow_workflow import (
    RoboflowWorkflowError,
    run_conjunctiva_workflow,
)


# ============================================================
# MODEL
# ============================================================

DEFAULT_MODEL = Path(
    r"D:\weight_detect\runs\famacha\famacha_seg_v1-2\weights\best.pt"
)

EXPECTED_CLASSES = {
    0: "Nivel1",
    1: "Nivel2",
    2: "Nivel3",
    3: "Nivel4",
    4: "Nivel5",
}

# Small dataset -> allow lower confidence predictions.
FAMACHA_CONFIDENCE = 0.05

_model = None
_model_path: Path | None = None


# ============================================================
# MODEL LOADING
# ============================================================

def _path() -> Path:
    value = os.getenv("GOAT_FAMACHA_SEG_MODEL_PATH", "").strip()
    return Path(value) if value else DEFAULT_MODEL


def _load_model():
    global _model, _model_path

    path = _path()

    if not path.exists():
        _model = None
        _model_path = path
        return None

    if _model is None or _model_path != path:
        _model = YOLO(str(path))
        _model_path = path

    return _model


# ============================================================
# STATUS
# ============================================================

def status() -> dict[str, Any]:
    path = _path()
    model = _load_model()

    names = EXPECTED_CLASSES

    if model is not None:
        raw = getattr(model, "names", None)

        if isinstance(raw, dict):
            names = {
                int(k): str(v)
                for k, v in raw.items()
            }

    return {
        "available": model is not None,
        "model": path.name,
        "path": str(path),
        "task": "segment",
        "classes": {
            str(k): v
            for k, v in names.items()
        },
        "expected_classes": {
            str(k): v
            for k, v in EXPECTED_CLASSES.items()
        },
        "message": (
            "FAMACHA 1-5 model is loaded."
            if model is not None
            else f"FAMACHA model not found at: {path}"
        ),
    }


# ============================================================
# RISK
# ============================================================

def _risk(score: int | None) -> str:
    if score is None:
        return "Unavailable"

    if score <= 2:
        return "Lower anemia risk"

    if score == 3:
        return "Borderline / review"

    return "Higher anemia risk"


# ============================================================
# IMAGE ENCODING
# ============================================================

def _image_data_url(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            92,
        ],
    )

    if not ok:
        return ""

    payload = base64.b64encode(
        encoded.tobytes()
    ).decode("ascii")

    return f"data:image/jpeg;base64,{payload}"


# ============================================================
# ROBOfLOW HELPERS
# ============================================================

def _conjunctiva_detected(
    rf_result: dict[str, Any],
) -> bool:

    if not isinstance(rf_result, dict):
        return False

    if rf_result.get("detected") is True:
        return True

    count = rf_result.get("count")

    if isinstance(count, (int, float)) and count > 0:
        return True

    predictions = rf_result.get("predictions")

    if isinstance(predictions, list) and predictions:
        return True

    return False


def _rf_predictions(
    rf_result: dict[str, Any],
) -> list[dict[str, Any]]:

    if not isinstance(rf_result, dict):
        return []

    predictions = rf_result.get("predictions")

    if isinstance(predictions, list):
        return [
            p for p in predictions
            if isinstance(p, dict)
        ]

    return []


def _best_rf_prediction(
    rf_result: dict[str, Any],
) -> dict[str, Any] | None:

    best = rf_result.get("best_prediction")

    if isinstance(best, dict):
        return best

    predictions = _rf_predictions(rf_result)

    if not predictions:
        return None

    def confidence(item: dict[str, Any]) -> float:
        value = item.get("confidence", 0)

        if isinstance(value, (int, float)):
            return float(value)

        return 0.0

    return max(
        predictions,
        key=confidence,
    )


# ============================================================
# ROBOfLOW ROI EXTRACTION
# ============================================================

def _extract_conjunctiva_crop(
    frame: np.ndarray,
    rf_result: dict[str, Any],
    padding: float = 0.35,
) -> np.ndarray | None:

    prediction = _best_rf_prediction(rf_result)

    if not prediction:
        return None

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Preferred: x/y/width/height
    # --------------------------------------------------------

    x = prediction.get("x")
    y = prediction.get("y")
    box_w = prediction.get("width")
    box_h = prediction.get("height")

    if all(
        isinstance(v, (int, float))
        for v in (x, y, box_w, box_h)
    ):

        x1 = float(x) - float(box_w) / 2
        y1 = float(y) - float(box_h) / 2
        x2 = float(x) + float(box_w) / 2
        y2 = float(y) + float(box_h) / 2

        # Add padding.
        pad_x = float(box_w) * padding
        pad_y = float(box_h) * padding

        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y

        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))

        if x2 > x1 and y2 > y1:
            return frame[y1:y2, x1:x2].copy()

    # --------------------------------------------------------
    # Fallback: polygon points
    # --------------------------------------------------------

    points = prediction.get("points")

    if isinstance(points, list) and len(points) >= 3:

        coords = []

        for point in points:
            if not isinstance(point, dict):
                continue

            px = point.get("x")
            py = point.get("y")

            if isinstance(px, (int, float)) and isinstance(
                py, (int, float)
            ):
                coords.append(
                    (
                        int(px),
                        int(py),
                    )
                )

        if len(coords) >= 3:

            xs = [p[0] for p in coords]
            ys = [p[1] for p in coords]

            x1 = max(0, min(xs))
            y1 = max(0, min(ys))
            x2 = min(width, max(xs))
            y2 = min(height, max(ys))

            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)

            pad_x = int(box_w * padding)
            pad_y = int(box_h * padding)

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)

            if x2 > x1 and y2 > y1:
                return frame[y1:y2, x1:x2].copy()

    return None


# ============================================================
# FAMACHA INFERENCE
# ============================================================

def _predict_famacha(
    model,
    image: np.ndarray,
    source_name: str,
) -> list[dict[str, Any]]:

    predictions: list[dict[str, Any]] = []

    try:
        results = model.predict(
            source=image,
            conf=FAMACHA_CONFIDENCE,
            verbose=False,
        )
    except Exception as exc:
        print(
            f"[FAMACHA] Inference failed "
            f"({source_name}): {exc}"
        )
        return []

    for result in results:

        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)

        if boxes is None:
            continue

        for i in range(len(boxes)):

            try:
                cls = int(
                    boxes.cls[i].item()
                )

                conf = float(
                    boxes.conf[i].item()
                )

            except Exception:
                continue

            if cls < 0 or cls > 4:
                continue

            score = cls + 1

            name = EXPECTED_CLASSES.get(
                cls,
                str(cls),
            )

            polygon: list[list[float]] = []

            if (
                masks is not None
                and masks.xy is not None
                and i < len(masks.xy)
            ):
                polygon = [
                    [
                        float(x),
                        float(y),
                    ]
                    for x, y in masks.xy[i].tolist()
                ]

            predictions.append(
                {
                    "id": len(predictions) + 1,
                    "class_id": cls,
                    "class_name": name,
                    "confidence": round(
                        conf * 100,
                        1,
                    ),
                    "famacha_score": score,
                    "polygon": polygon,
                    "source": source_name,
                }
            )

    predictions.sort(
        key=lambda p: float(
            p.get("confidence", 0)
        ),
        reverse=True,
    )

    return predictions


# ============================================================
# ANNOTATE FAMACHA RESULT
# ============================================================

def _annotate_prediction(
    image: np.ndarray,
    prediction: dict[str, Any],
) -> np.ndarray:

    annotated = image.copy()

    polygon = prediction.get("polygon")

    if (
        isinstance(polygon, list)
        and len(polygon) >= 3
    ):

        pts = np.asarray(
            polygon,
            dtype=np.int32,
        )

        overlay = annotated.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            (0, 190, 80),
        )

        annotated = cv2.addWeighted(
            overlay,
            0.30,
            annotated,
            0.70,
            0,
        )

        cv2.polylines(
            annotated,
            [pts],
            True,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    score = prediction.get(
        "famacha_score"
    )

    confidence = prediction.get(
        "confidence"
    )

    label = (
        f"FAMACHA {score} "
        f"{confidence:.1f}%"
        if isinstance(
            confidence,
            (int, float),
        )
        else f"FAMACHA {score}"
    )

    cv2.putText(
        annotated,
        label,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze(
    file_bytes: bytes,
    filename: str = "image.jpg",
) -> dict[str, Any]:

    model = _load_model()
    model_status = status()

    if model is None:

        return {
            "status": "error",
            "model": model_status,
            "predictions": [],
            "famacha_score": None,
            "risk_level": "Unavailable",
            "message": model_status["message"],
        }

    # --------------------------------------------------------
    # Decode image
    # --------------------------------------------------------

    try:
        image = Image.open(
            io.BytesIO(file_bytes)
        ).convert("RGB")

        frame = np.asarray(image)

    except Exception as exc:

        return {
            "status": "error",
            "model": model_status,
            "predictions": [],
            "famacha_score": None,
            "risk_level": "Unavailable",
            "message": f"Invalid image: {exc}",
        }

    # --------------------------------------------------------
    # ROBOfLOW CONJUNCTIVA
    # --------------------------------------------------------

    conjunctiva: dict[str, Any] = {
        "configured": False,
        "success": False,
        "detected": False,
        "predictions": [],
        "confidence": None,
        "annotated_image": None,
    }

    roboflow_error = None

    try:

        rf_result = run_conjunctiva_workflow(
            file_bytes
        )

        detected = _conjunctiva_detected(
            rf_result
        )

        best_rf = _best_rf_prediction(
            rf_result
        )

        rf_confidence = None

        if best_rf:

            value = best_rf.get(
                "confidence"
            )

            if isinstance(
                value,
                (int, float),
            ):
                rf_confidence = (
                    float(value) * 100
                    if 0 <= float(value) <= 1
                    else float(value)
                )

        conjunctiva = {
            **rf_result,
            "configured": True,
            "success": True,
            "detected": detected,
            "confidence": rf_confidence,
        }

    except RoboflowWorkflowError as exc:

        roboflow_error = str(exc)

        conjunctiva["error"] = (
            roboflow_error
        )

    except Exception as exc:

        roboflow_error = str(exc)

        conjunctiva["error"] = (
            roboflow_error
        )

    # --------------------------------------------------------
    # FAMACHA ATTEMPT 1
    # Original image
    # --------------------------------------------------------

    print(
        "[FAMACHA] Trying original image..."
    )

    predictions = _predict_famacha(
        model,
        frame,
        "original",
    )

    # --------------------------------------------------------
    # FAMACHA ATTEMPT 2
    # Roboflow conjunctiva crop
    # --------------------------------------------------------

    crop = None

    if not predictions and conjunctiva.get(
        "detected"
    ):

        crop = _extract_conjunctiva_crop(
            frame,
            conjunctiva,
            padding=0.35,
        )

        if crop is not None:

            print(
                "[FAMACHA] Original image "
                "returned no prediction."
            )

            print(
                f"[FAMACHA] Trying conjunctiva "
                f"crop: {crop.shape}"
            )

            crop_predictions = (
                _predict_famacha(
                    model,
                    crop,
                    "conjunctiva_crop",
                )
            )

            if crop_predictions:
                predictions = crop_predictions

    # --------------------------------------------------------
    # FAMACHA ATTEMPT 3
    # Larger crop
    # --------------------------------------------------------

    if (
        not predictions
        and conjunctiva.get("detected")
    ):

        if crop is not None:

            # Resize crop while preserving
            # aspect ratio.
            h, w = crop.shape[:2]

            if h > 0 and w > 0:

                scale = 1.8

                new_w = int(
                    w * scale
                )

                new_h = int(
                    h * scale
                )

                expanded = cv2.resize(
                    crop,
                    (
                        new_w,
                        new_h,
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )

                print(
                    "[FAMACHA] Trying "
                    "upscaled conjunctiva crop..."
                )

                crop_predictions = (
                    _predict_famacha(
                        model,
                        expanded,
                        "conjunctiva_crop_upscaled",
                    )
                )

                if crop_predictions:
                    predictions = (
                        crop_predictions
                    )

    # --------------------------------------------------------
    # No prediction
    # --------------------------------------------------------

    if not predictions:

        message = (
            "Conjunctiva detected, but the "
            "FAMACHA model could not classify "
            "the image. Try a closer, clearer "
            "eye image."
            if conjunctiva.get("detected")
            else
            "No conjunctiva detected. "
            "Try a closer eye image."
        )

        response = {
            "status": "ok",
            "model": {
                **model_status,
                "inference_device": (
                    "cuda"
                    if getattr(
                        model,
                        "device",
                        None,
                    )
                    and str(
                        model.device
                    ).startswith("cuda")
                    else "cpu"
                ),
            },
            "predictions": [],
            "famacha_score": None,
            "risk_level": "Unavailable",
            "annotated_image": (
                conjunctiva.get(
                    "annotated_image"
                )
                or ""
            ),
            "message": message,
            "conjunctiva": conjunctiva,
            "famacha": {
                "score": None,
                "class_id": None,
                "class_name": None,
                "confidence": None,
                "risk_level": "Unavailable",
                "model": model_status["model"],
            },
        }

        if roboflow_error:
            response[
                "conjunctiva"
            ]["error"] = roboflow_error

        return response

    # --------------------------------------------------------
    # Best FAMACHA prediction
    # --------------------------------------------------------

    predictions.sort(
        key=lambda p: float(
            p.get("confidence", 0)
        ),
        reverse=True,
    )

    top = predictions[0]

    score = int(
        top["famacha_score"]
    )

    risk = _risk(score)

    # --------------------------------------------------------
    # Create local annotated image
    # --------------------------------------------------------

    if top.get("source") == "original":

        local_annotated = (
            _annotate_prediction(
                frame,
                top,
            )
        )

    else:

        # For crop predictions, show the
        # crop itself so the FAMACHA result
        # corresponds directly to the ROI.
        source_image = (
            crop
            if crop is not None
            else frame
        )

        local_annotated = (
            _annotate_prediction(
                source_image,
                top,
            )
        )

    local_image_url = _image_data_url(
        local_annotated
    )

    # Prefer Roboflow annotation for the
    # main website image.
    displayed_image = (
        conjunctiva.get(
            "annotated_image"
        )
        or local_image_url
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    message = (
        "Conjunctiva detected by Roboflow. "
        f"FAMACHA {score} calculated successfully "
        f"using the {top.get('source', 'model')}."
    )

    response = {
        "status": "ok",

        "model": {
            **model_status,
            "inference_device": (
                "cuda"
                if getattr(
                    model,
                    "device",
                    None,
                )
                and str(
                    model.device
                ).startswith("cuda")
                else "cpu"
            ),
        },

        "predictions": predictions,

        "famacha_score": score,

        "risk_level": risk,

        "annotated_image": displayed_image,

        "message": message,

        "conjunctiva": conjunctiva,

        "famacha": {
            "score": score,
            "class_id": top.get(
                "class_id"
            ),
            "class_name": top.get(
                "class_name"
            ),
            "confidence": top.get(
                "confidence"
            ),
            "risk_level": risk,
            "model": model_status["model"],
            "prediction_source": top.get(
                "source"
            ),
        },
    }

    if roboflow_error:

        response[
            "conjunctiva"
        ]["error"] = roboflow_error

    return response
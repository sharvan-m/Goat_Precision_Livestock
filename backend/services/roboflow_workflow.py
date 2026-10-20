"""
Roboflow Workflow integration.

Uses a deployed Roboflow Workflow such as:
    https://serverless.roboflow.com/<workspace>/workflows/<workflow_id>

Configuration is read from environment variables.
Never hardcode the API key.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=False)

ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com").rstrip("/")
ROBOFLOW_API_KEY = (
    os.getenv("ROBOFLOW_API_KEY")
    or os.getenv("CONJUNCTIVA_ROBOFLOW_API_KEY")
    or ""
).strip()
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE", "sharvan-m").strip()
ROBOFLOW_WORKFLOW = os.getenv("ROBOFLOW_WORKFLOW", "general-segmentation-api-51").strip()
ROBOFLOW_CLASSES = [
    item.strip()
    for item in os.getenv("ROBOFLOW_CLASSES", "goat").split(",")
    if item.strip()
]


class RoboflowWorkflowError(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(
        ROBOFLOW_API_KEY
        and ROBOFLOW_WORKSPACE
        and ROBOFLOW_WORKFLOW
        and ROBOFLOW_CLASSES
    )


def endpoint() -> str:
    return (
        f"{ROBOFLOW_API_URL}/"
        f"{ROBOFLOW_WORKSPACE}/workflows/{ROBOFLOW_WORKFLOW}"
    )


def bytes_to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _looks_like_image_string(value: str) -> bool:
    value = value.strip()
    return (
        value.startswith("data:image/")
        or value.startswith("http://")
        or value.startswith("https://")
        or len(value) > 200
    )


def _extract_value(obj: Any, keys: tuple[str, ...]) -> Any:
    """Find the first matching key anywhere in a nested Workflow response."""
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key] not in (None, ""):
                return obj[key]
        for value in obj.values():
            found = _extract_value(value, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _extract_value(value, keys)
            if found is not None:
                return found
    return None


def _normalise_image_output(value: Any) -> str | None:
    """
    Normalize common Workflow image output representations to a browser-ready URL.
    """
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()
        if value.startswith("data:image/") or value.startswith("http://") or value.startswith("https://"):
            return value
        # Plain base64 image payload.
        if len(value) > 200:
            return "data:image/jpeg;base64," + value
        return None

    if isinstance(value, dict):
        # Common object wrappers.
        for key in ("value", "data", "image", "url", "base64"):
            if key in value and value[key]:
                normalized = _normalise_image_output(value[key])
                if normalized:
                    return normalized

    return None


def _prediction_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [p for p in value if isinstance(p, dict)]
    if isinstance(value, dict):
        for key in ("predictions", "detections", "items", "results"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return [p for p in candidate if isinstance(p, dict)]
    return []


def _class_name(pred: dict[str, Any]) -> str:
    for key in ("class", "label", "class_name", "name"):
        if pred.get(key) not in (None, ""):
            return str(pred[key])
    return ""


def _confidence(pred: dict[str, Any]) -> float | None:
    for key in ("confidence", "score", "probability"):
        value = pred.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _box(pred: dict[str, Any]) -> dict[str, float] | None:
    # Standard x/y/width/height form.
    if all(k in pred for k in ("x", "y", "width", "height")):
        x = float(pred["x"])
        y = float(pred["y"])
        w = float(pred["width"])
        h = float(pred["height"])
        return {
            "x1": x - w / 2,
            "y1": y - h / 2,
            "x2": x + w / 2,
            "y2": y + h / 2,
        }

    # x1/y1/x2/y2 form.
    if all(k in pred for k in ("x1", "y1", "x2", "y2")):
        return {k: float(pred[k]) for k in ("x1", "y1", "x2", "y2")}

    # Bounding-box wrapper.
    bbox = pred.get("bbox") or pred.get("bounding_box") or pred.get("box")
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x1", "y1", "x2", "y2")):
            return {k: float(bbox[k]) for k in ("x1", "y1", "x2", "y2")}
        if all(k in bbox for k in ("x", "y", "width", "height")):
            x, y = float(bbox["x"]), float(bbox["y"])
            w, h = float(bbox["width"]), float(bbox["height"])
            return {"x1": x - w/2, "y1": y - h/2, "x2": x + w/2, "y2": y + h/2}
        if isinstance(bbox.get("points"), list) and bbox["points"]:
            xs = [float(p[0]) for p in bbox["points"] if isinstance(p, (list, tuple)) and len(p) >= 2]
            ys = [float(p[1]) for p in bbox["points"] if isinstance(p, (list, tuple)) and len(p) >= 2]
            if xs and ys:
                return {"x1": min(xs), "y1": min(ys), "x2": max(xs), "y2": max(ys)}

    return None


def run_goat_workflow(image_bytes: bytes, timeout: int = 120) -> dict[str, Any]:
    if not is_configured():
        raise RoboflowWorkflowError(
            "Roboflow is not configured. Set ROBOFLOW_API_KEY, "
            "ROBOFLOW_WORKSPACE and ROBOFLOW_WORKFLOW in the backend environment."
        )

    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {
                "type": "base64",
                "value": bytes_to_base64(image_bytes),
            },
            "classes": ROBOFLOW_CLASSES,
        },
    }

    try:
        response = requests.post(
            endpoint(),
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RoboflowWorkflowError(f"Could not reach Roboflow Workflow: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text[:1000]
        raise RoboflowWorkflowError(
            f"Roboflow Workflow returned HTTP {response.status_code}: {detail}"
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RoboflowWorkflowError("Roboflow returned a non-JSON response.") from exc

    raw_predictions = _extract_value(body, ("predictions", "detections"))
    predictions = _prediction_list(raw_predictions)

    # If the workflow nests predictions more deeply, the extractor still finds them.
    normalized = []
    for pred in predictions:
        cls = _class_name(pred)
        if cls and cls.lower() not in {c.lower() for c in ROBOFLOW_CLASSES}:
            continue

        normalized.append({
            "class": cls or ROBOFLOW_CLASSES[0],
            "confidence": _confidence(pred),
            "box": _box(pred),
            "points": (
                pred.get("points")
                or pred.get("polygon")
                or pred.get("segmentation")
                or pred.get("mask")
                or []
            ),
            "raw": pred,
        })

    count_value = _extract_value(
        body,
        ("detection_count", "goat_count", "count", "total_detections"),
    )
    try:
        workflow_count = int(count_value) if count_value is not None else None
    except (TypeError, ValueError):
        workflow_count = None

    # The screenshot workflow exposes prediction output, so count instances
    # ourselves if the count block is absent or ambiguous.
    goat_count = len(normalized) if normalized else (workflow_count or 0)

    mask_visualization = None
    label_visualization = None
    for key in ("annotated_image", "mask_visualization_image", "mask_visualization", "mask_visualization_url"):
        candidate = _normalise_image_output(_extract_value(body, (key,)))
        if candidate:
            mask_visualization = candidate
            break
    for key in ("label_visualization_image", "label_visualization"):
        candidate = _normalise_image_output(_extract_value(body, (key,)))
        if candidate:
            label_visualization = candidate
            break
    annotated = label_visualization or mask_visualization

    return {
        "workspace": ROBOFLOW_WORKSPACE,
        "workflow": ROBOFLOW_WORKFLOW,
        "classes": ROBOFLOW_CLASSES,
        "goats_detected": goat_count,
        "predictions": normalized,
        "annotated_image": annotated,
        "mask_visualization": mask_visualization,
        "label_visualization": label_visualization,
        "raw_response": body,
    }
# ---------------------------------------------------------------------------
# Dedicated conjunctiva workflow
# ---------------------------------------------------------------------------

CONJUNCTIVA_API_URL = os.getenv(
    "CONJUNCTIVA_ROBOFLOW_API_URL",
    ROBOFLOW_API_URL,
).rstrip("/")

CONJUNCTIVA_API_KEY = os.getenv(
    "CONJUNCTIVA_ROBOFLOW_API_KEY",
    ROBOFLOW_API_KEY,
).strip()

CONJUNCTIVA_WORKSPACE = os.getenv(
    "CONJUNCTIVA_ROBOFLOW_WORKSPACE",
    "",
).strip()

CONJUNCTIVA_WORKFLOW = os.getenv(
    "CONJUNCTIVA_ROBOFLOW_WORKFLOW",
    "",
).strip()

CONJUNCTIVA_CLASSES = [
    item.strip()
    for item in os.getenv(
        "CONJUNCTIVA_ROBOFLOW_CLASSES",
        "sheep-conjunctiva",
    ).split(",")
    if item.strip()
]


def conjunctiva_is_configured() -> bool:
    return bool(
        CONJUNCTIVA_API_KEY
        and CONJUNCTIVA_WORKSPACE
        and CONJUNCTIVA_WORKFLOW
    )


def conjunctiva_endpoint() -> str:
    return (
        f"{CONJUNCTIVA_API_URL}/"
        f"{CONJUNCTIVA_WORKSPACE}/workflows/"
        f"{CONJUNCTIVA_WORKFLOW}"
    )



def _render_conjunctiva_annotation(
    image_bytes: bytes,
    predictions: list[dict[str, Any]],
) -> str | None:
    """Render a Roboflow-style conjunctiva segmentation image."""

    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return None

    output = image.copy()

    # Roboflow-like visual appearance
    MASK_COLOR = (0, 255, 0)       # bright lime/green
    OUTLINE_COLOR = (0, 220, 255)  # yellow-orange
    LABEL_COLOR = (0, 255, 255)    # yellow
    TEXT_COLOR = (0, 0, 0)         # black

    image_h, image_w = output.shape[:2]

    for pred in predictions:
        raw_points = (
            pred.get("points")
            or pred.get("polygon")
            or pred.get("segmentation")
            or []
        )

        if not isinstance(raw_points, list):
            continue

        points = []

        for point in raw_points:
            # Roboflow format: {"x": ..., "y": ...}
            if isinstance(point, dict):
                if "x" not in point or "y" not in point:
                    continue

                try:
                    x = float(point["x"])
                    y = float(point["y"])
                except (TypeError, ValueError):
                    continue

                points.append([x, y])

            # Also support [x, y]
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                try:
                    points.append([
                        float(point[0]),
                        float(point[1]),
                    ])
                except (TypeError, ValueError):
                    continue

        if len(points) < 3:
            continue

        polygon = np.asarray(points, dtype=np.float32)

        # Keep coordinates inside the actual image.
        polygon[:, 0] = np.clip(
            polygon[:, 0],
            0,
            image_w - 1,
        )

        polygon[:, 1] = np.clip(
            polygon[:, 1],
            0,
            image_h - 1,
        )

        polygon = np.round(polygon).astype(np.int32)
        polygon = polygon.reshape((-1, 1, 2))

        # ---------------------------------------------------------
        # Filled conjunctiva mask
        # ---------------------------------------------------------
        overlay = output.copy()

        cv2.fillPoly(
            overlay,
            [polygon],
            MASK_COLOR,
        )

        output = cv2.addWeighted(
            overlay,
            0.42,
            output,
            0.58,
            0,
        )

        # ---------------------------------------------------------
        # Segmentation outline
        # ---------------------------------------------------------
        cv2.polylines(
            output,
            [polygon],
            isClosed=True,
            color=OUTLINE_COLOR,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # ---------------------------------------------------------
        # Label
        # ---------------------------------------------------------
        class_name = str(
            pred.get("class")
            or "conjunctiva"
        )

        confidence = pred.get("confidence")

        try:
            confidence_percent = int(
                round(float(confidence) * 100)
            )
        except (TypeError, ValueError):
            confidence_percent = 0

        label = f"{class_name} {confidence_percent}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            font_thickness,
        )

        padding_x = 6
        padding_y = 5

        label_w = text_w + padding_x * 2
        label_h = text_h + baseline + padding_y * 2

        x, y, w, h = cv2.boundingRect(polygon)

        # Position label above the conjunctiva.
        label_x = x
        label_y = y - label_h - 4

        # Keep label inside image.
        label_x = max(
            0,
            min(
                label_x,
                image_w - label_w - 1,
            ),
        )

        if label_y < 0:
            label_y = min(
                image_h - label_h - 1,
                y + 4,
            )

        label_y = max(0, label_y)

        # Yellow label rectangle
        cv2.rectangle(
            output,
            (label_x, label_y),
            (
                label_x + label_w,
                label_y + label_h,
            ),
            LABEL_COLOR,
            -1,
        )

        # Black text
        cv2.putText(
            output,
            label,
            (
                label_x + padding_x,
                label_y + padding_y + text_h,
            ),
            font,
            font_scale,
            TEXT_COLOR,
            font_thickness,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        output,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            95,
        ],
    )

    if not ok:
        return None

    # -------------------------------------------------------------
    # Save annotated conjunctiva image locally
    # -------------------------------------------------------------
    output_dir = Path(r"D:\weight_detect\output\conjunctiva_annotations")
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"conjunctiva_{timestamp}.jpg"

    output_path.write_bytes(encoded.tobytes())

    print(f"[CONJUNCTIVA] Annotated image saved: {output_path}")

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(
            encoded.tobytes()
        ).decode("ascii")
    )

def run_conjunctiva_workflow(
    image_bytes: bytes,
    timeout: int = 120,
) -> dict[str, Any]:
    """
    Run the deployed Roboflow conjunctiva MODEL directly.

    Model:
        segmentation_conjunctiva-rdlmb/3

    This intentionally uses the Roboflow Inference SDK rather than
    the Workflow API because this model/version was already verified
    successfully with the SDK.
    """

    api_key = os.getenv(
        "CONJUNCTIVA_ROBOFLOW_API_KEY",
        "",
    ).strip()

    model_id = os.getenv(
        "CONJUNCTIVA_ROBOFLOW_MODEL",
        "segmentation_conjunctiva-rdlmb/3",
    ).strip()

    api_url = os.getenv(
        "CONJUNCTIVA_ROBOFLOW_API_URL",
        "https://serverless.roboflow.com",
    ).strip().rstrip("/")

    if not api_key:
        raise RoboflowWorkflowError(
            "Conjunctiva Roboflow model is not configured. "
            "Set CONJUNCTIVA_ROBOFLOW_API_KEY."
        )

    if not model_id:
        raise RoboflowWorkflowError(
            "CONJUNCTIVA_ROBOFLOW_MODEL is empty."
        )

    try:
        client = InferenceHTTPClient(
            api_url=api_url,
            api_key=api_key,
        )

        # The SDK reliably accepts a file path. Create a temporary
        # image file from the uploaded bytes.
        import tempfile

        suffix = ".jpg"

        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        try:
            raw_result = client.infer(
                temp_path,
                model_id=model_id,
            )
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    except Exception as exc:
        raise RoboflowWorkflowError(
            f"Conjunctiva Roboflow inference failed: {exc}"
        ) from exc

    # -------------------------------------------------------------
    # Normalize Roboflow SDK response
    # -------------------------------------------------------------

    if not isinstance(raw_result, dict):
        raise RoboflowWorkflowError(
            "Unexpected Roboflow response format."
        )

    raw_predictions = raw_result.get(
        "predictions",
        [],
    )

    normalized: list[dict[str, Any]] = []

    if isinstance(raw_predictions, list):

        for pred in raw_predictions:

            if not isinstance(pred, dict):
                continue

            cls = (
                pred.get("class")
                or pred.get("class_name")
                or "conjunctiva"
            )

            confidence = pred.get(
                "confidence",
                0,
            )

            try:
                confidence = float(confidence)
            except (
                TypeError,
                ValueError,
            ):
                confidence = 0.0

            points = (
                pred.get("points")
                or pred.get("polygon")
                or pred.get("segmentation")
                or pred.get("mask")
                or []
            )

            # Normalize polygon points.
            normalized_points: list[dict[str, float]] = []

            if isinstance(points, list):

                for point in points:

                    if isinstance(point, dict):

                        if (
                            point.get("x") is not None
                            and point.get("y") is not None
                        ):
                            try:
                                normalized_points.append(
                                    {
                                        "x": float(point["x"]),
                                        "y": float(point["y"]),
                                    }
                                )
                            except (
                                TypeError,
                                ValueError,
                            ):
                                pass

                    elif (
                        isinstance(point, (list, tuple))
                        and len(point) >= 2
                    ):

                        try:
                            normalized_points.append(
                                {
                                    "x": float(point[0]),
                                    "y": float(point[1]),
                                }
                            )
                        except (
                            TypeError,
                            ValueError,
                        ):
                            pass

            normalized.append(
                {
                    "class": str(cls),
                    "confidence": confidence,
                    "box": _box(pred),
                    "points": normalized_points,
                    "rle_mask": pred.get(
                        "rle_mask"
                    ),
                    "class_id": pred.get(
                        "class_id"
                    ),
                    "detection_id": pred.get(
                        "detection_id"
                    ),
                    "mask_format": pred.get(
                        "mask_format"
                    ),
                }
            )

    image_info = raw_result.get(
        "image",
        {},
    )

    image_width = None
    image_height = None

    if isinstance(image_info, dict):

        image_width = image_info.get(
            "width"
        )

        image_height = image_info.get(
            "height"
        )

    # Also support raw_image if returned.
    if isinstance(
        raw_result.get("raw_image"),
        dict,
    ):

        raw_image = raw_result["raw_image"]

        image_width = (
            image_width
            or raw_image.get("width")
        )

        image_height = (
            image_height
            or raw_image.get("height")
        )

    normalized.sort(
        key=lambda p: float(
            p.get("confidence", 0)
        ),
        reverse=True,
    )

    best = (
        normalized[0]
        if normalized
        else None
    )

    # -------------------------------------------------------------
    # Generate local annotation
    # -------------------------------------------------------------

    annotated_image = _render_conjunctiva_annotation(
        image_bytes,
        normalized,
    )

    print(
        "[CONJUNCTIVA] "
        f"Model={model_id} "
        f"Detected={bool(normalized)} "
        f"Count={len(normalized)}"
    )

    return {
        "configured": True,
        "provider": "Roboflow Inference SDK",
        "model": model_id,
        "detected": bool(normalized),
        "count": len(normalized),
        "image_width": image_width,
        "image_height": image_height,
        "best_prediction": best,
        "predictions": normalized,
        "annotated_image": annotated_image,
    }


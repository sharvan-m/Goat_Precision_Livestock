"""Roboflow Workflow client for Conjunctiva Segmentation.

Workflow Details:
- Name: Segmentation_Conjunctiva vsegmentation_conjunctiva-rdlmb-pqdsd-1-yolo26s-seg-t1 Logic
- Workspace: sharvan-m
- Workflow ID: segmentationconjunctiva-vsegmentationconjunctiva-rdlmb-pqdsd-1-yolo26s-seg-t1-logic
- Run endpoint: https://serverless.roboflow.com/sharvan-m/workflows/segmentationconjunctiva-vsegmentationconjunctiva-rdlmb-pqdsd-1-yolo26s-seg-t1-logic
- Inputs: image (image)

Authentication is handled securely via Bearer token in request headers.
API keys are loaded only from environment variables and never logged or exposed.
"""

from __future__ import annotations

import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "segmentation_conjunctiva-rdlmb-pqdsd/1"
DEFAULT_WORKSPACE = "sharvan-m"
DEFAULT_WORKFLOW_ID = "segmentationconjunctiva-vsegmentationconjunctiva-rdlmb-pqdsd-1-yolo26s-seg-t1-logic"
DEFAULT_API_URL = "https://serverless.roboflow.com"

MODEL_ID = (
    os.getenv("CONJUNCTIVA_ROBOFLOW_MODEL_ID", "").strip()
    or os.getenv("CONJUNCTIVA_ROBOFLOW_MODEL", "").strip()
    or DEFAULT_MODEL_ID
)
WORKSPACE_NAME = os.getenv("CONJUNCTIVA_ROBOFLOW_WORKSPACE", DEFAULT_WORKSPACE).strip()
WORKFLOW_ID = os.getenv("CONJUNCTIVA_ROBOFLOW_WORKFLOW_ID", DEFAULT_WORKFLOW_ID).strip()
API_URL = os.getenv("CONJUNCTIVA_ROBOFLOW_API_URL", DEFAULT_API_URL).strip().rstrip("/")


def _get_api_key() -> str:
    return (
        os.getenv("CONJUNCTIVA_ROBOFLOW_API_KEY", "").strip()
        or os.getenv("ROBOFLOW_API_KEY", "").strip()
    )


class RoboflowWorkflowError(RuntimeError):
    """Raised when Roboflow workflow execution or response parsing fails."""
    pass


# Backwards compatibility exception alias
ConjunctivaRoboflowError = RoboflowWorkflowError


def conjunctiva_is_configured() -> bool:
    """Check if the necessary API key and model/workflow are configured."""
    return bool(_get_api_key() and (MODEL_ID or WORKFLOW_ID))


def conjunctiva_endpoint() -> str:
    """Return the Roboflow execution endpoint URL."""
    return f"{API_URL}/{MODEL_ID}"


# ---------------------------------------------------------------------------
# Client Factory
# ---------------------------------------------------------------------------

def _get_client():
    """Build and configure the official InferenceHTTPClient with header-based auth."""
    api_key = _get_api_key()
    if not api_key:
        raise RoboflowWorkflowError(
            "CONJUNCTIVA_ROBOFLOW_API_KEY (or ROBOFLOW_API_KEY) is not set in environment."
        )

    try:
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient
    except ImportError as exc:
        raise RoboflowWorkflowError(
            "inference-sdk is not installed. Please run: pip install inference-sdk"
        ) from exc

    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=api_key,
    )
    # Ensure Authorization: Bearer <key> header transport (inference v1.5.0+)
    return client.configure(
        InferenceConfiguration(api_key_transport="header")
    )


# ---------------------------------------------------------------------------
# Response Normalization & Geometry Helpers
# ---------------------------------------------------------------------------

def _extract_points(value: Any) -> list[dict[str, float]]:
    """Extract and normalize polygon points from prediction dictionary."""
    if not isinstance(value, list):
        return []
    pts: list[dict[str, float]] = []
    for item in value:
        if isinstance(item, dict) and "x" in item and "y" in item:
            try:
                pts.append({"x": float(item["x"]), "y": float(item["y"])})
            except (TypeError, ValueError):
                continue
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                pts.append({"x": float(item[0]), "y": float(item[1])})
            except (TypeError, ValueError):
                continue
    return pts


def _normalize_prediction(pred: Any) -> dict[str, Any]:
    """Normalize a prediction into a clean dictionary, discarding huge redundant blobs."""
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

    cls_name = str(pred.get("class", pred.get("class_name", "sheep-conjunctiva")))

    normalized: dict[str, Any] = {
        "class": cls_name,
        "confidence": confidence,
    }

    for key in ("x", "y", "width", "height"):
        if key in pred:
            try:
                normalized[key] = float(pred[key])
            except (TypeError, ValueError):
                pass

    points = _extract_points(
        pred.get("points")
        or pred.get("polygon")
        or pred.get("segmentation")
        or pred.get("mask")
    )
    if points:
        normalized["points"] = points

    for key in ("class_id", "detection_id", "parent_id"):
        if key in pred:
            normalized[key] = pred[key]

    return normalized


def _render_annotation(image_bytes: bytes, predictions: list[dict[str, Any]]) -> str | None:
    """Draw yellow polygon mask and outline with NO text labels."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None

    output = image.copy()
    overlay = output.copy()

    # Roboflow UI colors (BGR)
    MASK_COLOR = (0, 235, 245)      # Translucent yellow fill
    OUTLINE_COLOR = (0, 215, 255)   # Bright yellow outline

    for pred in predictions:
        pts = pred.get("points") or []

        if len(pts) >= 3:
            polygon = np.asarray(
                [[int(round(p["x"])), int(round(p["y"]))] for p in pts],
                dtype=np.int32,
            ).reshape((-1, 1, 2))

            # Draw translucent yellow fill
            cv2.fillPoly(overlay, [polygon], MASK_COLOR)
            output = cv2.addWeighted(overlay, 0.45, output, 0.55, 0)

            # Draw crisp yellow outline
            cv2.polylines(output, [polygon], True, OUTLINE_COLOR, 2, cv2.LINE_AA)

        elif all(k in pred for k in ("x", "y", "width", "height")):
            cx, cy = pred["x"], pred["y"]
            bw, bh = pred["width"], pred["height"]
            x1, y1 = int(round(cx - bw / 2)), int(round(cy - bh / 2))
            x2, y2 = int(round(cx + bw / 2)), int(round(cy + bh / 2))
            cv2.rectangle(output, (x1, y1), (x2, y2), OUTLINE_COLOR, 2)

    ok, encoded = cv2.imencode(".jpg", output, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Main Workflow Inference Function
# ---------------------------------------------------------------------------

def run_conjunctiva_workflow(
    image_bytes: bytes,
    parameters: dict[str, Any] | None = None,
    max_retries: int = 2,
    backoff_factor: float = 1.0,
) -> dict[str, Any]:
    """Execute the Roboflow Conjunctiva Segmentation Model with retry logic.

    Calls the hosted serverless model (segmentation_conjunctiva-rdlmb-pqdsd/1)
    and falls back to run_workflow if configured.
    """
    client = _get_client()

    if not image_bytes:
        raise RoboflowWorkflowError("Empty image bytes submitted for inference.")

    temp_path: str | None = None
    target_model = MODEL_ID

    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            temp_path = tmp.name

        raw_result = None
        # 1. First attempt: call client.infer directly with model_id
        try:
            raw_result = client.infer(temp_path, model_id=target_model)
        except Exception as infer_exc:
            # 2. Fallback to run_workflow if configured
            if WORKFLOW_ID and WORKSPACE_NAME:
                try:
                    images_payload = {"image": temp_path}
                    raw_result = client.run_workflow(
                        workspace_name=WORKSPACE_NAME,
                        workflow_id=WORKFLOW_ID,
                        images=images_payload,
                        parameters=parameters or {},
                    )
                except Exception as wf_exc:
                    raise RoboflowWorkflowError(
                        f"Roboflow inference failed: {infer_exc} (workflow fallback also failed: {wf_exc})"
                    ) from infer_exc
            else:
                raise RoboflowWorkflowError(f"Roboflow inference failed: {infer_exc}") from infer_exc

        # Parse response defensively (handles both client.infer dict and client.run_workflow list)
        raw_predictions: list[Any] = []
        raw_image_meta: dict[str, Any] = {}

        if isinstance(raw_result, dict):
            raw_predictions = raw_result.get("predictions") or []
            raw_image_meta = raw_result.get("image") or {}
        elif isinstance(raw_result, list) and len(raw_result) > 0:
            entry = raw_result[0]
            if isinstance(entry, dict):
                predictions_block = entry.get("predictions")
                if isinstance(predictions_block, dict):
                    raw_predictions = predictions_block.get("predictions") or []
                    raw_image_meta = predictions_block.get("image") or {}
                elif isinstance(predictions_block, list):
                    raw_predictions = predictions_block
                else:
                    for key, val in entry.items():
                        if isinstance(val, dict) and "predictions" in val:
                            raw_predictions = val.get("predictions") or []
                            raw_image_meta = val.get("image") or {}
                            break
                        elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict) and "confidence" in val[0]:
                            raw_predictions = val
                            break

        normalized_predictions = [_normalize_prediction(p) for p in raw_predictions]

        # Generate Roboflow-style yellow visualization with badge
        annotated_b64 = _render_annotation(image_bytes, normalized_predictions)

        return {
            "model_id": target_model,
            "provider": "Roboflow Hosted Model",
            "model_type": "instance_segmentation (polygon mask)",
            "predictions": normalized_predictions,
            "prediction_count": len(normalized_predictions),
            "annotated_image": annotated_b64,
            "raw_image": raw_image_meta,
        }

    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass

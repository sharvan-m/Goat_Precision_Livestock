"""Local Cow AI inference engine.

Uses the project's trained checkpoints without modifying them.
The backend is intentionally stateless at API level; models are cached in
process and inference runs only when an endpoint is called.
"""
from __future__ import annotations

import base64
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models as tv_models, transforms

try:
    from pycocotools import mask as coco_mask
except Exception:
    coco_mask = None

try:
    from inference_sdk import (
        InferenceHTTPClient,
        InferenceConfiguration,
    )
    ROBOFLOW_AVAILABLE = True
except Exception:
    InferenceHTTPClient = None
    InferenceConfiguration = None
    ROBOFLOW_AVAILABLE = False

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from transformers import (
        AutoImageProcessor,
        AutoModelForDepthEstimation,
        Sam2Processor,
        Sam2Model,
    )
except Exception:
    AutoImageProcessor = None
    AutoModelForDepthEstimation = None
    Sam2Processor = None
    Sam2Model = None

try:
    from skimage.color import rgb2gray as skimage_rgb2gray
    from skimage.feature import canny as skimage_canny
    SKIMAGE_AVAILABLE = True
except Exception:
    skimage_rgb2gray = None
    skimage_canny = None
    SKIMAGE_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAMES = {
    "detector": "weights.pt",
    "angle": "cow_angle_classifier.pt",
    "angle_binary": "best_angle_binary_classifier.pt",
    "weight": "fusion_v6_efficientnet_yolo.pt",
    "bcs": "bcs_model.pt",
    "bcs_meta": "bcs_model_meta.json",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEPTH_MODEL_ID = os.getenv("DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf")
SAM2_MODEL_ID = os.getenv("SAM2_MODEL_ID", "facebook/sam2.1-hiera-small")
GENERIC_DETECTOR_MODEL = os.getenv("GENERIC_DETECTOR_MODEL", "yolo11n-seg.pt")

# Never promote ultra-low-confidence tiled detections into real cows.
# Those were responsible for decorative wall cows / background shapes being
# returned as 0-1% detections. The project detector remains primary; the
# generic COCO cow-segmentation model is only a fallback when the project
# detector returns no reliable cow.
MIN_VALID_DET_CONF = float(os.getenv("MIN_VALID_DET_CONF", "0.20"))
MIN_VALID_MASK_FRACTION = float(os.getenv("MIN_VALID_MASK_FRACTION", "0.003"))

# The trained 5-class angle model is intentionally conservative here.
# Class 0 is the approved side-on view; low-confidence argmax predictions
# are treated as BAD VIEW instead of being allowed to unlock weight/BCS.
ANGLE_GOOD_MIN_CONF = float(os.getenv("ANGLE_GOOD_MIN_CONF", "0.40"))

# In-process model cache. This means starting the backend does not perform
# inference; models load on first use of each corresponding capability.
_CACHE: dict[str, Any] = {}


ProgressCallback = Callable[[int, str], None]


def _emit_progress(
    callback: ProgressCallback | None,
    percent: int,
    message: str,
) -> None:
    """Send a bounded progress update without letting UI updates break inference."""
    if callback is None:
        return
    try:
        callback(max(0, min(100, int(percent))), str(message))
    except Exception:
        pass

ROBOFLOW_WORKSPACE = os.getenv(
    "ROBOFLOW_WORKSPACE",
    "sharvan-m",
)

ROBOFLOW_WORKFLOW = os.getenv(
    "ROBOFLOW_WORKFLOW",
    "general-segmentation-api-6",
)

ROBOFLOW_CLASSES = os.getenv(
    "ROBOFLOW_CLASSES",
    "cow",
)


# Roboflow side-view model deployed by the project.
ROBOFLOW_SIDE_MODEL = os.getenv(
    "ROBOFLOW_SIDE_MODEL",
    "cow-side-view/9",
)

# Roboflow behavior workflow deployed by the project.
ROBOFLOW_BEHAVIOR_WORKFLOW = os.getenv(
    "ROBOFLOW_BEHAVIOR_WORKFLOW",
    "general-segmentation-api-14",
)

# Exact class prompts used by the behavior workflow.
ROBOFLOW_BEHAVIOR_CLASSES = (
    "cow_sitting",
    "cow_eating",
    "cow_standing",
)

# Minimum confidence for the side-view gate.
SIDE_VIEW_MIN_CONF = float(
    os.getenv("SIDE_VIEW_MIN_CONF", "0.50")
)

# Shared side-view annotation layer used by both photos and videos.
# These are expressed in RGB order for image arrays and are also valid
# for video output because green/red are symmetric under RGB/BGR reversal.
SIDE_ANNOTATION_COLOR = (55, 220, 90)
NOT_SIDE_ANNOTATION_COLOR = (235, 65, 65)
SIDE_ANNOTATION_ALPHA = float(
    os.getenv("SIDE_ANNOTATION_ALPHA", "0.38")
)

# Video side-view checks are intentionally less frequent than SAM 3 calls.
# A new track is checked immediately; existing tracks are rechecked on this
# frame interval and keep their last classification in between checks.
VIDEO_SIDE_VIEW_EVERY = max(
    1,
    int(os.getenv("VIDEO_SIDE_VIEW_EVERY", "30")),
)

# Minimum overlap used when associating behavior predictions with a
# Roboflow SAM 3 cow.
BEHAVIOR_MATCH_IOU = float(
    os.getenv("BEHAVIOR_MATCH_IOU", "0.10")
)


def load_roboflow_client():
    if "roboflow" in _CACHE:
        return _CACHE["roboflow"]

    if not ROBOFLOW_AVAILABLE:
        raise RuntimeError(
            "inference-sdk is not installed. "
            "Run: python -m pip install inference-sdk"
        )

    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set."
        )

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    ).configure(
        InferenceConfiguration(
            api_key_transport="header",
        )
    )

    _CACHE["roboflow"] = client
    return client


def run_roboflow_cow_workflow(
    image_bytes: bytes,
    suffix: str = ".jpg",
) -> tuple[bytes, dict[str, Any]]:

    client = load_roboflow_client()

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix or ".jpg",
    ) as f:
        f.write(image_bytes)
        image_path = f.name

    try:
        result = client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE,
            workflow_id=ROBOFLOW_WORKFLOW,
            images={
                "image": image_path,
            },
            parameters={
                "classes": ROBOFLOW_CLASSES,
            },
            use_cache=True,
        )
    finally:
        try:
            os.unlink(image_path)
        except OSError:
            pass

    if not isinstance(result, list) or not result:
        raise RuntimeError(
            "Roboflow workflow returned an empty result."
        )

    item = result[0]

    if not isinstance(item, dict):
        raise RuntimeError(
            "Unexpected Roboflow workflow response."
        )

    annotated_image = item.get("annotated_image")

    if not annotated_image:
        raise RuntimeError(
            "Roboflow workflow did not return annotated_image."
        )

    if isinstance(annotated_image, str):
        payload = annotated_image

        if payload.startswith("data:image"):
            payload = payload.split(",", 1)[1]

        try:
            annotated_bytes = base64.b64decode(payload)
        except Exception as exc:
            raise RuntimeError(
                f"Could not decode Roboflow annotated image: {exc}"
            ) from exc
    else:
        raise RuntimeError(
            "Roboflow annotated_image has an unsupported type."
        )

    return annotated_bytes, item


def _decode_roboflow_rle(
    rle_mask: dict[str, Any],
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    """Decode a Roboflow COCO-style RLE mask to the target image size."""
    if coco_mask is None:
        return None
    if not isinstance(rle_mask, dict):
        return None

    size = rle_mask.get("size")
    counts = rle_mask.get("counts")
    if not size or counts is None:
        return None

    try:
        decoded = coco_mask.decode({
            "size": [int(size[0]), int(size[1])],
            "counts": counts,
        })
    except Exception:
        return None

    if decoded.ndim == 3:
        decoded = decoded[:, :, 0]

    decoded = (decoded > 0).astype(np.uint8)
    target_h, target_w = target_shape

    if decoded.shape != (target_h, target_w):
        decoded = cv2.resize(
            decoded,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return decoded.astype(bool)


def _prediction_box(
    prediction: dict[str, Any],
    src_width: int,
    src_height: int,
    rf_width: float,
    rf_height: float,
) -> list[int]:
    """Convert Roboflow x/y/width/height to original image coordinates."""
    px = float(prediction.get("x", 0.0))
    py = float(prediction.get("y", 0.0))
    pw = float(prediction.get("width", 0.0))
    ph = float(prediction.get("height", 0.0))

    sx = src_width / max(rf_width, 1.0)
    sy = src_height / max(rf_height, 1.0)

    x1 = int(round((px - pw / 2.0) * sx))
    y1 = int(round((py - ph / 2.0) * sy))
    x2 = int(round((px + pw / 2.0) * sx))
    y2 = int(round((py + ph / 2.0) * sy))

    return [
        max(0, min(src_width, x1)),
        max(0, min(src_height, y1)),
        max(0, min(src_width, x2)),
        max(0, min(src_height, y2)),
    ]


def _box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter

    return inter / max(union, 1e-6)


def _box_center_inside(inner_box, outer_box) -> bool:
    x1, y1, x2, y2 = map(float, inner_box)
    ox1, oy1, ox2, oy2 = map(float, outer_box)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def _extract_predictions(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        if result and isinstance(result[0], dict):
            first = result[0]
            if isinstance(first.get("predictions"), dict):
                preds = first["predictions"].get("predictions", [])
                return preds if isinstance(preds, list) else []
            if isinstance(first.get("predictions"), list):
                return first["predictions"]
        return [x for x in result if isinstance(x, dict)]

    if isinstance(result, dict):
        preds = result.get("predictions", [])
        if isinstance(preds, dict):
            preds = preds.get("predictions", [])
        return preds if isinstance(preds, list) else []

    return []


def _normalize_side_label(label: str) -> str:
    return " ".join(
        str(label or "")
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def _is_explicit_not_side_label(label: str) -> bool:
    normalized = _normalize_side_label(label)
    return normalized in {
        "not side",
        "not side view",
        "non side",
        "non side view",
        "no side",
        "not a side view",
    } or normalized.startswith("not side ")


def _is_side_label(label: str) -> bool:
    normalized = _normalize_side_label(label)
    if not normalized or _is_explicit_not_side_label(normalized):
        return False
    return (
        normalized == "side"
        or normalized == "side view"
        or normalized.startswith("side view ")
        or normalized.endswith(" side")
    )


def run_side_view_model(
    client,
    crop_rgb_image: np.ndarray,
) -> tuple[bool, float, str]:
    """Run the deployed cow-side-view/9 model on one cow crop."""
    if crop_rgb_image is None or crop_rgb_image.size == 0:
        return False, 0.0, "NOT SIDE"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg",
    ) as f:
        ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(crop_rgb_image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        if not ok:
            raise RuntimeError("Could not encode side-view crop")
        f.write(encoded.tobytes())
        crop_path = f.name

    try:
        result = client.infer(
            crop_path,
            model_id=ROBOFLOW_SIDE_MODEL,
        )
    finally:
        try:
            os.unlink(crop_path)
        except OSError:
            pass

    predictions = _extract_predictions(result)
    best_conf = 0.0
    best_label = "NOT SIDE"

    # Prefer an explicit NOT-SIDE prediction over a substring match.
    # This avoids the classic bug where "not_side" contains the word "side"
    # and could otherwise be incorrectly marked as SIDE.
    for pred in predictions:
        raw_label = str(pred.get("class", "")).strip()
        normalized = _normalize_side_label(raw_label)
        conf = float(pred.get("confidence", 0.0) or 0.0)

        if _is_explicit_not_side_label(normalized):
            if conf >= best_conf:
                best_conf = conf
                best_label = "NOT SIDE"
        elif _is_side_label(normalized):
            if conf > best_conf:
                best_conf = conf
                best_label = "SIDE"

    side_ok = best_label == "SIDE" and best_conf >= SIDE_VIEW_MIN_CONF

    return (
        side_ok,
        best_conf,
        "SIDE" if side_ok else "NOT SIDE",
    )


def render_side_annotation(
    target: np.ndarray,
    mask: np.ndarray | None,
    side_ok: bool,
) -> None:
    """Shared visual layer: green for SIDE, red for NOT SIDE."""
    if mask is None or not np.asarray(mask).any():
        return

    mask_bool = np.asarray(mask).astype(bool)
    color = SIDE_ANNOTATION_COLOR if side_ok else NOT_SIDE_ANNOTATION_COLOR
    alpha = max(0.0, min(1.0, SIDE_ANNOTATION_ALPHA))

    overlay = np.zeros_like(target)
    overlay[mask_bool] = color

    target[mask_bool] = (
        target[mask_bool].astype(np.float32) * (1.0 - alpha)
        + overlay[mask_bool].astype(np.float32) * alpha
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_bool.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    thickness = max(
        2,
        int(round(min(target.shape[0], target.shape[1]) / 500)),
    )
    cv2.drawContours(
        target,
        contours,
        -1,
        color,
        thickness,
        cv2.LINE_AA,
    )


def run_behavior_workflow(
    client,
    image_bytes: bytes,
) -> list[dict[str, Any]]:
    """Run the behavior workflow once per exact behavior class."""
    results: list[dict[str, Any]] = []

    for behavior_class in ROBOFLOW_BEHAVIOR_CLASSES:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as f:
            f.write(image_bytes)
            image_path = f.name

        try:
            result = client.run_workflow(
                workspace_name=ROBOFLOW_WORKSPACE,
                workflow_id=ROBOFLOW_BEHAVIOR_WORKFLOW,
                images={
                    "image": image_path,
                },
                parameters={
                    "classes": behavior_class,
                },
                use_cache=True,
            )
        finally:
            try:
                os.unlink(image_path)
            except OSError:
                pass

        for pred in _extract_predictions(result):
            label = str(pred.get("class", "")).strip().lower()
            conf = float(pred.get("confidence", 0.0) or 0.0)

            if label not in ROBOFLOW_BEHAVIOR_CLASSES:
                # The workflow should return the requested class, but keep a
                # defensive check in case the deployed graph returns aliases.
                continue

            results.append({
                "class": label,
                "confidence": conf,
                "x": pred.get("x"),
                "y": pred.get("y"),
                "width": pred.get("width"),
                "height": pred.get("height"),
                "rle_mask": pred.get("rle_mask"),
            })

    return results


def match_behavior_to_cow(
    cow: dict[str, Any],
    behaviors: list[dict[str, Any]],
    src_width: int,
    src_height: int,
    rf_width: float,
    rf_height: float,
) -> tuple[str, float]:
    """Associate the strongest spatially matching behavior with one cow."""
    best_label = "UNKNOWN"
    best_score = 0.0

    cow_box = cow["box"]

    for behavior in behaviors:
        beh_box = _prediction_box(
            behavior,
            src_width,
            src_height,
            rf_width,
            rf_height,
        )

        iou = _box_iou(cow_box, beh_box)
        center_inside = _box_center_inside(cow_box, beh_box) or _box_center_inside(beh_box, cow_box)
        confidence = float(behavior.get("confidence", 0.0))

        if iou < BEHAVIOR_MATCH_IOU and not center_inside:
            continue

        # Prefer spatial agreement first, then confidence.
        spatial = max(iou, 0.20 if center_inside else 0.0)
        score = spatial * 0.75 + confidence * 0.25

        if score > best_score:
            best_score = score
            best_label = str(behavior.get("class", "UNKNOWN"))

    pretty = {
        "cow_standing": "STANDING",
        "cow_sitting": "SITTING",
        "cow_eating": "EATING",
    }

    return pretty.get(best_label, best_label), best_score


def analyze_roboflow_cow(
    client,
    image_rgb: np.ndarray,
    cow_prediction: dict[str, Any],
    rf_width: float,
    rf_height: float,
) -> dict[str, Any]:
    """Side-view gate, then weight and BCS for one SAM 3 cow."""
    image_h, image_w = image_rgb.shape[:2]

    box_list = _prediction_box(
        cow_prediction,
        image_w,
        image_h,
        rf_width,
        rf_height,
    )
    box = np.asarray(box_list, dtype=np.float32)

    # Use SAM 3's own mask for geometry/BCS.
    mask = _decode_roboflow_rle(
        cow_prediction.get("rle_mask"),
        (image_h, image_w),
    )

    if mask is None:
        mask = np.zeros((image_h, image_w), dtype=bool)
        x1, y1, x2, y2 = box_list
        mask[y1:y2, x1:x2] = True

    crop = crop_rgb(
        image_rgb,
        box,
        pad_ratio=0.08,
    )

    # ---------------- Side-view gate ----------------
    try:
        side_ok, side_conf, side_label = run_side_view_model(
            client,
            crop,
        )
    except Exception as exc:
        side_ok = False
        side_conf = 0.0
        side_label = f"SIDE VIEW ERROR: {exc}"

    quality_score, geometry_quality = side_profile_score(
        mask,
        box,
        image_rgb.shape,
    )

    # No weight/BCS for non-side views.
    if not side_ok:
        return {
            "box": box_list,
            "mask": mask,
            "side_ok": False,
            "view_quality": "NOT SIDE",
            "view_confidence": round(side_conf, 4),
            "geometry_quality": geometry_quality,
            "quality_score": round(float(quality_score), 4),
            "weight_kg": None,
            "weight_engine": "blocked - NOT SIDE VIEW",
            "bcs": None,
            "bcs_confidence": None,
            "bcs_state": "blocked - NOT SIDE VIEW",
        }

    # ---------------- Weight ----------------
    try:
        geom = mask_geometry(
            mask,
            box,
            image_rgb.shape,
            float(cow_prediction.get("confidence", 0.0)),
        )
        weight, weight_engine = predict_weight(
            crop,
            geom,
        )
    except Exception as exc:
        weight = None
        weight_engine = f"weight inference failed: {exc}"

    # ---------------- BCS ----------------
    bcs_result = None
    bcs_state = "UNAVAILABLE"

    try:
        mask_crop = crop_mask(
            mask,
            box,
            pad_ratio=0.08,
        )

        dge, dge_source = rgb_to_dge(
            crop,
            mask_crop,
        )

        if dge is None:
            bcs_state = dge_source
        else:
            bcs_result, bcs_kind = predict_bcs(dge)
            bcs_state = (
                f"{bcs_kind}; source={dge_source}"
                if bcs_result is not None
                else bcs_kind
            )
    except Exception as exc:
        bcs_state = f"BCS inference failed: {exc}"

    return {
        "box": box_list,
        "mask": mask,
        "side_ok": True,
        "view_quality": "SIDE",
        "view_confidence": round(side_conf, 4),
        "geometry_quality": geometry_quality,
        "quality_score": round(float(quality_score), 4),
        "weight_kg": (
            None if weight is None else round(float(weight), 1)
        ),
        "weight_engine": weight_engine,
        "bcs": (
            None
            if bcs_result is None
            else int(bcs_result["bcs"])
        ),
        "bcs_confidence": (
            None
            if bcs_result is None
            else round(float(bcs_result["confidence"]), 4)
        ),
        "bcs_state": bcs_state,
    }

def load_sam2():
    """Load SAM 2 as the mask-refinement model.

    YOLO remains the cow detector. SAM 2 receives each YOLO cow box and
    returns a cleaner whole-object silhouette. The checkpoint is downloaded
    by Hugging Face Transformers on first use when not already cached.
    """
    if "sam2" in _CACHE:
        return _CACHE["sam2"]
    if Sam2Model is None or Sam2Processor is None:
        return None

    try:
        processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
        model = Sam2Model.from_pretrained(SAM2_MODEL_ID)
        model = model.to(DEVICE).eval()
        _CACHE["sam2"] = (processor, model)
        return _CACHE["sam2"]
    except Exception:
        # Keep the application usable with the original YOLO mask if SAM 2
        # cannot be loaded (for example, offline or missing dependencies).
        _CACHE["sam2"] = None
        return None


def refine_mask_with_sam2(image_rgb: np.ndarray, box: np.ndarray, detector_mask: np.ndarray | None = None) -> np.ndarray | None:
    """Use the YOLO cow bounding box to obtain a whole-cow SAM 2 mask."""
    loaded = load_sam2()
    if loaded is None:
        return None

    processor, model = loaded
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]

    # Give SAM2 a small amount of context around the detector box while
    # keeping the prompt tied to the detected cow.
    pad_x = max((x2 - x1) * 0.04, 4.0)
    pad_y = max((y2 - y1) * 0.04, 4.0)
    prompt_box = [
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(w - 1), x2 + pad_x),
        min(float(h - 1), y2 + pad_y),
    ]

    try:
        inputs = processor(
            images=Image.fromarray(image_rgb),
            input_boxes=[[prompt_box]],
            return_tensors="pt",
        )
        inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        masks = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
        )[0]

        # SAM2 normally returns multiple candidates for a box. Prefer the
        # candidate that agrees with the YOLO cow mask, not simply the SAM2
        # candidate with the highest internal IoU score. This prevents SAM2
        # from selecting wall/floor/background regions inside a loose box.
        iou_scores = outputs.iou_scores.detach().cpu().numpy().reshape(-1)
        mask_arr = masks.detach().cpu().numpy()

        if mask_arr.ndim == 4:
            mask_arr = mask_arr[0]
        if mask_arr.ndim != 3 or len(mask_arr) == 0:
            return None

        pred_iou = iou_scores if len(iou_scores) == len(mask_arr) else np.zeros(len(mask_arr), dtype=np.float32)
        if detector_mask is not None and detector_mask.shape == (h, w):
            dm = np.asarray(detector_mask).astype(bool)
            detector_area = max(float(dm.sum()), 1.0)
            scores = []
            for m0, piou in zip(mask_arr, pred_iou):
                candidate = m0.astype(bool)
                inter = float(np.logical_and(candidate, dm).sum())
                union = float(np.logical_or(candidate, dm).sum())
                mask_iou = inter / max(union, 1.0)
                containment = inter / detector_area
                # Predicted IoU is only a secondary term. Agreement with the
                # project detector's actual cow evidence is primary.
                scores.append(0.65 * mask_iou + 0.25 * containment + 0.10 * float(piou))

            best = int(np.argmax(scores))
            refined = mask_arr[best].astype(bool)
            best_inter = float(np.logical_and(refined, dm).sum())
            best_iou = best_inter / max(float(np.logical_or(refined, dm).sum()), 1.0)
            best_containment = best_inter / detector_area
            # If SAM2 is not actually agreeing with the detector mask, do not
            # replace the detector mask with a potentially unrelated region.
            if best_iou < 0.10 and best_containment < 0.25:
                return None
        else:
            best = int(np.argmax(pred_iou))
            refined = mask_arr[best].astype(bool)

        # Fill small holes only. Do NOT apply a morphological OPEN here: on a
        # whole-animal mask the head/neck and each leg are thin connections
        # to the torso, and an OPEN (erode+dilate) reliably severs them. The
        # previous version did that, then kept only the single connected
        # component sitting under the detector-box center -- which silently
        # discarded the now-disconnected head and legs as if they were noise.
        refined_u8 = (refined.astype(np.uint8) * 255)
        kernel = np.ones((5, 5), np.uint8)
        refined_u8 = cv2.morphologyEx(refined_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Drop only genuine speckle noise (tiny stray blobs SAM2 sometimes
        # returns well outside the animal), by area -- keep every component
        # that is a real part of the cow, not just the one under the center.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(refined_u8, connectivity=8)
        if n > 1:
            box_area = max((x2 - x1) * (y2 - y1), 1.0)
            min_area = max(20.0, box_area * 0.0015)
            keep = np.zeros_like(refined_u8)
            for label in range(1, n):
                if stats[label, cv2.CC_STAT_AREA] >= min_area:
                    keep[labels == label] = 255
            if keep.any():
                refined_u8 = keep

        return refined_u8 > 0
    except Exception:
        return None


def _clean_state_dict(sd: Any) -> Any:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def _unwrap_checkpoint(obj: Any) -> Any:
    if isinstance(obj, nn.Module):
        return obj
    if isinstance(obj, dict):
        for k in ("model", "module", "net"):
            if isinstance(obj.get(k), nn.Module):
                return obj[k]
        for k in ("state_dict", "model_state_dict", "weights"):
            if isinstance(obj.get(k), dict):
                return obj[k]
    return obj


def model_exists(key: str) -> bool:
    return (MODEL_DIR / MODEL_NAMES[key]).exists()


def load_detector():
    if "detector" in _CACHE:
        return _CACHE["detector"]
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Run: python -m pip install ultralytics")
    path = MODEL_DIR / MODEL_NAMES["detector"]
    if not path.exists():
        raise FileNotFoundError(f"Cow segmentation model not found: {path}")
    model = YOLO(str(path))
    _CACHE["detector"] = model
    return model

def load_generic_detector():
    """Load a generic COCO cow-segmentation model only as a fallback."""
    if "generic_detector" in _CACHE:
        return _CACHE["generic_detector"]
    if YOLO is None:
        return None
    try:
        model = YOLO(GENERIC_DETECTOR_MODEL)
        _CACHE["generic_detector"] = model
        return model
    except Exception as exc:
        print(f"[generic-detector] unavailable: {exc}")
        _CACHE["generic_detector"] = None
        return None


def _cow_class_ids(detector) -> set[int]:
    names = getattr(getattr(detector, "model", None), "names", None)
    if isinstance(names, dict):
        lowered = {int(k): str(v).strip().lower() for k, v in names.items()}
        ids = {k for k, v in lowered.items() if "cow" in v}
        if ids:
            return ids
        if len(lowered) == 1:
            return set(lowered.keys())
    if isinstance(names, (list, tuple)):
        ids = {i for i, v in enumerate(names) if "cow" in str(v).strip().lower()}
        if ids:
            return ids
        if len(names) == 1:
            return {0}
    # COCO fallback retained for compatibility with a generic detector.
    return {19, 0}


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + ab - inter, 1e-6)


def side_profile_score(mask: np.ndarray, box: np.ndarray, image_shape: tuple[int, ...]) -> tuple[float, str]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    aspect = bw / bh
    fill = float(mask.sum()) / max(bw * bh, 1.0)
    aspect_score = np.clip((aspect - 1.0) / 1.0, 0, 1)
    fill_score = np.clip((fill - 0.15) / 0.45, 0, 1)
    score = 0.60 * aspect_score + 0.40 * fill_score
    if score >= 0.45 and aspect >= 1.18 and fill >= 0.23:
        label = "GOOD"
    elif score >= 0.28:
        label = "PARTIAL"
    else:
        label = "POOR"
    return float(score), label


def _extract_result_detections(
    result,
    source_shape: tuple[int, ...],
    allowed: set[int],
    conf_min: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert one Ultralytics segmentation result into full-image detections."""
    h, w = source_shape[:2]
    if result.masks is None or result.boxes is None:
        return []

    masks = result.masks.data.detach().cpu().numpy()
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)

    dets: list[dict[str, Any]] = []
    for i in range(len(boxes)):
        score = float(confs[i])
        cls = int(classes[i])
        if score < conf_min or cls not in allowed:
            continue

        mask_local = cv2.resize(
            masks[i].astype(np.uint8),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        # For tile inference, the caller pastes this local mask into the
        # corresponding full-image window.
        x1, y1, x2, y2 = boxes[i].astype(np.float32)

        q, qlabel = side_profile_score(
            mask_local,
            np.array([boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]], dtype=np.float32),
            source_shape,
        )
        dets.append({
            "mask_local": mask_local,
            "box": np.array([x1, y1, x2, y2], dtype=np.float32),
            "confidence": score,
            "class_id": cls,
            "quality_score_local": q,
            "quality_local": qlabel,
        })
    return dets


def _dedupe_detections(dets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate full-image detections produced by multiple passes/tiles."""
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: float(d["confidence"]), reverse=True)
    unique: list[dict[str, Any]] = []

    for d in dets:
        db = d["box"]
        dm = d["mask"]
        duplicate = False

        for kept in unique:
            kb = kept["box"]
            km = kept["mask"]
            box_iou = _bbox_iou(db, kb)

            inter = float(np.logical_and(dm, km).sum())
            union = float(np.logical_or(dm, km).sum())
            mask_iou = inter / max(union, 1.0)

            d_area = max(float(dm.sum()), 1.0)
            k_area = max(float(km.sum()), 1.0)
            mask_containment = inter / min(d_area, k_area)

            bx1, by1, bx2, by2 = map(float, db)
            kx1, ky1, kx2, ky2 = map(float, kb)
            inter_w = max(0.0, min(bx2, kx2) - max(bx1, kx1))
            inter_h = max(0.0, min(by2, ky2) - max(by1, ky1))
            inter_box = inter_w * inter_h
            d_box_area = max((bx2 - bx1) * (by2 - by1), 1.0)
            k_box_area = max((kx2 - kx1) * (ky2 - ky1), 1.0)
            box_containment = inter_box / min(d_box_area, k_box_area)

            # Same-cow detections from overlapping tiles usually have strong
            # box overlap or one mask contained in the other. Keep the
            # highest-confidence instance.
            dcx = ((bx1 + bx2) * 0.5) - ((kx1 + kx2) * 0.5)
            dcy = ((by1 + by2) * 0.5) - ((ky1 + ky2) * 0.5)
            center_dist = float(np.hypot(dcx, dcy))
            d_diag = float(np.hypot(max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)))
            k_diag = float(np.hypot(max(kx2 - kx1, 1.0), max(ky2 - ky1, 1.0)))
            near_same_center = center_dist <= 0.28 * min(d_diag, k_diag)

            if (
                box_iou >= 0.42
                or mask_iou >= 0.42
                or (box_containment >= 0.55 and mask_containment >= 0.55)
                or (near_same_center and box_containment >= 0.20)
            ):
                duplicate = True
                break

        if not duplicate:
            unique.append(d)

    return unique


def _make_tiles(h: int, w: int, tile_size: int = 1024, overlap: float = 0.30) -> list[tuple[int, int, int, int]]:
    """Create a small overlapping tile grid for small/distant cows."""
    tile_size = int(max(384, tile_size))
    if h <= tile_size and w <= tile_size:
        return [(0, 0, w, h)]

    step = max(int(tile_size * (1.0 - overlap)), 256)
    xs = list(range(0, max(w - tile_size, 0) + 1, step))
    ys = list(range(0, max(h - tile_size, 0) + 1, step))

    if not xs or xs[-1] != max(w - tile_size, 0):
        xs.append(max(w - tile_size, 0))
    if not ys or ys[-1] != max(h - tile_size, 0):
        ys.append(max(h - tile_size, 0))

    tiles = []
    for y1 in sorted(set(ys)):
        for x1 in sorted(set(xs)):
            x2 = min(w, x1 + tile_size)
            y2 = min(h, y1 + tile_size)
            tiles.append((x1, y1, x2, y2))
    return tiles


def detect_cows(image_rgb: np.ndarray, conf: float = 0.43) -> list[dict[str, Any]]:
    """Detect goats with multi-pass recall while preserving the existing masks.

    Detection strategy:
      1. Primary project detector at 640px for fast, high-confidence detections.
      2. The SAME trained detector again at high resolution so small/distant goats
         are not lost simply because the first pass found some goats.
      3. Overlapping high-resolution tiles for crowded/herd images. Tiles are
         merged back into original-image coordinates and deduplicated.
      4. The generic segmentation model is an additional recovery source, not
         a replacement for the project detector.

    Important: finding one goat never stops detection. Every recovery pass is
    allowed to add missing goats, and the existing final confidence/mask guards
    remain in place to prevent ultra-low-confidence background detections.
    """
    image_rgb = np.asarray(image_rgb).astype(np.uint8)
    h, w = image_rgb.shape[:2]
    device = 0 if DEVICE.type == "cuda" else "cpu"

    def run(model, source: np.ndarray, threshold: float, image_size: int):
        return model.predict(
            source=source,
            conf=float(threshold),
            imgsz=int(image_size),
            iou=0.82,
            max_det=300,
            retina_masks=True,
            verbose=False,
            device=device,
        )[0]

    def collect(result, source_shape, allowed, conf_min, offset=(0, 0)):
        out = []
        ox, oy = offset
        for d in _extract_result_detections(
            result,
            source_shape,
            allowed,
            conf_min=conf_min,
        ):
            local_mask = d["mask_local"]

            if offset != (0, 0):
                full_mask = np.zeros((h, w), dtype=bool)
                y2 = min(h, oy + local_mask.shape[0])
                x2 = min(w, ox + local_mask.shape[1])
                full_mask[oy:y2, ox:x2] = local_mask[: y2 - oy, : x2 - ox]

                box = d["box"].copy()
                box[[0, 2]] += ox
                box[[1, 3]] += oy
                mask = full_mask
            else:
                mask = local_mask
                box = d["box"]

            area_fraction = float(mask.sum()) / max(float(h * w), 1.0)
            if float(d["confidence"]) < MIN_VALID_DET_CONF:
                continue
            if area_fraction < MIN_VALID_MASK_FRACTION:
                continue

            q, ql = side_profile_score(mask, box, image_rgb.shape)
            out.append({
                "mask": mask,
                "box": np.asarray(box, dtype=np.float32),
                "confidence": float(d["confidence"]),
                "class_id": int(d["class_id"]),
                "quality_score": q,
                "quality": ql,
            })
        return out

    candidates: list[dict[str, Any]] = []

    # ------------------------------------------------------------
    # 1) Primary project detector - fast/high precision pass.
    # ------------------------------------------------------------
    project = load_detector()
    project_allowed = _cow_class_ids(project)

    try:
        result = run(
            project,
            image_rgb,
            max(float(conf), MIN_VALID_DET_CONF),
            640,
        )
        candidates.extend(
            collect(
                result,
                image_rgb.shape,
                project_allowed,
                max(float(conf), MIN_VALID_DET_CONF),
            )
        )
    except Exception as exc:
        print(f"[cow-detect] primary 640 pass failed: {exc}")

    # ------------------------------------------------------------
    # 2) ALWAYS run a high-resolution pass.
    #    Previously this happened only when zero goats were found. That meant
    #    an image with 3 detected goats stopped there and all small/occluded
    #    goats were never searched for.
    # ------------------------------------------------------------
    highres_conf = max(MIN_VALID_DET_CONF, min(0.24, float(conf) * 0.70))
    try:
        result = run(
            project,
            image_rgb,
            highres_conf,
            1536,
        )
        candidates.extend(
            collect(
                result,
                image_rgb.shape,
                project_allowed,
                highres_conf,
            )
        )
    except Exception as exc:
        print(f"[cow-detect] high-resolution project pass failed: {exc}")

    # ------------------------------------------------------------
    # 3) High-resolution overlapping tiles.
    #    This is especially useful for herd images where each goat occupies
    #    only a small part of the full frame.
    # ------------------------------------------------------------
    if max(h, w) > 1024:
        for tx1, ty1, tx2, ty2 in _make_tiles(
            h,
            w,
            tile_size=1024,
            overlap=0.30,
        ):
            tile = image_rgb[ty1:ty2, tx1:tx2]
            if tile.size == 0:
                continue

            try:
                result = run(
                    project,
                    tile,
                    MIN_VALID_DET_CONF,
                    1024,
                )
                candidates.extend(
                    collect(
                        result,
                        tile.shape,
                        project_allowed,
                        MIN_VALID_DET_CONF,
                        offset=(tx1, ty1),
                    )
                )
            except Exception as exc:
                print(
                    f"[cow-detect] project tile "
                    f"({tx1},{ty1},{tx2},{ty2}) failed: {exc}"
                )

    # ------------------------------------------------------------
    # 4) Generic segmentation detector as an additional recovery source.
    #    It is no longer limited to the zero-detection case. Its results are
    #    merged with the project detector and deduplicated below.
    # ------------------------------------------------------------
    generic = load_generic_detector()
    if generic is not None:
        generic_allowed = _cow_class_ids(generic)

        try:
            result = run(
                generic,
                image_rgb,
                0.25,
                1536,
            )
            candidates.extend(
                collect(
                    result,
                    image_rgb.shape,
                    generic_allowed,
                    0.25,
                )
            )
        except Exception as exc:
            print(f"[generic-detector] full-frame recovery failed: {exc}")

        # Generic tiled recovery is restricted to difficult/large scenes so
        # normal single-goat images do not pay the full extra inference cost.
        if max(h, w) > 1024:
            for tx1, ty1, tx2, ty2 in _make_tiles(
                h,
                w,
                tile_size=1024,
                overlap=0.30,
            ):
                tile = image_rgb[ty1:ty2, tx1:tx2]
                if tile.size == 0:
                    continue
                try:
                    result = run(
                        generic,
                        tile,
                        0.22,
                        1024,
                    )
                    candidates.extend(
                        collect(
                            result,
                            tile.shape,
                            generic_allowed,
                            0.22,
                            offset=(tx1, ty1),
                        )
                    )
                except Exception:
                    continue

    if not candidates:
        print("[cow-detect] no candidates from any detection pass")
        return []

    # ------------------------------------------------------------
    # 5) Merge all passes before SAM2 refinement.
    # ------------------------------------------------------------
    unique = _dedupe_detections(candidates)
    print(
        f"[cow-detect] raw_candidates={len(candidates)} "
        f"unique_before_sam2={len(unique)}"
    )

    # ------------------------------------------------------------
    # 6) Preserve the existing SAM2 whole-object refinement.
    # ------------------------------------------------------------
    for det in unique:
        refined = refine_mask_with_sam2(
            image_rgb,
            det["box"],
            detector_mask=det["mask"],
        )
        if refined is not None and int(refined.sum()) > 0:
            det["mask"] = refined
            q, ql = side_profile_score(
                refined,
                det["box"],
                image_rgb.shape,
            )
            det["quality_score"] = q
            det["quality"] = ql

    unique = _dedupe_detections(unique)

    # ------------------------------------------------------------
    # 7) Final safety guard. Keep the existing thresholds unchanged.
    # ------------------------------------------------------------
    filtered = []
    for det in unique:
        area_fraction = (
            float(np.asarray(det["mask"]).astype(bool).sum())
            / max(float(h * w), 1.0)
        )
        if (
            float(det["confidence"]) >= MIN_VALID_DET_CONF
            and area_fraction >= MIN_VALID_MASK_FRACTION
        ):
            filtered.append(det)

    filtered.sort(
        key=lambda d: (
            float(d["box"][0]),
            float(d["box"][1]),
        )
    )

    print(
        f"[cow-detect] reliable_cows={len(filtered)} "
        f"from_raw={len(candidates)}"
    )
    return filtered

def crop_rgb(image_rgb: np.ndarray, box: np.ndarray, pad_ratio: float = 0.08) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, int(x1) - px)
    y1 = max(0, int(y1) - py)
    x2 = min(w, int(x2) + px)
    y2 = min(h, int(y2) + py)
    return image_rgb[y1:y2, x1:x2].copy()


def crop_mask(mask: np.ndarray, box: np.ndarray, pad_ratio: float = 0.08) -> np.ndarray:
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    px, py = int(bw * pad_ratio), int(bh * pad_ratio)
    x1 = max(0, int(x1) - px)
    y1 = max(0, int(y1) - py)
    x2 = min(w, int(x2) + px)
    y2 = min(h, int(y2) + py)
    return mask[y1:y2, x1:x2].copy()


# --------------------------- angle ---------------------------

def load_angle_model():
    if "angle" in _CACHE:
        return _CACHE["angle"]
    path = MODEL_DIR / MODEL_NAMES["angle"]
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, nn.Module):
        model = ckpt.to(DEVICE).eval()
        _CACHE["angle"] = model
        return model
    sd = _clean_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
    model = tv_models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.30),
        nn.Linear(1280, 128),
        nn.ReLU(),
        nn.Dropout(0.20),
        nn.Linear(128, 5),
    )
    model.load_state_dict(sd, strict=True)
    model = model.to(DEVICE).eval()
    _CACHE["angle"] = model
    return model


def classify_angle(image_rgb: np.ndarray, angle_model) -> tuple[str, float]:
    if angle_model is None or image_rgb.size == 0:
        return "UNKNOWN", 0.0
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tfm(image_rgb).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = angle_model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)[0]
        cls = int(torch.argmax(probs).item())
        conf = float(probs[cls].item())
    # Class 0 is the approved side-on class. Require a minimum probability
    # so uncertain predictions do not unlock weight/BCS or green annotation.
    good = cls == 0 and conf >= ANGLE_GOOD_MIN_CONF
    # TEMP DIAGNOSTIC: remove once the BAD-VIEW-for-everything issue is
    # confirmed fixed. Prints the full softmax so we can see whether class 0
    # is simply losing confidence, or whether a different class is now
    # winning altogether (checkpoint/class-order mismatch).
    print(f"[angle-debug] argmax_cls={cls} conf={conf:.3f} probs={[round(p,3) for p in probs.tolist()]}")
    return ("GOOD VIEW" if good else "BAD VIEW"), conf


# --------------------------- weight --------------------------
class FusionB3Geometry(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = tv_models.efficientnet_b3(weights=None)
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.geometry_encoder = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.ReLU(), nn.Linear(32, 32)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(1568, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1)
        )

    def forward(self, image, geom):
        visual = self.backbone(image)
        g = self.geometry_encoder(geom)
        return self.fusion_head(torch.cat([visual, g], dim=1))


def load_weight_model():
    if "weight" in _CACHE:
        return _CACHE["weight"]
    path = MODEL_DIR / MODEL_NAMES["weight"]
    if not path.exists():
        _CACHE["weight"] = None
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obj = _unwrap_checkpoint(ckpt)
    if isinstance(obj, nn.Module):
        model = obj.to(DEVICE).eval()
        _CACHE["weight"] = model
        return model
    model = FusionB3Geometry()
    sd = _clean_state_dict(ckpt.get("model_state_dict", {})) if isinstance(ckpt, dict) else _clean_state_dict(obj)
    model.load_state_dict(sd, strict=True)
    model = model.to(DEVICE).eval()
    _CACHE["weight"] = model
    return model


def mask_geometry(mask: np.ndarray, box: np.ndarray, image_shape: tuple[int, ...], confidence: float) -> np.ndarray:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = map(float, box)
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    return np.array([
        bw / max(float(w), 1.0),
        bh / max(float(h), 1.0),
        (bw * bh) / max(float(h * w), 1.0),
        bw / bh,
        float(confidence),
        1.0,
    ], dtype=np.float32)


def predict_weight(rgb_crop: np.ndarray, geom: np.ndarray) -> tuple[float | None, str]:
    model = load_weight_model()
    if model is None:
        return None, "weight model unavailable"
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tfm(rgb_crop).unsqueeze(0).to(DEVICE)
    g = torch.from_numpy(np.asarray(geom, dtype=np.float32)).float().unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        y = model(x, g)
        if isinstance(y, (tuple, list)):
            y = y[0]
        value = float(y.reshape(-1)[0].item())
    kg = float(np.expm1(value))
    if not np.isfinite(kg):
        return None, "non-finite weight prediction"
    return kg, "EfficientNet-B3 + YOLO geometry"


# --------------------------- BCS -----------------------------
def build_dge_bcs_backbone(n_classes=5):
    model = tv_models.mobilenet_v3_small(weights=None)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, n_classes)
    return model


def load_bcs_model():
    if "bcs" in _CACHE:
        return _CACHE["bcs"]
    path = MODEL_DIR / MODEL_NAMES["bcs"]
    meta_path = MODEL_DIR / MODEL_NAMES["bcs_meta"]
    if not path.exists():
        _CACHE["bcs"] = None
        return None
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    classes = meta.get("classes", [2, 3, 4, 5, 6])
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    obj = _unwrap_checkpoint(ckpt)
    if isinstance(obj, nn.Module):
        model = obj.to(DEVICE).eval()
    else:
        model = build_dge_bcs_backbone(len(classes))
        model.load_state_dict(_clean_state_dict(obj), strict=True)
        model = model.to(DEVICE).eval()
    result = (model, classes)
    _CACHE["bcs"] = result
    return result


def _norm_channel(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.uint8)
    return (np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def estimate_depth(rgb_crop: np.ndarray) -> tuple[np.ndarray | None, str]:
    if AutoImageProcessor is None or AutoModelForDepthEstimation is None:
        return None, "transformers unavailable"
    processor, model = _CACHE.get("depth", (None, None))
    if processor is None or model is None:
        processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID).to(DEVICE).eval()
        _CACHE["depth"] = (processor, model)
    pil = Image.fromarray(rgb_crop.astype(np.uint8), mode="RGB")
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    processed = processor.post_process_depth_estimation(outputs, target_sizes=[(pil.height, pil.width)])
    d = processed[0]["predicted_depth"].detach().float().cpu().numpy().astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    positive = d > 1e-6
    if not positive.any():
        return None, "empty depth map"
    inv = np.zeros_like(d, dtype=np.float32)
    inv[positive] = 1.0 / d[positive]
    return inv, f"Depth Anything V2 Small ({DEPTH_MODEL_ID})"


def rgb_to_dge(rgb_crop: np.ndarray, mask_crop: np.ndarray) -> tuple[np.ndarray | None, str]:
    depth_raw, source = estimate_depth(rgb_crop)
    if depth_raw is None:
        return None, source
    mask = np.asarray(mask_crop) > 0
    depth_raw[~mask] = 0.0
    depth_u8 = _norm_channel(depth_raw)
    depth_u8[~mask] = 0

    masked_rgb = np.where(mask[..., None], rgb_crop, 0)
    if SKIMAGE_AVAILABLE and skimage_rgb2gray is not None:
        gray_f = skimage_rgb2gray(masked_rgb)
    else:
        gray_f = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_u8 = np.clip(gray_f * 255.0, 0, 255).astype(np.uint8)
    gray_u8[~mask] = 0

    if SKIMAGE_AVAILABLE and skimage_canny is not None:
        edge = skimage_canny(depth_u8, sigma=1.0)
    else:
        edge = cv2.Canny(cv2.GaussianBlur(depth_u8, (0, 0), 1.0), 50, 100) > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    edge_u8 = (edge.astype(np.uint8) * 255)
    edge_u8[~dilated] = 0

    return np.stack([depth_u8, gray_u8, edge_u8], axis=-1).astype(np.uint8), source


def predict_bcs(dge: np.ndarray) -> tuple[dict[str, Any] | None, str]:
    loaded = load_bcs_model()
    if loaded is None:
        return None, "BCS checkpoint unavailable"
    model, classes = loaded
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tfm(dge).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)[0]
        idx = int(torch.argmax(probs).item())
    return {
        "bcs": int(classes[idx]),
        "confidence": float(probs[idx].item()),
        "probabilities": {int(c): float(p) for c, p in zip(classes, probs.detach().cpu().tolist())},
    }, "Approximate DGE from RGB + monocular depth"


# ------------------------- visualization ----------------------
def annotate_image(
    image_rgb: np.ndarray,
    detections: list[dict[str, Any]],
    run_weight: bool = True,
    run_bcs: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:

    out = image_rgb.copy()
    angle_model = load_angle_model()
    rows: list[dict[str, Any]] = []

    h, w = image_rgb.shape[:2]

    for idx, det in enumerate(detections, start=1):
        box = det["box"]
        mask = np.asarray(det["mask"]).astype(bool)

        crop = crop_rgb(image_rgb, box)

        # --------------------------------------------------------
        # AI analysis
        # --------------------------------------------------------
        angle_label, angle_conf = classify_angle(crop, angle_model)
        quality_score = float(det["quality_score"])

        weight = None
        weight_engine = "not loaded"

        if angle_label == "GOOD VIEW" and run_weight:
            geom = mask_geometry(
                mask,
                box,
                image_rgb.shape,
                det["confidence"],
            )
            weight, weight_engine = predict_weight(crop, geom)

        bcs_result = None
        bcs_state = "UNAVAILABLE"

        if angle_label == "GOOD VIEW" and run_bcs:
            mc = crop_mask(mask, box)

            try:
                dge, dge_source = rgb_to_dge(crop, mc)
            except Exception as exc:
                dge = None
                dge_source = f"DGE generation failed: {exc}"

            if dge is not None:
                bcs_result, bcs_kind = predict_bcs(dge)
                bcs_state = (
                    f"{bcs_kind}; source={dge_source}"
                    if bcs_result
                    else bcs_kind
                )
            else:
                bcs_state = dge_source

        # --------------------------------------------------------
        # SHARED SIDE-VIEW VISUAL ANNOTATION
        # --------------------------------------------------------
        render_side_annotation(
            out,
            mask,
            angle_label == "GOOD VIEW",
        )

        # --------------------------------------------------------
        # RESULT DATA
        # --------------------------------------------------------
        x1, y1, x2, y2 = [int(v) for v in box]

        rows.append({
            "cow_id": idx,
            "confidence": round(float(det["confidence"]), 4),
            "view_quality": angle_label,
            "annotation_color": "green" if angle_label == "GOOD VIEW" else "red",
            "view_confidence": round(float(angle_conf), 4),
            "geometry_quality": det["quality"],
            "quality_score": round(float(quality_score), 4),
            "box": [x1, y1, x2, y2],
            "weight_kg": (
                None
                if weight is None
                else round(float(weight), 1)
            ),
            "weight_engine": weight_engine,
            "bcs": (
                None
                if bcs_result is None
                else int(bcs_result["bcs"])
            ),
            "bcs_confidence": (
                None
                if bcs_result is None
                else round(float(bcs_result["confidence"]), 4)
            ),
            "bcs_state": bcs_state,
        })

    return out, rows


def analyze_image_bytes(
    data: bytes,
    conf: float = 0.43,
    progress_callback: ProgressCallback | None = None,
) -> tuple[bytes, list[dict[str, Any]]]:
    """
    Full image pipeline:

    1. Roboflow Workflow 6 / SAM 3 is the authoritative cow detector and
       image annotation source.
    2. Each SAM 3 cow is passed to the deployed side-view model.
    3. Only SIDE cows are allowed to reach the weight and BCS models.
    4. Workflow 14 supplies STANDING / SITTING / EATING behavior labels.
    5. The SAM 3 masks are recolored by the side-view gate: green for SIDE
       and red for NOT SIDE. The same annotation layer is used by video.
    """

    _emit_progress(progress_callback, 3, "Preparing image…")
    image = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(image)

    _emit_progress(progress_callback, 10, "Detecting cows with SAM 3…")

    # ---------------------------------------------------------
    # 1. SAM 3 cow detection + exact Roboflow annotation
    # ---------------------------------------------------------
    annotated_bytes, workflow_result = run_roboflow_cow_workflow(
        data,
        suffix=".jpg",
    )
    _emit_progress(progress_callback, 25, "Cow detection complete.")

    predictions_block = workflow_result.get("predictions", {})
    if not isinstance(predictions_block, dict):
        predictions_block = {}

    cow_predictions = predictions_block.get(
        "predictions",
        [],
    )
    if not isinstance(cow_predictions, list):
        cow_predictions = []

    if not cow_predictions:
        print("[image-analysis] roboflow_cows=0")
        return annotated_bytes, []

    rf_image = predictions_block.get("image", {})
    if not isinstance(rf_image, dict):
        rf_image = {}

    rf_width = float(
        rf_image.get("width") or arr.shape[1]
    )
    rf_height = float(
        rf_image.get("height") or arr.shape[0]
    )

    # Custom output canvas: same SAM 3 masks, recolored by the deployed
    # side-view classifier. This is independent of Roboflow's display palette.
    annotated_rgb = arr.copy()

    # ---------------------------------------------------------
    # 2. Client is shared for side-view + behavior calls
    # ---------------------------------------------------------
    client = load_roboflow_client()

    _emit_progress(progress_callback, 30, "Analyzing cow behavior…")

    # ---------------------------------------------------------
    # 3. Behavior workflow on the original image
    # ---------------------------------------------------------
    try:
        behavior_predictions = run_behavior_workflow(
            client,
            data,
        )
        print(
            f"[behavior] predictions={len(behavior_predictions)}"
        )
        _emit_progress(
            progress_callback,
            45,
            f"Behavior analysis complete ({len(behavior_predictions)} predictions).",
        )
    except Exception as exc:
        behavior_predictions = []
        print(
            f"[behavior] workflow failed: {exc}"
        )
        _emit_progress(
            progress_callback,
            45,
            "Behavior analysis unavailable; continuing.",
        )

    # ---------------------------------------------------------
    # 4. Analyze each SAM 3 cow independently
    # ---------------------------------------------------------
    rows: list[dict[str, Any]] = []

    for cow_id, prediction in enumerate(
        cow_predictions,
        start=1,
    ):

        if not isinstance(prediction, dict):
            continue

        class_name = str(
            prediction.get("class", "")
        ).strip().lower()

        # Workflow 6 should return only cow. Keep a defensive filter.
        if class_name and class_name != "cow":
            continue

        cow_progress = 45 + int(45 * (cow_id - 1) / max(len(cow_predictions), 1))
        _emit_progress(
            progress_callback,
            cow_progress,
            f"Analyzing Cow {cow_id} of {len(cow_predictions)}…",
        )

        try:
            analytics = analyze_roboflow_cow(
                client,
                arr,
                prediction,
                rf_width,
                rf_height,
            )
        except Exception as exc:
            print(
                f"[cow-analytics] Cow {cow_id} failed: {exc}"
            )
            fallback_box = _prediction_box(
                prediction,
                arr.shape[1],
                arr.shape[0],
                rf_width,
                rf_height,
            )
            fallback_mask = _decode_roboflow_rle(
                prediction.get("rle_mask"),
                arr.shape[:2],
            )
            if fallback_mask is None:
                fallback_mask = np.zeros(arr.shape[:2], dtype=bool)
                fx1, fy1, fx2, fy2 = fallback_box
                fallback_mask[fy1:fy2, fx1:fx2] = True

            analytics = {
                "box": fallback_box,
                "mask": fallback_mask,
                "side_ok": False,
                "view_quality": "UNKNOWN",
                "view_confidence": 0.0,
                "geometry_quality": "UNAVAILABLE",
                "quality_score": 0.0,
                "weight_kg": None,
                "weight_engine": f"analytics failed: {exc}",
                "bcs": None,
                "bcs_confidence": None,
                "bcs_state": f"analytics failed: {exc}",
            }

        # -----------------------------------------------------
        # Match behavior to the same cow.
        # -----------------------------------------------------
        behavior, behavior_score = match_behavior_to_cow(
            analytics,
            behavior_predictions,
            arr.shape[1],
            arr.shape[0],
            rf_width,
            rf_height,
        )

        # Shared annotation layer: the exact SAM 3 mask gets the side-view
        # color. SIDE = green; NOT SIDE / unavailable = red.
        render_side_annotation(
            annotated_rgb,
            analytics.get("mask"),
            bool(analytics.get("side_ok", False)),
        )

        detector_confidence = float(
            prediction.get("confidence", 0.0)
        )

        rows.append({
            "cow_id": cow_id,
            "confidence": round(
                detector_confidence,
                4,
            ),
            "view_quality": analytics["view_quality"],
            "annotation_color": "green" if analytics.get("side_ok") else "red",
            "view_confidence": analytics["view_confidence"],
            "geometry_quality": analytics["geometry_quality"],
            "quality_score": analytics["quality_score"],
            "box": analytics["box"],
            "weight_kg": analytics["weight_kg"],
            "weight_engine": analytics["weight_engine"],
            "bcs": analytics["bcs"],
            "bcs_confidence": analytics["bcs_confidence"],
            "bcs_state": analytics["bcs_state"],

            # New behavior fields for the frontend/API.
            "behavior": behavior,
            "behavior_confidence": round(
                float(behavior_score),
                4,
            ),
        })

        print(
            f"[cow {cow_id}] "
            f"det={detector_confidence:.3f} "
            f"view={analytics['view_quality']} "
            f"view_conf={analytics['view_confidence']:.3f} "
            f"behavior={behavior}"
        )

    _emit_progress(progress_callback, 96, "Finalizing image result…")

    print(
        f"[image-analysis] roboflow_cows={len(rows)}"
    )

    output_buffer = io.BytesIO()
    Image.fromarray(annotated_rgb).save(
        output_buffer,
        format="JPEG",
        quality=92,
        optimize=True,
    )
    output_bytes = output_buffer.getvalue()

    _emit_progress(progress_callback, 100, "Image analysis complete.")
    return output_bytes, rows

def analyze_video_bytes(
    data: bytes,
    suffix: str = ".mp4",
    conf: float = 0.28,
    sample_every: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """
    Roboflow-SAM3 video detection + lightweight temporal tracking + side-view
    color annotation.

    Primary detector for video:
        general-segmentation-api-6 -> SAM 3 -> cow masks

    Tracking:
        lightweight IoU/center tracker between Roboflow detections

    Performance:
        VIDEO_ROBOFLOW_EVERY controls how often SAM 3 is called.
        Default is every 5th frame. Intermediate frames reuse the most
        recent track mask translated by the estimated track velocity.

    This keeps image and video cow detection on the same Roboflow model while
    avoiding one cloud workflow call for every video frame.
    """
    import shutil
    import subprocess

    client = load_roboflow_client()

    rf_every = max(
        1,
        int(os.getenv("VIDEO_ROBOFLOW_EVERY", "5")),
    )
    video_conf = float(os.getenv("VIDEO_CONF", str(conf)))
    max_hold_frames = max(
        0,
        int(os.getenv("VIDEO_MASK_HOLD_FRAMES", "6")),
    )
    match_iou = float(
        os.getenv("VIDEO_TRACK_IOU", "0.10")
    )
    max_center_distance = float(
        os.getenv("VIDEO_TRACK_CENTER_DISTANCE", "0.18")
    )
    max_remote_width = max(
        640,
        int(os.getenv("VIDEO_ROBOFLOW_MAX_WIDTH", "1280")),
    )
    side_view_every = VIDEO_SIDE_VIEW_EVERY

    _emit_progress(
        progress_callback,
        2,
        "Preparing video…",
    )

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix or ".mp4",
    ) as f:
        f.write(data)
        input_path = f.name

    raw_output_path = tempfile.mktemp(suffix=".mp4")
    h264_output_path = tempfile.mktemp(suffix=".mp4")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = cv2.VideoWriter(
        raw_output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        try:
            os.unlink(input_path)
        except OSError:
            pass
        raise RuntimeError(
            "Could not create intermediate video writer."
        )

    frame_no = 0
    detections_total = 0
    remote_detection_frames = 0
    seen_tracks: set[int] = set()
    next_track_id = 1
    last_percent = -1

    # track_id -> temporal state
    track_state: dict[int, dict[str, Any]] = {}

    def resize_for_roboflow(
        rgb: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        """Resize frame for remote inference and return x/y scale to source."""
        src_h, src_w = rgb.shape[:2]
        if src_w <= max_remote_width:
            return rgb, 1.0, 1.0

        scale = max_remote_width / float(src_w)
        new_w = max_remote_width
        new_h = max(1, int(round(src_h * scale)))

        resized = cv2.resize(
            rgb,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        return resized, src_w / float(new_w), src_h / float(new_h)

    def render_mask(
        target: np.ndarray,
        mask: np.ndarray,
        side_ok: bool,
    ) -> None:
        render_side_annotation(
            target,
            mask,
            side_ok,
        )

    def translate_mask(
        mask: np.ndarray,
        shift_x: float,
        shift_y: float,
    ) -> np.ndarray:
        if mask is None or not mask.any():
            return mask

        if abs(shift_x) < 0.5 and abs(shift_y) < 0.5:
            return mask

        matrix = np.float32([
            [1.0, 0.0, float(shift_x)],
            [0.0, 1.0, float(shift_y)],
        ])

        moved = cv2.warpAffine(
            mask.astype(np.uint8),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return moved.astype(bool)

    def center_distance_ratio(
        box_a: list[int],
        box_b: list[int],
    ) -> float:
        ax1, ay1, ax2, ay2 = map(float, box_a)
        bx1, by1, bx2, by2 = map(float, box_b)

        acx = (ax1 + ax2) * 0.5
        acy = (ay1 + ay2) * 0.5
        bcx = (bx1 + bx2) * 0.5
        bcy = (by1 + by2) * 0.5

        dx = acx - bcx
        dy = acy - bcy
        distance = float(np.sqrt(dx * dx + dy * dy))

        diag = max(
            1.0,
            float(np.sqrt(width * width + height * height)),
        )
        return distance / diag

    def match_detections(
        detections: list[dict[str, Any]],
    ) -> dict[int, int]:
        """Greedy one-to-one matching: detection index -> existing track ID."""
        candidates: list[tuple[float, int, int]] = []

        for det_index, det in enumerate(detections):
            det_box = det["box"]
            for tid, state in track_state.items():
                gap = frame_no - int(state["last_seen"])
                if gap > max_hold_frames + rf_every:
                    continue

                predicted_box = state["box"]
                velocity = state.get(
                    "velocity",
                    np.zeros(2, dtype=np.float32),
                )
                shift = velocity * max(1, gap)
                px1, py1, px2, py2 = map(float, predicted_box)
                predicted = [
                    int(round(px1 + shift[0])),
                    int(round(py1 + shift[1])),
                    int(round(px2 + shift[0])),
                    int(round(py2 + shift[1])),
                ]

                iou = _box_iou(
                    det_box,
                    predicted,
                )
                distance = center_distance_ratio(
                    det_box,
                    predicted,
                )

                if iou < match_iou and distance > max_center_distance:
                    continue

                score = (0.75 * iou) + (
                    0.25 * max(
                        0.0,
                        1.0 - distance / max(max_center_distance, 1e-6),
                    )
                )
                candidates.append(
                    (score, det_index, tid)
                )

        candidates.sort(reverse=True)

        matched: dict[int, int] = {}
        used_tracks: set[int] = set()

        for score, det_index, tid in candidates:
            if det_index in matched or tid in used_tracks:
                continue
            matched[det_index] = tid
            used_tracks.add(tid)

        return matched

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            out = frame.copy()
            active_ids: set[int] = set()

            should_query_roboflow = (
                frame_no % rf_every == 0
                or not track_state
            )

            if should_query_roboflow:
                remote_rgb, sx, sy = resize_for_roboflow(rgb)

                encode_ok, encoded = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(
                        remote_rgb,
                        cv2.COLOR_RGB2BGR,
                    ),
                    [cv2.IMWRITE_JPEG_QUALITY, 88],
                )

                if not encode_ok:
                    raise RuntimeError(
                        "Could not encode video frame for Roboflow."
                    )

                frame_bytes = encoded.tobytes()
                rf_annotated, rf_result = run_roboflow_cow_workflow(
                    frame_bytes,
                    suffix=".jpg",
                )

                predictions_block = rf_result.get(
                    "predictions",
                    {},
                )
                if not isinstance(predictions_block, dict):
                    predictions_block = {}

                rf_meta = predictions_block.get(
                    "image",
                    {},
                )
                if not isinstance(rf_meta, dict):
                    rf_meta = {}

                rf_width = float(
                    rf_meta.get("width")
                    or remote_rgb.shape[1]
                )
                rf_height = float(
                    rf_meta.get("height")
                    or remote_rgb.shape[0]
                )

                predictions = predictions_block.get(
                    "predictions",
                    [],
                )
                if not isinstance(predictions, list):
                    predictions = []

                # Build source-sized detections from Roboflow predictions.
                remote_detections: list[dict[str, Any]] = []

                for pred in predictions:
                    if not isinstance(pred, dict):
                        continue

                    class_name = str(
                        pred.get("class", "")
                    ).strip().lower()

                    if class_name and class_name != "cow":
                        continue

                    pconf = float(
                        pred.get("confidence", 0.0) or 0.0
                    )
                    if pconf < video_conf:
                        continue

                    # Coordinates are in Roboflow's returned frame space.
                    rf_box = _prediction_box(
                        pred,
                        int(rf_width),
                        int(rf_height),
                        rf_width,
                        rf_height,
                    )

                    # Map RF coordinates first to resized source, then to
                    # original video coordinates.
                    box_source = [
                        int(round(rf_box[0] * sx)),
                        int(round(rf_box[1] * sy)),
                        int(round(rf_box[2] * sx)),
                        int(round(rf_box[3] * sy)),
                    ]

                    box_source = [
                        max(0, min(width, box_source[0])),
                        max(0, min(height, box_source[1])),
                        max(0, min(width, box_source[2])),
                        max(0, min(height, box_source[3])),
                    ]

                    mask_remote = _decode_roboflow_rle(
                        pred.get("rle_mask"),
                        remote_rgb.shape[:2],
                    )

                    if mask_remote is None:
                        mask_source = np.zeros(
                            (height, width),
                            dtype=bool,
                        )
                        x1, y1, x2, y2 = box_source
                        if x2 > x1 and y2 > y1:
                            mask_source[y1:y2, x1:x2] = True
                    else:
                        mask_source = cv2.resize(
                            mask_remote.astype(np.uint8),
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)

                    center = np.asarray(
                        [
                            (box_source[0] + box_source[2]) * 0.5,
                            (box_source[1] + box_source[3]) * 0.5,
                        ],
                        dtype=np.float32,
                    )

                    remote_detections.append({
                        "box": box_source,
                        "mask": mask_source,
                        "confidence": pconf,
                        "center": center,
                    })

                remote_detection_frames += 1
                detections_total += len(remote_detections)

                # ---------------------------------------------------------
                # Associate current Roboflow detections with existing tracks
                # ---------------------------------------------------------
                matched = match_detections(remote_detections)
                assigned_ids: set[int] = set()

                for det_index, det in enumerate(remote_detections):
                    tid = matched.get(det_index)

                    if tid is None:
                        tid = next_track_id
                        next_track_id += 1

                    previous = track_state.get(tid)
                    current_center = det["center"]

                    if previous is not None:
                        velocity = (
                            current_center
                            - previous["center"]
                        ) / max(
                            1,
                            frame_no - int(previous["last_seen"]),
                        )
                        # Smooth velocity to avoid large mask jumps.
                        velocity = (
                            0.65 * previous.get(
                                "velocity",
                                np.zeros(2, dtype=np.float32),
                            )
                            + 0.35 * velocity
                        )
                    else:
                        velocity = np.zeros(
                            2,
                            dtype=np.float32,
                        )

                    # -----------------------------------------------------
                    # Side-view classification for annotation color.
                    # New tracks are checked immediately; existing tracks are
                    # rechecked only every VIDEO_SIDE_VIEW_EVERY frames.
                    # -----------------------------------------------------
                    previous_side_ok = bool(previous.get("side_ok", False)) if previous else False
                    previous_side_conf = float(previous.get("side_confidence", 0.0)) if previous else 0.0
                    previous_side_label = str(previous.get("side_label", "NOT SIDE")) if previous else "NOT SIDE"
                    previous_side_frame = int(previous.get("last_side_frame", -10**9)) if previous else -10**9
                    should_check_side = (
                        previous is None
                        or frame_no - previous_side_frame >= side_view_every
                    )

                    side_ok = previous_side_ok
                    side_confidence = previous_side_conf
                    side_label = previous_side_label

                    if should_check_side:
                        try:
                            cow_crop = crop_rgb(
                                rgb,
                                np.asarray(det["box"], dtype=np.float32),
                                pad_ratio=0.08,
                            )
                            side_ok, side_confidence, side_label = run_side_view_model(
                                client,
                                cow_crop,
                            )
                        except Exception as exc:
                            side_ok = False
                            side_confidence = 0.0
                            side_label = f"SIDE VIEW ERROR: {exc}"

                    track_state[tid] = {
                        "mask": det["mask"],
                        "box": np.asarray(
                            det["box"],
                            dtype=np.float32,
                        ),
                        "center": current_center,
                        "velocity": velocity,
                        "last_seen": frame_no,
                        "confidence": det["confidence"],
                        "side_ok": bool(side_ok),
                        "side_confidence": float(side_confidence),
                        "side_label": side_label,
                        "last_side_frame": frame_no if should_check_side else previous_side_frame,
                    }

                    active_ids.add(tid)
                    assigned_ids.add(tid)
                    seen_tracks.add(tid)

                    render_mask(
                        out,
                        det["mask"],
                        bool(side_ok),
                    )

            # -------------------------------------------------------------
            # Between Roboflow calls, propagate the latest tracked masks.
            # -------------------------------------------------------------
            if not should_query_roboflow:
                for tid, state in list(track_state.items()):
                    gap = frame_no - int(state["last_seen"])
                    if gap < 1 or gap > max_hold_frames:
                        continue

                    velocity = state.get(
                        "velocity",
                        np.zeros(2, dtype=np.float32),
                    )

                    shifted = translate_mask(
                        state["mask"],
                        float(velocity[0]),
                        float(velocity[1]),
                    )

                    # Do not mutate the stored observation. `gap` is measured
                    # from the last real Roboflow detection, so the next query
                    # can predict the correct position from that anchor.

                    render_mask(
                        out,
                        shifted,
                        bool(state.get("side_ok", False)),
                    )

            # -------------------------------------------------------------
            # Drop very old states so they cannot reappear much later.
            # -------------------------------------------------------------
            for tid in list(track_state):
                gap = frame_no - int(
                    track_state[tid]["last_seen"]
                )
                if gap > max_hold_frames + rf_every:
                    del track_state[tid]

            writer.write(out)
            frame_no += 1

            if total > 0:
                percent = int(
                    min(
                        97,
                        round(
                            (frame_no / total) * 97
                        ),
                    )
                )

                if percent != last_percent:
                    last_percent = percent
                    _emit_progress(
                        progress_callback,
                        percent,
                        (
                            f"Processing video: frame "
                            f"{frame_no}/{total} · "
                            f"{len(seen_tracks)} cows tracked"
                        ),
                    )

    finally:
        cap.release()
        writer.release()

    _emit_progress(
        progress_callback,
        98,
        "Encoding browser-compatible video…",
    )

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        for path in (
            input_path,
            raw_output_path,
        ):
            try:
                os.unlink(path)
            except OSError:
                pass
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and make sure "
            "ffmpeg.exe is available in PATH."
        )

    conversion = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            raw_output_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            h264_output_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if conversion.returncode != 0:
        for path in (
            input_path,
            raw_output_path,
            h264_output_path,
        ):
            try:
                os.unlink(path)
            except OSError:
                pass
        raise RuntimeError(
            "FFmpeg H.264 conversion failed:\n"
            + conversion.stderr[-3000:]
        )

    output = Path(
        h264_output_path
    ).read_bytes()

    for path in (
        input_path,
        raw_output_path,
        h264_output_path,
    ):
        try:
            os.unlink(path)
        except OSError:
            pass

    track_summary = []
    for tid, state in sorted(
        track_state.items()
    ):
        track_summary.append({
            "track_id": int(tid),
            "last_confidence": round(
                float(state["confidence"]),
                4,
            ),
            "last_seen_frame": int(
                state["last_seen"]
            ),
            "side_view": "SIDE" if state.get("side_ok") else "NOT SIDE",
            "side_confidence": round(
                float(state.get("side_confidence", 0.0)),
                4,
            ),
        })

    _emit_progress(
        progress_callback,
        100,
        "Video analysis complete.",
    )

    return output, {
        "frames_processed": frame_no,
        "detections": detections_total,
        "input_frames": total,
        "tracked_cows": len(seen_tracks),
        "sample_every": sample_every,
        "roboflow_every": rf_every,
        "roboflow_detection_frames": remote_detection_frames,
        "side_view_every": side_view_every,
        "annotation": {
            "side_color": "green",
            "not_side_color": "red",
            "alpha": SIDE_ANNOTATION_ALPHA,
        },
        "codec": "H.264 / yuv420p",
        "video_conf": video_conf,
        "video_remote_width": max_remote_width,
        "track_summary": track_summary,
    }

def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models": {
            k: model_exists(k)
            for k in MODEL_NAMES
        },
        "bcs_input": "DGE: depth + grayscale + edge",
        "depth_model": DEPTH_MODEL_ID,
        "roboflow_cow_workflow": ROBOFLOW_WORKFLOW,
        "roboflow_side_model": ROBOFLOW_SIDE_MODEL,
        "roboflow_behavior_workflow": ROBOFLOW_BEHAVIOR_WORKFLOW,
        "side_view_min_conf": SIDE_VIEW_MIN_CONF,
    }

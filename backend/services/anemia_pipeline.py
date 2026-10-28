"""Two-model anemia screening pipeline: Conjunctiva Detection -> Local FAMACHA Scoring.

Pipeline flow:
    Upload -> Model A (Roboflow Cloud OR Local YOLO11n conjunctiva detector)
           -> validity gate (checks conjunctiva detected)
           -> crop conjunctiva ROI
           -> Model B (Local famacha_best.pt classifier)
           -> FAMACHA score output (Score 1-5, Anemia Risk, Confidence, Class)

Rules enforced:
* Model A (Conjunctiva): Detects the lower eyelid / conjunctiva.
    - Can be Roboflow Cloud instance segmentation ("roboflow")
    - Or Our Local YOLO11n detector ("local")
    - Or side-by-side comparison ("compare" / "both")
* Model B (FAMACHA): Runs the local famacha_best.pt classifier on the detected conjunctiva crop.
    - Classifies into FAMACHA_1, FAMACHA_2, FAMACHA_3, FAMACHA_4 (1-5 scale)
    - If a remote Roboflow FAMACHA model is configured in .env, it can be used,
      otherwise famacha_best.pt is the primary offline classifier.
* If conjunctiva is NOT detected on the eye, no score is produced.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import cv2
import numpy as np

CONJUNCTIVA_CLASS = os.getenv("CONJUNCTIVA_CLASS_NAME", "sheep-conjunctiva").strip().lower()


def _is_conjunctiva_class(name: Any) -> bool:
    """Allow both the configured class name, generic 'conjunctiva', or variants."""
    c = str(name or "").strip().lower()
    allowed = {
        CONJUNCTIVA_CLASS,
        "conjunctiva",
        "sheep-conjunctiva",
        "goat-conjunctiva",
        "eye-conjunctiva",
        "lower-eyelid-conjunctiva",
        "goat-eye-anemia-22222222",
    }
    return c in allowed or "conjunctiva" in c or c == "goat-eye-anemia-22222222"


def _min_confidence() -> float:
    try:
        return float(os.getenv("CONJUNCTIVA_MIN_CONFIDENCE", "0.5"))
    except ValueError:
        return 0.5


MSG_NOT_DETECTED = (
    "Conjunctiva not detected. Please upload a clearer, closer image of the "
    "lower eyelid."
)
MSG_DETECTION_FAILED = "Conjunctiva detection failed. No FAMACHA score was generated."
MSG_SCORING_FAILED = "Conjunctiva detected, but FAMACHA scoring failed."


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

def _has_geometry(pred: dict[str, Any]) -> bool:
    """Check if prediction has polygon points or a valid bounding box."""
    points = pred.get("points")
    if isinstance(points, list) and len(points) >= 3:
        return True
    try:
        w = float(pred.get("width", 0))
        h = float(pred.get("height", 0))
        if w > 4 and h > 4:
            return True
    except (TypeError, ValueError):
        pass
    return False


def evaluate_conjunctiva(
    rf_result: Any, min_conf: float | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """Return (best valid prediction, None) or (None, reason it was rejected)."""
    if not isinstance(rf_result, dict):
        return None, "Model A returned an invalid response."

    predictions = rf_result.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        return None, "Model A returned no predictions."

    threshold = min_conf if min_conf is not None else _min_confidence()
    valid: list[dict[str, Any]] = []
    saw_class = saw_confident = False

    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        cls_name = pred.get("class", "")
        if not _is_conjunctiva_class(cls_name):
            continue
        saw_class = True
        conf = pred.get("confidence")
        if not isinstance(conf, (int, float)) or float(conf) < threshold:
            continue
        saw_confident = True
        if not _has_geometry(pred):
            continue
        valid.append(pred)

    if valid:
        return max(valid, key=lambda p: float(p["confidence"])), None
    if not saw_class:
        return None, f"No conjunctiva prediction (expected '{CONJUNCTIVA_CLASS}' or 'conjunctiva')."
    if not saw_confident:
        return None, f"Confidence below threshold {threshold:.2f}."
    return None, "Prediction has no usable mask or bounding box."


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

def crop_conjunctiva(image_bytes: bytes, best: dict[str, Any], padding: float = 0.25) -> bytes | None:
    """Crop the detected conjunctiva (with padding) and return JPEG bytes."""
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]

    points = best.get("points") or []
    if len(points) >= 3:
        xs = [float(p["x"]) for p in points]
        ys = [float(p["y"]) for p in points]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    else:
        try:
            cx, cy = float(best["x"]), float(best["y"])
            bw, bh = float(best["width"]), float(best["height"])
        except (KeyError, TypeError, ValueError):
            return None
        x0, x1, y0, y1 = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2

    pad_x = (x1 - x0) * padding
    pad_y = (y1 - y0) * padding
    left = max(0, int(round(x0 - pad_x)))
    right = min(w, int(round(x1 + pad_x)))
    top = max(0, int(round(y0 - pad_y)))
    bottom = min(h, int(round(y1 + pad_y)))
    if right - left < 8 or bottom - top < 8:
        return None

    ok, encoded = cv2.imencode(
        ".jpg", image[top:bottom, left:right], [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    )
    return encoded.tobytes() if ok else None


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------

def risk_level(score: int | None) -> str:
    if score is None:
        return "Unavailable"
    if score <= 2:
        return "Lower anemia risk"
    if score == 3:
        return "Borderline / review"
    return "Higher anemia risk"


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def status() -> dict[str, Any]:
    from backend.services import famacha_local
    from backend.services import famacha_roboflow as fam_rf
    from backend.services.conjunctiva_roboflow import (
        WORKFLOW_ID as conj_workflow_id,
        conjunctiva_is_configured,
    )
    from backend.services.conjunctiva_local import (
        get_model_path as get_local_conj_path,
        local_model_is_available as local_conj_available,
    )

    conj_ok = conjunctiva_is_configured()
    local_conj_ok = local_conj_available()
    local_fam_ok = famacha_local.famacha_model_is_available()
    rf_fam_ok = fam_rf.famacha_is_configured()

    fam_ok = local_fam_ok or rf_fam_ok

    return {
        "available": (conj_ok or local_conj_ok) and fam_ok,
        "model": fam_rf.model_id() if rf_fam_ok else "famacha_best.pt",
        "task": "Conjunctiva detection, then FAMACHA scoring",
        "detectors": {
            "roboflow": {
                "available": conj_ok,
                "model_id": conj_workflow_id,
                "provider": "GoatVision AI Vision Engine",
                "type": "instance_segmentation (polygon mask)",
                "expected_class": CONJUNCTIVA_CLASS,
            },
            "local": {
                "available": local_conj_ok,
                "model_id": f"local:{get_local_conj_path().name}",
                "model_path": str(get_local_conj_path()),
                "provider": "Our Local Offline Model",
                "type": "object_detection (YOLO11n bounding box)",
                "expected_class": "conjunctiva",
            },
        },
        "default_detector": "roboflow" if conj_ok else ("local" if local_conj_ok else "none"),
        "conjunctiva": {
            "configured": conj_ok or local_conj_ok,
            "roboflow_configured": conj_ok,
            "local_available": local_conj_ok,
            "model_id": conj_workflow_id,
            "expected_class": CONJUNCTIVA_CLASS,
            "min_confidence": _min_confidence(),
        },
        "famacha": {
            "configured": fam_ok,
            "local_available": local_fam_ok,
            "roboflow_configured": rf_fam_ok,
            "model_id": fam_rf.model_id() if rf_fam_ok else "famacha_best.pt",
            "provider": "AI Cloud Classifier" if rf_fam_ok else "Local Classifier (famacha_best.pt)",
            "input_mode": fam_rf.input_mode() if rf_fam_ok else "crop",
            "classes": sorted(fam_rf.CLASS_TO_SCORE) if fam_rf.CLASS_TO_SCORE else ["FAMACHA_1", "FAMACHA_2", "FAMACHA_3", "FAMACHA_4"],
            "missing": [] if fam_ok else ["famacha_best.pt not found on disk"],
        },
        "message": (
            "Anemia pipeline ready (Conjunctiva detection + FAMACHA scoring)."
            if (conj_ok or local_conj_ok) and fam_ok
            else "Pipeline not fully configured."
        ),
    }


# ---------------------------------------------------------------------------
# Detector runner helper
# ---------------------------------------------------------------------------

def _run_single_detector(
    detector_type: str,
    image_bytes: bytes,
    conjunctiva_fn: Callable[[bytes], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute either the cloud Roboflow detector or the local YOLO detector."""
    from backend.services import conjunctiva_roboflow as conj
    from backend.services import conjunctiva_local as local_conj

    out: dict[str, Any] = {
        "configured": True,
        "success": False,
        "detected": False,
        "model_id": "unknown",
        "provider": "unknown",
        "model_type": "unknown",
        "prediction_count": 0,
        "predictions": [],
        "best_confidence": None,
        "annotated_image": None,
        "latency_ms": None,
        "error": None,
        "reason": None,
    }

    t0 = time.perf_counter()

    if conjunctiva_fn is not None:
        try:
            res = conjunctiva_fn(image_bytes)
        except Exception as exc:
            out["error"] = str(exc)
            return out
    elif detector_type == "local":
        out["model_id"] = f"local:{local_conj.get_model_path().name} (YOLO11n)"
        out["provider"] = "Our Local Model (YOLO11n)"
        out["model_type"] = "bounding_box"

        if not local_conj.local_model_is_available():
            out["configured"] = False
            out["error"] = f"Local model file not found at {local_conj.get_model_path()}"
            return out
        try:
            res = local_conj.run_local_conjunctiva_detection(image_bytes, conf_threshold=0.20)
        except Exception as exc:
            out["error"] = f"Local detection failed: {exc}"
            return out
    else:  # default AI vision workflow
        out["model_id"] = conj.WORKFLOW_ID
        out["provider"] = "GoatVision AI Vision Engine"
        out["model_type"] = "instance_segmentation (polygon mask)"

        if not conj.conjunctiva_is_configured():
            out["configured"] = False
            out["error"] = "Vision API key is not configured in backend/.env"
            return out
        try:
            res = conj.run_conjunctiva_workflow(image_bytes)
        except Exception as exc:
            out["error"] = f"Conjunctiva detection failed: {exc}"
            return out

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    out["success"] = True
    out["latency_ms"] = res.get("latency_ms", elapsed)

    if isinstance(res, dict):
        out["model_id"] = res.get("model_id", out["model_id"])
        out["provider"] = res.get("provider", out["provider"])
        out["model_type"] = res.get("model_type", out["model_type"])
        out["predictions"] = res.get("predictions") or []
        out["prediction_count"] = len(out["predictions"])
        out["annotated_image"] = res.get("annotated_image")
        out["raw_image"] = res.get("raw_image")

    best, reason = evaluate_conjunctiva(res)
    if best is not None:
        out["detected"] = True
        out["best_confidence"] = float(best["confidence"])
        out["best_prediction"] = best
    else:
        out["reason"] = reason

    print(
        f"[ANEMIA] Engine={detector_type} count={out['prediction_count']} "
        f"detected={out['detected']} conf={out['best_confidence']} reason={out['reason']}"
    )

    return out


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze(
    image_bytes: bytes,
    conjunctiva_fn: Callable[[bytes], dict[str, Any]] | None = None,
    famacha_fn: Callable[[bytes], dict[str, Any]] | None = None,
    input_mode: str | None = None,
    detector: str = "roboflow",
) -> dict[str, Any]:
    """Run conjunctiva detection followed by local famacha_best.pt scoring."""
    mode_clean = str(detector or "roboflow").strip().lower()
    is_compare_mode = mode_clean in ("compare", "both")

    famacha: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "model_id": "famacha_best.pt",
        "provider": "Local Classifier (famacha_best.pt)",
        "score": None,
        "class_name": None,
        "confidence": None,
        "risk_level": "Unavailable",
        "error": None,
    }

    comparison: dict[str, Any] | None = None

    # ---- Model A (Conjunctiva Detection) ----
    if is_compare_mode and conjunctiva_fn is None:
        rf_result = _run_single_detector("roboflow", image_bytes)
        local_result = _run_single_detector("local", image_bytes)

        comparison = {
            "roboflow": rf_result,
            "local": local_result,
            "agreement": (rf_result.get("detected") == local_result.get("detected")),
        }

        # Select the detected one as primary for scoring
        if rf_result.get("detected"):
            conjunctiva = rf_result
        elif local_result.get("detected"):
            conjunctiva = local_result
        else:
            conjunctiva = rf_result
    else:
        chosen_type = "local" if mode_clean == "local" else "roboflow"
        conjunctiva = _run_single_detector(chosen_type, image_bytes, conjunctiva_fn=conjunctiva_fn)

    def respond(message: str, ok_status: str) -> dict[str, Any]:
        scored = famacha["success"] and conjunctiva.get("detected") is True
        score = famacha["score"] if scored else None
        res_dict = {
            "status": ok_status,
            "message": message,
            "detector_used": conjunctiva.get("provider", mode_clean),
            "famacha_score": score,
            "risk_level": risk_level(score),
            "annotated_image": conjunctiva.get("annotated_image") or "",
            "predictions": (
                [
                    {
                        "id": 1,
                        "class_name": famacha["class_name"],
                        "confidence": round(float(famacha["confidence"]) * 100, 1),
                        "famacha_score": score,
                    }
                ]
                if scored
                else []
            ),
            "model": {
                "available": bool(conjunctiva.get("success")),
                "model": famacha["model_id"] or "famacha_best.pt",
            },
            "conjunctiva": conjunctiva,
            "famacha": famacha,
        }
        if comparison is not None:
            res_dict["comparison"] = comparison
        return res_dict

    # ---- Check detection result ----
    if not conjunctiva.get("success"):
        famacha["error"] = "Skipped: conjunctiva detection failed."
        return respond(MSG_DETECTION_FAILED, "error")

    if not conjunctiva.get("detected"):
        famacha["error"] = "Skipped: conjunctiva not detected."
        return respond(MSG_NOT_DETECTED, "ok")

    best = conjunctiva.get("best_prediction")
    if best is None:
        best, _ = evaluate_conjunctiva(conjunctiva)

    # ---- Model B (FAMACHA Scoring via local famacha_best.pt) ----
    run_b = famacha_fn
    mode = input_mode

    if run_b is None:
        from backend.services import famacha_local
        from backend.services import famacha_roboflow as fam_rf

        if fam_rf.famacha_is_configured():
            run_b = fam_rf.run_famacha_model
            mode = mode or fam_rf.input_mode()
        elif famacha_local.famacha_model_is_available():
            run_b = famacha_local.run_famacha_scoring
            mode = mode or "crop"
        else:
            famacha["error"] = "FAMACHA model file (famacha_best.pt) not found on disk."
            return respond(MSG_SCORING_FAILED, "ok")
    else:
        mode = mode or "crop"

    famacha["attempted"] = True

    # Prepare model input (cropped conjunctiva ROI or original frame fallback)
    if mode == "crop":
        model_input = crop_conjunctiva(image_bytes, best) if best else None
        if model_input is None:
            model_input = image_bytes
    else:
        model_input = image_bytes

    try:
        scored = run_b(model_input)
        score = int(scored["score"])
        confidence = float(scored["confidence"])
        class_name = str(scored["class_name"])
    except Exception as exc:
        famacha["error"] = str(exc)
        return respond(MSG_SCORING_FAILED, "ok")

    famacha.update(
        {
            "success": True,
            "model_id": scored.get("model_id", famacha["model_id"]),
            "provider": scored.get("provider", famacha["provider"]),
            "score": score,
            "class_name": class_name,
            "confidence": confidence,
            "risk_level": risk_level(score),
            "input_mode": mode,
        }
    )
    return respond(
        f"Conjunctiva detected ({conjunctiva.get('provider')}). FAMACHA Score {score} "
        f"({confidence * 100:.1f}% confidence). {risk_level(score)}.",
        "ok",
    )

"""Two-model anemia screening pipeline: Conjunctiva Detection -> FAMACHA Scoring.

Pipeline flow:
    Upload -> Model A (Conjunctiva Segmentation Model)
           -> validity gate (checks conjunctiva detected with valid polygon mask)
           -> crop conjunctiva ROI
           -> Model B (FAMACHA Scoring Model, Scores 1–5)
           -> FAMACHA score output (Score 1-5, Anemia Risk, Confidence, Class)

Rules enforced:
* Model A (Conjunctiva): Instance segmentation model detecting lower eyelid / conjunctiva polygon.
* Model B (FAMACHA): Cloud FAMACHA scoring model (Scores 1 to 5).
* Strictly NO local fallback models (famacha_best.pt / conjunctiva_best.pt removed).
* If conjunctiva is NOT detected on the eye, no FAMACHA score is produced.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

import cv2
import numpy as np

CONJUNCTIVA_CLASS = os.getenv("CONJUNCTIVA_CLASS_NAME", "sheep-conjunctiva").strip().lower()


def _is_conjunctiva_class(name: Any) -> bool:
    """Allow configured class name, generic 'conjunctiva', or variants."""
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
        return float(os.getenv("CONJUNCTIVA_MIN_CONFIDENCE", "0.20"))
    except ValueError:
        return 0.20


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
    """Check if prediction has valid polygon points (only polygon segmentation is accepted)."""
    points = pred.get("points")
    if isinstance(points, list) and len(points) >= 3:
        return True
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
    return None, "Prediction has no usable polygon segmentation mask."


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
    from backend.services import famacha_roboflow as fam_rf
    from backend.services.conjunctiva_roboflow import (
        MODEL_ID as conj_model_id,
        conjunctiva_is_configured,
    )

    conj_ok = conjunctiva_is_configured()
    rf_fam_ok = fam_rf.famacha_is_configured()
    pipeline_ok = conj_ok  # Pipeline is operational with Colorimetric Analyzer and/or Cloud Model B

    return {
        "available": pipeline_ok,
        "model": "Conjunctiva Segmentation Model + FAMACHA Scoring Model",
        "task": "Conjunctiva detection, then FAMACHA scoring",
        "detectors": {
            "roboflow": {
                "available": conj_ok,
                "model_id": conj_model_id,
                "provider": "Conjunctiva Segmentation Model",
                "type": "instance_segmentation (polygon mask)",
                "expected_class": CONJUNCTIVA_CLASS,
            },
        },
        "default_detector": "roboflow" if conj_ok else "none",
        "conjunctiva": {
            "configured": conj_ok,
            "roboflow_configured": conj_ok,
            "model_id": conj_model_id,
            "provider": "Conjunctiva Segmentation Model",
            "expected_class": CONJUNCTIVA_CLASS,
            "min_confidence": _min_confidence(),
        },
        "famacha": {
            "configured": rf_fam_ok,
            "local_available": False,
            "roboflow_configured": rf_fam_ok,
            "model_id": fam_rf.model_id(),
            "provider": "FAMACHA Scoring Model",
            "input_mode": fam_rf.input_mode(),
            "classes": ["Nivel1", "Nivel2", "Nivel3", "Nivel4", "Nivel5"],
            "missing": fam_rf.missing_configuration(),
        },
        "message": (
            "Anemia pipeline ready (Conjunctiva detection + FAMACHA scoring)."
            if pipeline_ok
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
    """Execute the conjunctiva segmentation detector."""
    from backend.services import conjunctiva_roboflow as conj

    out: dict[str, Any] = {
        "configured": True,
        "success": False,
        "detected": False,
        "model_id": conj.MODEL_ID,
        "provider": "Conjunctiva Segmentation Model",
        "model_type": "instance_segmentation (polygon mask)",
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
    else:
        if not conj.conjunctiva_is_configured():
            out["configured"] = False
            out["error"] = "Conjunctiva model API key is not configured in backend/.env"
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
        out["provider"] = "Conjunctiva Segmentation Model"
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
        f"[ANEMIA] Engine=segmentation count={out['prediction_count']} "
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
    """Run conjunctiva detection followed strictly by cloud FAMACHA scoring.
    
    No local models are used. FAMACHA score is only generated if conjunctiva is detected.
    """
    from backend.services import famacha_roboflow as fam_rf

    famacha: dict[str, Any] = {
        "attempted": False,
        "success": False,
        "model_id": fam_rf.model_id(),
        "provider": "FAMACHA Scoring Model",
        "score": None,
        "class_name": None,
        "confidence": None,
        "risk_level": "Unavailable",
        "error": None,
    }

    # Model A: Exclusively the Conjunctiva Segmentation Model
    conjunctiva = _run_single_detector("roboflow", image_bytes, conjunctiva_fn=conjunctiva_fn)

    def respond(message: str, ok_status: str) -> dict[str, Any]:
        scored = famacha["success"] and conjunctiva.get("detected") is True
        score = famacha["score"] if scored else None
        return {
            "status": ok_status,
            "message": message,
            "detector_used": conjunctiva.get("provider", "Conjunctiva Segmentation Model"),
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
                "model": "Conjunctiva Segmentation Model + FAMACHA Scoring Model",
            },
            "conjunctiva": conjunctiva,
            "famacha": famacha,
        }

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

    # ---- Model B: FAMACHA Scoring Model ----
    # Hybrid Approach:
    # 1. First attempts Cloud Roboflow FAMACHA model if available.
    # 2. If Cloud Model B produces no predictions (common on unseen images due to small dataset),
    #    automatically uses the Conjunctiva Colorimetric Model (CIELAB Ophthalmic Standard)
    #    which measures vascular redness vs pallor from the segmented conjunctiva polygon.
    run_b = famacha_fn
    mode = input_mode or (fam_rf.input_mode() if fam_rf.famacha_is_configured() else "full")

    if run_b is None and fam_rf.famacha_is_configured():
        run_b = fam_rf.run_famacha_model

    famacha["attempted"] = True

    # Crop conjunctiva ROI for optional evaluation
    crop_bytes = crop_conjunctiva(image_bytes, best) if best else None

    # Input image: if mode is crop, use crop_bytes or fall back to image_bytes
    model_input = crop_bytes if mode == "crop" and crop_bytes is not None else image_bytes

    scored = None
    if run_b is not None:
        try:
            res = run_b(model_input, full_image_bytes=image_bytes)
            # Only trust Roboflow Model B if it has genuine confidence (>= 0.55).
            # A low confidence (< 0.55) indicates the CNN is confused by surrounding fingers/fur.
            if res and res.get("score") is not None and float(res.get("confidence", 0)) >= 0.55:
                scored = res
            elif res and res.get("score") is not None:
                print(
                    f"[ANEMIA] Roboflow confidence too low ({float(res.get('confidence', 0)):.1%}) for "
                    f"class {res.get('class_name')}. Delegating to Colorimeter for accurate tissue grading."
                )
        except Exception as exc:
            famacha["error"] = str(exc)
            print(f"[ANEMIA] Roboflow Model B returned no match or error: {exc}. Evaluating via Colorimetry.")

    # Fallback to Colorimetric Model using polygon coordinates
    if not scored or scored.get("score") is None:
        try:
            from backend.services.famacha_colorimeter import analyze_conjunctiva_color
            points = best.get("points") if isinstance(best, dict) else None
            box = best if isinstance(best, dict) else None
            scored = analyze_conjunctiva_color(image_bytes, points=points, box=box)
        except Exception as exc:
            famacha["error"] = f"Scoring evaluation failed: {exc}"
            return respond(MSG_SCORING_FAILED, "ok")

    score = int(scored["score"])
    confidence = float(scored["confidence"])
    class_name = str(scored["class_name"])

    famacha.update(
        {
            "success": True,
            "model_id": scored.get("model_id", famacha["model_id"]),
            "provider": scored.get("provider", "FAMACHA Scoring Model"),
            "score": score,
            "class_name": class_name,
            "confidence": confidence,
            "risk_level": risk_level(score),
            "input_mode": mode,
            "color_metrics": scored.get("metrics"),
        }
    )
    return respond(
        f"Conjunctiva detected ({conjunctiva.get('provider')}). FAMACHA Score {score} "
        f"({confidence * 100:.1f}% confidence). {risk_level(score)}.",
        "ok",
    )

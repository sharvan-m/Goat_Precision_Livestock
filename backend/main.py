"""
GoatVision FastAPI backend.

Run from the project root:
    python -m uvicorn backend.main:app --reload --port 8000

This API expects the existing project folders to remain at the project root:
    goat_weight_system/
    models/
    data/
    output/
"""

import base64
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / "backend" / ".env")
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from goat_weight_system.config import (  # noqa: E402
    CSV_LOG_PATH,
    EXCEL_LOG_PATH,
    DATASET_CSV_PATH,
    BEST_MODEL_PATH,
    OUTPUT_DIR,
)
from goat_weight_system.detector import GoatDetector, DetectionResult  # noqa: E402
from goat_weight_system.tracker import GoatTracker  # noqa: E402
from goat_weight_system.view_evaluator import BestViewEvaluator  # noqa: E402
from goat_weight_system.measurement_extractor import MeasurementExtractor  # noqa: E402
from goat_weight_system.regression_models import GoatWeightRegressor  # noqa: E402
from goat_weight_system.evaluation import ConfidenceCalculator  # noqa: E402
from goat_weight_system.visualizer import GoatVisualizer  # noqa: E402
from goat_weight_system.data_logger import DataLogger  # noqa: E402
from goat_weight_system.video_processor import VideoProcessor  # noqa: E402
from goat_weight_system.bytetrack_system import GoatByteTrackSystem  # noqa: E402
from backend.services.roboflow_workflow import (
    RoboflowWorkflowError,
    endpoint as roboflow_endpoint,
    is_configured as roboflow_is_configured,
    run_goat_workflow,
)
from backend.services.conjunctiva_roboflow import (
    MODEL_ID as CONJUNCTIVA_MODEL_ID,
    conjunctiva_is_configured,
    conjunctiva_endpoint,
)
from backend.services.famacha_roboflow import famacha_is_configured
from backend.services import anemia_pipeline
from backend.services.side_classifier import (  # noqa: E402
    configured as side_classifier_configured,
    classify as classify_side,
)


app = FastAPI(
    title="GoatVision API",
    version="1.0.0",
    description="Computer-vision backend for goat monitoring, tracking and weight estimation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "https://goat-precision-livestock.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MEDIA_ROOT = OUTPUT_DIR / "web_media"
MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(MEDIA_ROOT)), name="media")


@app.get("/")
def root_status():
    return {"name": "GoatVision API", "status": "online", "docs": "/docs", "health": "/api/health"}

_executor = ThreadPoolExecutor(max_workers=2)
_job_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def load_regressor():
    if not BEST_MODEL_PATH.exists():
        return None
    try:
        return GoatWeightRegressor.load(str(BEST_MODEL_PATH))
    except Exception:
        return None


def file_to_data_url(path: Path) -> str:
    mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def image_bytes_to_data_url(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def media_url(path: Path | str) -> str:
    """Return a URL-safe API media reference for a local output file."""
    return "/api/media?path=" + quote(str(path), safe="")


def image_data_url(data: bytes) -> str:
    return image_bytes_to_data_url(data)


def _coerce_points(value: Any) -> list[tuple[float, float]]:
    """Normalize polygon points from Roboflow instance-segmentation output."""
    if not isinstance(value, list):
        return []

    result: list[tuple[float, float]] = []
    for point in value:
        if isinstance(point, dict) and "x" in point and "y" in point:
            try:
                result.append((float(point["x"]), float(point["y"])))
            except (TypeError, ValueError):
                continue
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                result.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError):
                continue
    return result


def _prediction_geometry(pred: dict[str, Any]) -> tuple[list[tuple[float, float]], dict[str, float] | None]:
    points = _coerce_points(
        pred.get("points")
        or pred.get("polygon")
        or pred.get("segmentation")
        or pred.get("mask")
        or []
    )

    box = pred.get("box")
    normalized_box = None
    if isinstance(box, dict):
        try:
            normalized_box = {
                "x1": float(box["x1"]),
                "y1": float(box["y1"]),
                "x2": float(box["x2"]),
                "y2": float(box["y2"]),
            }
        except (KeyError, TypeError, ValueError):
            normalized_box = None

    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        normalized_box = {
            "x1": min(xs),
            "y1": min(ys),
            "x2": max(xs),
            "y2": max(ys),
        }

    return points, normalized_box


def _decode_visualization(value: Any) -> np.ndarray | None:
    """Decode a Roboflow visualization returned as data URL, URL or base64."""
    if not isinstance(value, str) or not value:
        return None
    try:
        if value.startswith("data:image/"):
            _, payload = value.split(",", 1)
            raw = base64.b64decode(payload)
        elif value.startswith(("http://", "https://")):
            import requests
            response = requests.get(value, timeout=60)
            response.raise_for_status()
            raw = response.content
        else:
            raw = base64.b64decode(value)
        return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def _polygon_points(pred):
    value = pred.get("points") or pred.get("polygon") or pred.get("segmentation") or pred.get("mask") or []
    if not isinstance(value, list): return []
    out=[]
    for q in value:
        try:
            if isinstance(q, dict): out.append((float(q["x"]), float(q["y"])))
            elif isinstance(q, (list,tuple)) and len(q)>=2: out.append((float(q[0]), float(q[1])))
        except (KeyError,TypeError,ValueError): pass
    return out

def _body_center_for_id(pred, width, height):
    pts=_polygon_points(pred)
    if len(pts)>=3:
        import numpy as _np
        contour=_np.asarray(pts,dtype=_np.float32).reshape((-1,1,2))
        m=cv2.moments(contour)
        if abs(m.get("m00",0.0))>1e-6:
            return (max(4,min(width-4,int(round(m["m10"]/m["m00"])) )), max(4,min(height-4,int(round(m["m01"]/m["m00"])) )))
    box=pred.get("box")
    if not isinstance(box,dict): return None
    try:
        x1,y1,x2,y2=(float(box[k]) for k in ("x1","y1","x2","y2"))
    except (KeyError,TypeError,ValueError): return None
    return (max(4,min(width-4,int(round((x1+x2)/2)))), max(4,min(height-4,int(round(y1+0.56*(y2-y1))))))


def _detector_mask_to_bool(mask: Any, height: int, width: int) -> np.ndarray | None:
    """Normalize common detector mask shapes into a HxW boolean mask."""
    if mask is None:
        return None

    try:
        arr = np.asarray(mask)
    except Exception:
        return None

    if arr.ndim == 3:
        arr = arr[0]

    if arr.ndim != 2:
        return None

    if arr.shape != (height, width):
        try:
            arr = cv2.resize(
                arr.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        except Exception:
            return None

    return arr > 0


def _draw_same_video_model_annotation(
    frame: np.ndarray,
    tracked_detections: list[Any],
    ready_by_track_id: dict[int, bool] | None = None,
) -> np.ndarray:
    """
    Clean production annotation for photo/video goat results.

    Requirements:
      - entire segmentation mask is filled, not just outlined
      - GREEN = measurement-ready / usable view
      - RED = not measurement-ready
      - no bounding boxes
      - no class/confidence/weight/BCS text on the image
      - only the numeric goat ID is displayed
    """
    h, w = frame.shape[:2]
    output = frame.copy()
    mask_layer = output.copy()
    valid: list[tuple[int, Any, np.ndarray]] = []

    # BGR colors used by OpenCV.
    GREEN = (70, 205, 70)
    RED = (65, 65, 225)
    alpha = 0.30

    for index, det in enumerate(tracked_detections, start=1):
        mask = _detector_mask_to_bool(getattr(det, "mask", None), h, w)
        if mask is None or not mask.any():
            continue

        track_id = getattr(det, "track_id", None)
        try:
            track_key = int(track_id) if track_id is not None else index
        except (TypeError, ValueError):
            track_key = index

        ready = bool(ready_by_track_id.get(track_key, False)) if ready_by_track_id else False
        color = GREEN if ready else RED

        # Fill the complete segmentation mask. No rectangle and no contour.
        mask_layer[mask] = color
        valid.append((index, det, mask))

    if valid:
        output = cv2.addWeighted(mask_layer, alpha, output, 1.0 - alpha, 0)

    # IDs only, centered on the detected goat's body/mask centroid.
    for index, det, mask in valid:
        moments = cv2.moments(mask.astype(np.uint8))
        if abs(moments.get("m00", 0.0)) > 1e-6:
            cx = int(round(moments["m10"] / moments["m00"]))
            cy = int(round(moments["m01"] / moments["m00"]))
        else:
            box = getattr(det, "box", None)
            if not box:
                continue
            cx = int(round((float(box[0]) + float(box[2])) / 2))
            cy = int(round(float(box[1]) + 0.56 * (float(box[3]) - float(box[1]))))

        label = f"{index:02d}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.55, min(0.82, w / 1450.0))
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        tx = max(3, min(w - tw - 3, int(round(cx - tw / 2))))
        ty = max(th + baseline + 3, min(h - 3, int(round(cy + th / 2))))

        # White ID with a subtle shadow; no background badge.
        cv2.putText(output, label, (tx + 1, ty + 1), font, scale,
                    (8, 12, 10), thickness + 2, cv2.LINE_AA)
        cv2.putText(output, label, (tx, ty), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)

    return output


def _add_ids_only(image: np.ndarray, predictions: list[dict[str, Any]]) -> np.ndarray:
    """
    Add numeric IDs only.

    `predictions` is the authoritative result list from GoatDetector, so the
    ID placement always follows the same detector used by the video workflow.
    """
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(0.82, w / 1450.0))
    thickness = 2

    for index, pred in enumerate(predictions, 1):
        box = pred.get("box")
        if not isinstance(box, dict):
            continue

        try:
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
        except (KeyError, TypeError, ValueError):
            continue

        # Torso/stomach center instead of head.
        cx = (x1 + x2) / 2.0
        cy = y1 + 0.56 * (y2 - y1)

        label = f"{index:02d}"
        (tw, th), baseline = cv2.getTextSize(
            label, font, scale, thickness
        )

        tx = max(3, min(w - tw - 3, int(round(cx - tw / 2))))
        ty = max(
            th + baseline + 3,
            min(h - 3, int(round(cy + th / 2))),
        )

        # Text shadow only. No rectangle/chip/background.
        cv2.putText(
            image,
            label,
            (tx + 1, ty + 1),
            font,
            scale,
            (7, 14, 9),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            label,
            (tx, ty),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return image



def _encode_image_data_url(image: np.ndarray) -> str | None:
    ok, enc=cv2.imencode('.jpg',image,[int(cv2.IMWRITE_JPEG_QUALITY),95])
    if not ok: return None
    return 'data:image/jpeg;base64,'+base64.b64encode(enc.tobytes()).decode('ascii')


def _render_model_annotation_only(source_bytes, rf_data, predictions):
    base=_decode_visualization(rf_data.get("mask_visualization") or rf_data.get("annotated_image"))
    if base is None:
        base=cv2.imdecode(np.frombuffer(source_bytes,np.uint8),cv2.IMREAD_COLOR)
    if base is None: return None
    return _encode_image_data_url(_add_ids_only(base.copy(),predictions))



def _box_iou(a: dict[str, float] | None, b: dict[str, float] | None) -> float:
    if not a or not b:
        return 0.0
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _browser_video_copy(source: Path) -> Path:
    """
    Chromium-compatible video copy. VideoProcessor often writes MP4 with a
    codec/container combination that browsers won't decode. Prefer H.264/AAC
    with fast-start metadata when FFmpeg is installed.
    """
    source = Path(source)
    target = source.with_name(source.stem + "_browser.mp4")

    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return source

    cmd = [
        ffmpeg, "-y",
        "-i", str(source),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(target),
    ]

    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=900,
        )
        if target.exists() and target.stat().st_size > 0:
            return target
    except (subprocess.SubprocessError, OSError):
        pass

    return source


def _fallback_measurement_ready(mask: Any, box: Any, image_shape: tuple[int, int, int]) -> bool:
    """Relax the old binary side gate only when the segmented body is usable.

    This is deliberately conservative: it is a fallback for animals whose
    body is sufficiently visible but whose side classifier/evaluator is too
    strict. It rejects strongly front/rear views and tiny/partial masks.
    """
    h, w = image_shape[:2]
    m = _detector_mask_to_bool(mask, h, w)
    if m is None or int(m.sum()) < max(500, int(h * w * 0.02)):
        return False

    ys, xs = np.where(m)
    if xs.size < 20:
        return False

    bw = float(xs.max() - xs.min() + 1)
    bh = float(ys.max() - ys.min() + 1)
    if bw < 0.75 * w and bh < 0.45 * h:
        return False

    # The animal should occupy a meaningful, mostly complete silhouette.
    fill = float(m.sum()) / max(bw * bh, 1.0)
    if fill < 0.16:
        return False

    # Principal-axis orientation: a usable lateral/angled body is more
    # horizontally extended than a straight-on goat. This is only a fallback.
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    pts -= pts.mean(axis=0, keepdims=True)
    if len(pts) < 20:
        return False
    cov = np.cov(pts, rowvar=False)
    try:
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, int(np.argmax(vals))]
        horizontalness = abs(float(v[0]))
    except Exception:
        horizontalness = bw / max(bh, 1.0)

    # Allow mildly diagonal goats; reject strongly vertical/front-facing views.
    aspect = bw / max(bh, 1.0)
    return horizontalness >= 0.52 and aspect >= 0.92


def analyze_image_bytes(data: bytes, filename: str, pixel_scale: float):
    """
    Photo analysis uses the SAME GoatDetector pipeline as video analysis.

    The detector itself is authoritative for:
      â€¢ goat count
      â€¢ detections
      â€¢ IDs for this image
      â€¢ view / side validation
      â€¢ measurements
      â€¢ weight prediction

    The photo annotation is produced directly from the detector's own
    segmentation masks, so Roboflow's dense `goat` label visualization cannot
    leak into the photo result.
    """
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode the uploaded image.")

    h, w = frame.shape[:2]

    # EXACT SAME DETECTOR CLASS USED BY THE VIDEO WEIGHT PIPELINE.
    detector = GoatDetector()
    tracker = GoatTracker()
    evaluator = BestViewEvaluator()
    extractor = MeasurementExtractor(pixel_scale_cm=pixel_scale)
    logger = DataLogger()
    regressor = load_regressor()

    detections = detector.detect(frame)
    tracked = tracker.update(detections)

    results = []

    for index, det in enumerate(tracked, start=1):
        if det.track_id is None:
            det.track_id = index

        track_len = (
            tracker.tracks[det.track_id].total_frames_seen
            if det.track_id in tracker.tracks
            else 1
        )

        view_eval = evaluator.evaluate(
            det,
            (h, w),
            tracked,
            frame=frame,
            track_history_length=track_len,
        )

        pred_weight = None
        confidence = None
        measurements = None

        measurement_ready = bool(view_eval.can_estimate_weight)

        # Preserve the proven view evaluator first. If it rejects the image
        # only because the side gate is too strict, allow a geometry-based
        # fallback when the complete goat body is actually measurable.
        if not measurement_ready:
            measurement_ready = _fallback_measurement_ready(
                det.mask, det.box, frame.shape
            )

        if measurement_ready:
            measurements = extractor.extract_features(det.mask, det.box)

            if regressor is not None:
                pred_weight = float(regressor.predict(measurements)[0])
            else:
                pred_weight = (
                    40.0
                    + measurements["body_length"] * 0.2
                    + measurements["girth"] * 0.3
                )

            confidence, _ = ConfidenceCalculator.calculate_confidence(
                view_eval,
                pred_weight,
                det.confidence,
            )

            logger.log_record(
                goat_id=f"GOAT_{det.track_id:02d}",
                image_filename=filename,
                predicted_weight=pred_weight,
                measurements=measurements,
                view_score=view_eval.view_score,
                view_category=view_eval.category,
                confidence_level=confidence,
            )

        x1, y1, x2, y2 = [float(v) for v in det.box]

        # Side/view classification is ON by default using the existing
        # evaluator. A dedicated Roboflow side classifier can override this
        # later when configured, but it never replaces GoatDetector.
        side_class = (
            "SIDE"
            if view_eval.category in {"GOOD", "ACCEPTABLE"}
            else "NOT SIDE"
        )
        side_confidence = round(float(view_eval.view_score), 1)

        if side_classifier_configured() and x2 > x1 and y2 > y1:
            try:
                sx1 = max(0, min(w - 1, int(round(x1))))
                sy1 = max(0, min(h - 1, int(round(y1))))
                sx2 = max(0, min(w - 1, int(round(x2))))
                sy2 = max(0, min(h - 1, int(round(y2))))

                if sx2 > sx1 and sy2 > sy1:
                    crop = frame[sy1:sy2, sx1:sx2]
                    ok_crop, crop_jpg = cv2.imencode(".jpg", crop)
                    if ok_crop:
                        side_result = classify_side(crop_jpg.tobytes())
                        if side_result.get("class"):
                            side_class = str(side_result["class"])

                        raw_conf = side_result.get("confidence")
                        if isinstance(raw_conf, (int, float)):
                            raw_conf = float(raw_conf)
                            side_confidence = (
                                round(raw_conf * 100, 1)
                                if abs(raw_conf) <= 1
                                else round(raw_conf, 1)
                            )
            except Exception:
                pass

        results.append({
            "goat_id": f"GOAT_{det.track_id:02d}",
            "detector_confidence": round(float(det.confidence) * 100, 1),
            "view_score": round(float(view_eval.view_score), 1),
            "view_category": view_eval.category,
            "side_class": side_class,
            "side_confidence": side_confidence,
            "weight_kg": round(pred_weight, 2) if pred_weight is not None else None,
            "confidence": confidence,
            "can_estimate_weight": bool(measurement_ready),
            "rejection_reasons": list(view_eval.rejection_reasons or []),
            "measurements": measurements,
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
        })

    # IMPORTANT:
    # Do not call the Roboflow visualization workflow for the primary photo
    # image. The photo output must match the detector that already works in
    # video analysis.
    ready_by_track_id = {}
    for item in results:
        try:
            track_num = int(str(item["goat_id"]).split("_")[-1])
            ready_by_track_id[track_num] = bool(item["can_estimate_weight"])
        except (KeyError, TypeError, ValueError):
            continue

    annotated_frame = _draw_same_video_model_annotation(
        frame, tracked, ready_by_track_id=ready_by_track_id
    )

    annotated = _encode_image_data_url(annotated_frame)

    ok, encoded_source = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )
    if not ok:
        raise ValueError("Unable to encode source image.")

    return {
        "filename": filename,
        "width": w,
        "height": h,
        "goats_detected": len(results),
        "results": results,
        "source_image": image_bytes_to_data_url(encoded_source.tobytes()),
        "annotated_image": annotated,
        "annotation_source": "Same GoatDetector segmentation as video",
        "roboflow_error": None,
        "model": "Same GoatDetector used by video analysis",
    }



def video_job(job_id: str, video_path: Path, frame_stride: int, pixel_scale: float):
    try:
        regressor = load_regressor()
        processor = VideoProcessor(regressor=regressor, pixel_scale_cm=pixel_scale)

        def progress(cur, total, fps, goats):
            pct = float(cur / max(total, 1))
            with _job_lock:
                _jobs[job_id].update({
                    "progress": round(pct, 4),
                    "current_frame": int(cur),
                    "total_frames": int(total),
                    "fps": round(float(fps), 2),
                    "goats_tracked": int(goats),
                })

        result = processor.process_video(
            str(video_path),
            frame_stride=frame_stride,
            progress_callback=progress,
        )

        browser_video_path = _browser_video_copy(Path(result["output_video_path"]))

        summary = []
        for _, item in result.get("summary", {}).items():
            summary.append({
                "goat_id": item["goat_id"],
                "final_weight_kg": item["final_weight_kg"],
                "best_view_score": item["best_view_score"],
                "view_category": item["view_category"],
                "side_class": "SIDE" if item["view_category"] in {"GOOD", "ACCEPTABLE"} else "NOT SIDE",
                "side_confidence": round(float(item["best_view_score"]), 1),
                "confidence": item["confidence"],
                "valid_frames_count": item["valid_frames_count"],
                "individual_weights": item["individual_weights"],
                "best_frame_url": media_url(item["best_frame_path"]),
            })

        with _job_lock:
            _jobs[job_id].update({
                "status": "completed",
                "progress": 1.0,
                "result": {
                    "total_goats": result["total_goats"],
                    "output_video_url": media_url(browser_video_path),
                    "download_video_url": media_url(Path(result["output_video_path"])),
                    "video_browser_compatible": browser_video_path != Path(result["output_video_path"]),
                    "excel_url": media_url(result["excel_results_path"]),
                    "csv_url": media_url(result["csv_results_path"]),
                    "summary": summary,
                },
            })
    except Exception as exc:
        with _job_lock:
            _jobs[job_id].update({
                "status": "failed",
                "error": str(exc),
            })



@app.get("/api/side/status")
def side_status():
    return {
        "configured": side_classifier_configured(),
        "note": "Dedicated Roboflow side classifier" if side_classifier_configured() else "BestViewEvaluator fallback"
    }


@app.get("/api/roboflow/status")
def roboflow_status():
    return {
        "configured": roboflow_is_configured(),
        "endpoint": roboflow_endpoint(),
        "provider": "Roboflow Serverless Model",
        "note": "API key is read only from the backend environment.",
    }


def _annotate_with_side_classification(
    image: np.ndarray,
    detections: list[DetectionResult],
    mask_visualization: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], int, int]:
    """
    Annotate goats strictly based on side classification for weight feasibility.
    - Full-body segmentation mask annotation (like Roboflow's full-body overlay).
    - Green (0, 220, 0): Right angle / valid side view suitable for weight estimation.
    - Red (0, 0, 220): Invalid angle (frontal, rear-facing, or heavily occluded/clipped).
    - Absolutely NO bounding boxes.
    - Absolutely NO 'goat' text labels on top of the image.
    """
    annotated = image.copy()
    h, w = image.shape[:2]
    evaluator = BestViewEvaluator()

    # 1. If Roboflow mask_visualization is available, extract precise full-body masks
    if mask_visualization is not None and len(detections) > 0:
        if mask_visualization.shape[:2] != (h, w):
            mask_vis = cv2.resize(mask_visualization, (w, h))
        else:
            mask_vis = mask_visualization

        # Diff between visualization and original image
        diff = cv2.absdiff(mask_vis, image)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, raw_changed = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY)

        # Mask out Roboflow's label badges (magenta/purple [255, 81, 168] BGR and white text inside)
        b_ch, g_ch, r_ch = mask_vis[:, :, 0], mask_vis[:, :, 1], mask_vis[:, :, 2]
        badge_color_mask = (b_ch > 200) & (g_ch < 140) & (r_ch > 120)
        white_text_mask = (b_ch > 220) & (g_ch > 220) & (r_ch > 220)
        badge_color_dilated = cv2.dilate(badge_color_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        badge_pixels = badge_color_mask | (white_text_mask & badge_color_dilated)
        badge_expanded = cv2.dilate(badge_pixels.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

        # Body mask across all goats with tags removed
        goat_bodies_mask = (raw_changed > 0) & (~badge_expanded)

        if np.any(goat_bodies_mask):
            y_coords, x_coords = np.where(goat_bodies_mask)
            centers = [((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2) for d in detections]
            det_masks = [np.zeros((h, w), dtype=np.uint8) for _ in detections]

            for y, x in zip(y_coords, x_coords):
                best_i = None
                min_dist = float("inf")
                for i, d in enumerate(detections):
                    x1, y1, x2, y2 = d.box
                    if (x1 - 15 <= x <= x2 + 15) and (y1 - 15 <= y <= y2 + 15):
                        cx, cy = centers[i]
                        dist = (x - cx) ** 2 + (y - cy) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            best_i = i
                if best_i is not None:
                    det_masks[best_i][y, x] = 255

            for i, det in enumerate(detections):
                if np.any(det_masks[i]):
                    det.mask = det_masks[i]
    else:
        # Fallback for detections lacking mask
        for det in detections:
            if det.mask is None:
                m = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = det.box.astype(int)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                ax, ay = max(5, (x2 - x1) // 2), max(5, (y2 - y1) // 2)
                cv2.ellipse(m, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
                det.mask = m

    predictions = []
    valid_count = 0
    invalid_count = 0
    validity_list = []

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det.box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        side_eval = evaluator.evaluate(
            det,
            (h, w),
            all_detections=detections,
            frame=image,
        )

        is_valid = bool(side_eval.can_estimate_weight)
        score = round(float(side_eval.view_score), 1)
        validity_list.append(is_valid)

        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1

        predictions.append({
            "class": "goat",
            "confidence": round(float(det.confidence), 4),
            "box": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
            "side_view_score": score,
            "is_valid_side": is_valid,
            "side_category": side_eval.category,
            "rejection_reasons": side_eval.rejection_reasons,
        })

    # Render clean full-body mask overlays (GREEN for valid side view, RED for invalid angle)
    overlay = annotated.copy()
    for det, is_valid in zip(detections, validity_list):
        color = (0, 220, 0) if is_valid else (0, 0, 220)
        if det.mask is not None and np.any(det.mask):
            overlay[det.mask > 0] = color

    # Blend with original image
    annotated = cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0)

    # Draw smooth contour boundaries around each goat's body mask (no rectangular boxes)
    for det, is_valid in zip(detections, validity_list):
        color = (0, 220, 0) if is_valid else (0, 0, 220)
        if det.mask is not None and np.any(det.mask):
            contours, _ = cv2.findContours(det.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, color, 2, cv2.LINE_AA)

    return annotated, predictions, valid_count, invalid_count


@app.post("/api/detect/goats")
async def detect_goats(
    file: UploadFile = File(...),
    detector: str = Form("roboflow"),
):
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported image type.")

    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Unable to decode image.")
    h, w = image.shape[:2]
    source_b64 = base64.b64encode(image_bytes).decode("ascii")
    source_data_url = f"data:image/jpeg;base64,{source_b64}"

    if detector == "roboflow":
        try:
            result = run_goat_workflow(image_bytes)
            raw_preds = result.get("predictions", [])
            rf_dets = []
            for p in raw_preds:
                b = p.get("box", {})
                bx1 = int(b.get("x1", 0))
                by1 = int(b.get("y1", 0))
                bx2 = int(b.get("x2", 0))
                by2 = int(b.get("y2", 0))

                rf_dets.append(DetectionResult(
                    box=np.array([bx1, by1, bx2, by2]),
                    confidence=float(p.get("confidence") or 0.85),
                    class_id=0,
                    mask=None,
                ))

            mask_vis_img = None
            mask_vis_data = result.get("mask_visualization")
            if mask_vis_data:
                try:
                    b64_str = mask_vis_data.split(",")[-1]
                    mask_vis_bytes = base64.b64decode(b64_str)
                    mask_vis_img = cv2.imdecode(np.frombuffer(mask_vis_bytes, np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    mask_vis_img = None

            annotated, predictions, valid_count, invalid_count = _annotate_with_side_classification(
                image, rf_dets, mask_visualization=mask_vis_img
            )
            _, enc = cv2.imencode(".jpg", annotated)
            annotated_b64 = base64.b64encode(enc.tobytes()).decode("ascii")

            return {
                "source_image": source_data_url,
                "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
                "goats_detected": len(predictions),
                "valid_side_views": valid_count,
                "invalid_views": invalid_count,
                "predictions": predictions,
                "workspace": "roboflow",
                "workflow": "Roboflow SAM 3",
                "image_width": w,
                "image_height": h,
                "annotation_source": "Side Angle Validated (Green=Valid Side View, Red=Unsuitable Angle)",
            }
        except RoboflowWorkflowError as exc:
            raise HTTPException(status_code=502, detail=f"Roboflow Workflow error: {exc}")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Roboflow detection error: {exc}")

    # Local Custom GoatDetector (YOLO11n-seg)
    conf_thresh = float(os.getenv("GOAT_MIN_CONFIDENCE", "0.20"))
    goat_detector = GoatDetector(conf_thresh=conf_thresh)
    detections = goat_detector.detect(image)

    # Fallback to general animal detector if custom weights found 0 on wild/diverse pasture image
    if len(detections) == 0:
        try:
            from ultralytics import YOLO
            gen_model = YOLO("weights/yolo26n.pt")
            gen_res = gen_model.predict(image, conf=conf_thresh, verbose=False)
            for b in gen_res[0].boxes:
                cid = int(b.cls[0])
                if cid in [16, 17, 18, 19, 20]:  # dog, horse, sheep, cow
                    detections.append(DetectionResult(
                        box=b.xyxy[0].cpu().numpy(),
                        confidence=float(b.conf[0]),
                        class_id=0,
                        mask=None,
                    ))
        except Exception:
            pass

    annotated, predictions, valid_count, invalid_count = _annotate_with_side_classification(image, detections)
    _, enc = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(enc.tobytes()).decode("ascii")

    return {
        "source_image": source_data_url,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
        "goats_detected": len(predictions),
        "valid_side_views": valid_count,
        "invalid_views": invalid_count,
        "predictions": predictions,
        "workspace": "local",
        "workflow": "yolo11n-seg (custom)",
        "image_width": w,
        "image_height": h,
        "annotation_source": "Side Angle Validated (Green=Valid, Red=Invalid)",
    }


@app.get("/api/anemia/status")
def get_anemia_status():
    """Status of the two-model anemia pipeline (conjunctiva -> FAMACHA)."""
    return anemia_pipeline.status()


@app.get("/api/anemia/conjunctiva/status")
def get_conjunctiva_status():
    """
    Status of the Roboflow conjunctiva model (Model A).

    The API key is never returned to the client.
    """
    return {
        "configured": conjunctiva_is_configured(),
        "endpoint": conjunctiva_endpoint(),
        "model_id": CONJUNCTIVA_MODEL_ID,
        "provider": "Roboflow Serverless Model",
        "note": "API key is read only from the backend environment.",
    }


@app.post("/api/anemia/analyze")
async def analyze_anemia_image(
    file: UploadFile = File(...),
    detector: str = Form("roboflow"),
):
    """
    Two-model anemia screening with selectable/comparable conjunctiva detector.

    1. Conjunctiva model (Model A) runs according to `detector`:
       - 'roboflow': hosted instance-segmentation polygon model
       - 'local': custom-trained local YOLO11n bounding box model
       - 'compare' / 'both': runs both side-by-side
    2. Only if a valid conjunctiva is found, the Roboflow FAMACHA model
       (Model B) runs.
    3. No local famacha_best.pt fallback. No detection -> no score.
    """
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format.",
        )

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )

    try:
        # The pipeline makes blocking network or local YOLO calls; keep the event loop free.
        return await run_in_threadpool(anemia_pipeline.analyze, data, detector=detector)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Anemia analysis failed: {exc}",
        ) from exc


@app.get("/api/health")
def health():
    regressor = load_regressor()
    return {
        "status": "online",
        "compute": "cuda" if torch.cuda.is_available() else "cpu",
        "weight_model_loaded": regressor is not None,
        "photo_video_detector_parity": True,
        "weight_model_name": regressor.model_name if regressor is not None else None,
        "records": len(read_records()),
        "bcs_available": False,
        "roboflow_configured": roboflow_is_configured(),
        "conjunctiva_roboflow_configured": conjunctiva_is_configured(),
        "conjunctiva_model_id": os.getenv("CONJUNCTIVA_ROBOFLOW_MODEL_ID", ""),
        "famacha_roboflow_configured": famacha_is_configured(),
        "famacha_model_id": os.getenv("FAMACHA_ROBOFLOW_MODEL_ID", ""),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
    }


@app.post("/api/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    pixel_scale: float = Form(1.0),
):
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported image type.")

    data = await file.read()
    try:
        return analyze_image_bytes(data, file.filename or "upload.jpg", pixel_scale)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/analyze/video")
async def start_video(
    file: UploadFile = File(...),
    frame_stride: int = Form(2),
    pixel_scale: float = Form(1.0),
):
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported video type.")

    video_dir = MEDIA_ROOT / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    destination = video_dir / f"{uuid.uuid4().hex}{suffix}"
    destination.write_bytes(await file.read())

    job_id = uuid.uuid4().hex
    with _job_lock:
        _jobs[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "current_frame": 0,
            "total_frames": 0,
            "fps": 0.0,
            "goats_tracked": 0,
        }

    _executor.submit(video_job, job_id, destination, max(1, frame_stride), pixel_scale)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _job_lock:
        data = _jobs.get(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return data


@app.post("/api/tracking/video")
async def tracking_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.35),
    activate: float = Form(0.25),
    lost_buffer: int = Form(45),
    match_threshold: float = Form(0.80),
    minimum_frames: int = Form(1),
):
    allowed = {".mp4", ".avi", ".mov", ".mkv"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported video type.")

    video_dir = MEDIA_ROOT / "tracking"
    video_dir.mkdir(parents=True, exist_ok=True)
    destination = video_dir / f"{uuid.uuid4().hex}{suffix}"
    destination.write_bytes(await file.read())

    system = GoatByteTrackSystem(
        conf_threshold=confidence,
        track_activation_threshold=activate,
        lost_track_buffer=lost_buffer,
        minimum_matching_threshold=match_threshold,
        minimum_consecutive_frames=minimum_frames,
    )
    try:
        result = system.process_video(video_path=str(destination))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    browser_video_path = _browser_video_copy(Path(result["output_video_path"]))
    return {
        "total_unique_goats": result["total_unique_goats"],
        "total_frames": result["total_frames"],
        "csv_rows_count": result["csv_rows_count"],
        "output_video_url": media_url(browser_video_path),
        "download_video_url": media_url(Path(result["output_video_path"])),
        "video_browser_compatible": browser_video_path != Path(result["output_video_path"]),
        "output_csv_url": media_url(result["output_csv_path"]),
        "goat_summary": result.get("goat_summary", []),
    }


def read_records():
    if not CSV_LOG_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(CSV_LOG_PATH)
    except Exception:
        return pd.DataFrame()


@app.get("/api/records")
def records():
    df = read_records()
    return {
        "count": len(df),
        "unique_goats": int(df["goat_id"].nunique()) if "goat_id" in df else 0,
        "rows": df.fillna("").tail(500).to_dict(orient="records"),
    }


@app.get("/api/analytics")
def analytics():
    df = read_records()
    if df.empty:
        return {
            "records": 0,
            "unique_goats": 0,
            "average_weight_kg": None,
            "median_weight_kg": None,
            "view_breakdown": {},
            "weights": [],
        }

    weights = (
        pd.to_numeric(df["predicted_weight_kg"], errors="coerce").dropna()
        if "predicted_weight_kg" in df else pd.Series(dtype=float)
    )
    views = df["view_category"].value_counts().to_dict() if "view_category" in df else {}

    return {
        "records": int(len(df)),
        "unique_goats": int(df["goat_id"].nunique()) if "goat_id" in df else 0,
        "average_weight_kg": round(float(weights.mean()), 2) if not weights.empty else None,
        "median_weight_kg": round(float(weights.median()), 2) if not weights.empty else None,
        "view_breakdown": views,
        "weights": [round(float(x), 2) for x in weights.tolist()],
    }


@app.get("/api/media")
def get_media(path: str):
    target = Path(path).resolve()
    media_root = MEDIA_ROOT.resolve()
    output_root = OUTPUT_DIR.resolve()

    if not (str(target).startswith(str(media_root)) or str(target).startswith(str(output_root))):
        raise HTTPException(status_code=403, detail="Invalid media path.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Media not found.")

    suffix = target.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return FileResponse(
        str(target),
        media_type=media_types.get(suffix),
        headers={
            "Content-Disposition": f'inline; filename="{target.name}"',
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
        },
    )


@app.get("/api/model")
def model_status():
    regressor = load_regressor()
    return {
        "available": regressor is not None,
        "path": str(BEST_MODEL_PATH),
        "name": regressor.model_name if regressor is not None else None,
        "dataset_present": DATASET_CSV_PATH.exists(),
        "csv_log_present": CSV_LOG_PATH.exists(),
        "excel_log_present": EXCEL_LOG_PATH.exists(),
    }



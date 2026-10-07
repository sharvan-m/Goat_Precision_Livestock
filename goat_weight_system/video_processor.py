"""
Fast, stable, and accurate video processing engine for goat weight estimation.
Features CUDA GPU acceleration, frame skipping, statistical outlier filtering (MAD/IQR),
median prediction aggregation, automatic output saving in the input video's directory,
and end-of-video summary overlay rendering.
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import time
import numpy as np
import pandas as pd
import cv2
import torch

from goat_weight_system.config import (
    COLOR_GOOD,
    COLOR_ACCEPTABLE,
    COLOR_POOR,
    COLOR_TEXT,
)
from goat_weight_system.detector import GoatDetector, DetectionResult
from goat_weight_system.tracker import GoatTracker, TrackedGoat
from goat_weight_system.view_evaluator import BestViewEvaluator, ViewEvaluation
from goat_weight_system.measurement_extractor import MeasurementExtractor
from goat_weight_system.regression_models import GoatWeightRegressor
from goat_weight_system.evaluation import ConfidenceCalculator
from goat_weight_system.visualizer import GoatVisualizer
from goat_weight_system.data_logger import DataLogger


def filter_weight_outliers(weights: List[float]) -> List[float]:
    """
    Remove extreme weight prediction outliers using Median Absolute Deviation (MAD).

    Args:
        weights: List of predicted weights across candidate frames.

    Returns:
        Filtered list of weights with extreme outliers removed.
    """
    if len(weights) <= 2:
        return weights

    arr = np.array(weights, dtype=np.float64)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))

    if mad == 0:
        # Fallback to standard deviation or IQR if MAD is zero
        std = np.std(arr)
        if std == 0:
            return weights
        filtered = arr[np.abs(arr - med) <= 2.0 * std]
    else:
        # Keep predictions within 2.5 * MAD from median
        filtered = arr[np.abs(arr - med) <= 2.5 * mad]

    if len(filtered) == 0:
        return weights

    return filtered.tolist()


class VideoProcessor:
    """Fast, GPU-accelerated video processing engine producing ONE final weight per unique Goat ID."""

    def __init__(
        self,
        detector: Optional[GoatDetector] = None,
        regressor: Optional[GoatWeightRegressor] = None,
        pixel_scale_cm: float = 1.0,
        device: Optional[str] = None,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = detector if detector else GoatDetector(device=self.device)
        self.tracker = GoatTracker(max_disappeared=60)
        self.evaluator = BestViewEvaluator()
        self.extractor = MeasurementExtractor(pixel_scale_cm=pixel_scale_cm)
        self.regressor = regressor
        self.visualizer = GoatVisualizer()

    def process_video(
        self,
        video_path: str,
        frame_stride: int = 1,
        max_candidate_frames: int = 5,
        progress_callback: Optional[Callable[[int, int, float, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Processes video stream efficiently and generates:
          1. Output video saved in SAME FOLDER as input video.
          2. Excel & CSV results saved in SAME FOLDER as input video.
          3. Best frame images saved in `{video_stem}_best_frames/` subfolder.

        Args:
            video_path: Path to input video file.
            frame_stride: Process detection every Nth frame for fast execution.
            max_candidate_frames: Max top view frames retained per Goat ID.
            progress_callback: Optional callback(current_frame, total_frames, current_fps, num_goats).

        Returns:
            Dictionary containing processing summary and per-goat final metrics.
        """
        # Validate frame_stride safely
        if callable(frame_stride):
            # Positional redirect in case progress_callback was passed as 2nd arg
            progress_callback = frame_stride
            frame_stride = 1
        elif frame_stride is None:
            frame_stride = 1
        else:
            try:
                frame_stride = max(1, int(frame_stride))
            except (ValueError, TypeError):
                frame_stride = 1

        input_file = Path(video_path).resolve()
        if not input_file.exists():
            raise FileNotFoundError(f"Input video not found at: {video_path}")

        # Target Output Paths in SAME FOLDER as input video
        input_dir = input_file.parent
        video_stem = input_file.stem

        output_video_path = input_dir / f"{video_stem}_weight_output.mp4"
        excel_results_path = input_dir / f"{video_stem}_weight_results.xlsx"
        csv_results_path = input_dir / f"{video_stem}_weight_results.csv"
        best_frames_dir = input_dir / f"{video_stem}_best_frames"
        best_frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if (fps and fps > 0) else 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        # Records storage per Goat ID
        # goat_id -> list of candidate dicts: {score, frame, detection, evaluation, weight_pred, measurements, frame_idx, timestamp_sec}
        goat_records: Dict[int, List[Dict[str, Any]]] = {}

        last_annotations: Tuple[List, List, List, List, List] = ([], [], [], [], [])
        frame_idx = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            timestamp_sec = frame_idx / fps

            # Stride logic:
            # If frame_stride == 1: process every frame
            # If frame_stride == 2: process every 2nd frame (frames 1, 3, 5, ...)
            # If frame_stride == 3: process every 3rd frame (frames 1, 4, 7, ...)
            is_stride_frame = (frame_stride == 1) or ((frame_idx - 1) % frame_stride == 0) or (frame_idx == 1)

            if is_stride_frame:
                current_detections = self.detector.detect(frame)
                tracked_detections = self.tracker.update(current_detections)

                evaluations: List[ViewEvaluation] = []
                display_predictions: List[Optional[float]] = []
                display_confidences: List[Optional[str]] = []
                measurements_list: List[Optional[Dict]] = []

                for i_d, det in enumerate(tracked_detections):
                    if det.track_id is None:
                        det.track_id = i_d + 1
                    goat_id = det.track_id

                    track_len = self.tracker.tracks[goat_id].total_frames_seen if goat_id in self.tracker.tracks else 1
                    view_eval = self.evaluator.evaluate(det, (h, w), tracked_detections, frame=frame, track_history_length=track_len)
                    evaluations.append(view_eval)

                    if view_eval.can_estimate_weight:
                        meas = self.extractor.extract_features(det.mask, det.box)
                        measurements_list.append(meas)

                        if self.regressor is not None:
                            pred_w = float(self.regressor.predict(meas)[0])
                        else:
                            pred_w = 40.0 + (meas["body_length"] * 0.2) + (meas["girth"] * 0.3)

                        conf_str, _ = ConfidenceCalculator.calculate_confidence(view_eval, pred_w, det.confidence)

                        if goat_id not in goat_records:
                            goat_records[goat_id] = []

                        goat_records[goat_id].append({
                            "score": view_eval.view_score,
                            "frame": frame.copy(),
                            "detection": det,
                            "evaluation": view_eval,
                            "predicted_weight": pred_w,
                            "measurements": meas,
                            "frame_idx": frame_idx,
                            "timestamp_sec": round(timestamp_sec, 2),
                        })

                        # Maintain top N frames sorted by View Score
                        goat_records[goat_id] = sorted(
                            goat_records[goat_id], key=lambda x: x["score"], reverse=True
                        )[:max_candidate_frames]

                        # Display running median weight during video processing
                        running_weights = [c["predicted_weight"] for c in goat_records[goat_id]]
                        filt_weights = filter_weight_outliers(running_weights)
                        running_median = float(np.median(filt_weights))

                        display_predictions.append(running_median)
                        display_confidences.append(conf_str)
                    else:
                        # If model is calculating or view is invalid
                        if goat_id in goat_records and len(goat_records[goat_id]) > 0:
                            running_weights = [c["predicted_weight"] for c in goat_records[goat_id]]
                            filt_weights = filter_weight_outliers(running_weights)
                            running_median = float(np.median(filt_weights))
                            display_predictions.append(running_median)
                            display_confidences.append("LOW")
                        else:
                            display_predictions.append(None)
                            display_confidences.append("LOW")
                        measurements_list.append(None)

                last_annotations = (tracked_detections, evaluations, display_predictions, display_confidences, measurements_list)
            else:
                # Reuse last annotations to preserve output video visual consistency without expensive re-computation
                tracked_detections, evaluations, display_predictions, display_confidences, measurements_list = last_annotations

            # Draw HUD annotation
            annotated_frame = self.visualizer.draw_annotated_frame(
                frame, tracked_detections, evaluations, display_predictions, display_confidences, measurements_list
            )
            writer.write(annotated_frame)

            # Progress update (periodic to avoid slowing down processing)
            if progress_callback and (frame_idx % 15 == 0 or frame_idx == total_frames):
                elapsed = time.time() - start_time
                current_fps = frame_idx / max(elapsed, 0.001)
                try:
                    progress_callback(frame_idx, total_frames, current_fps, len(self.tracker.tracks))
                except TypeError:
                    try:
                        progress_callback(frame_idx, total_frames)
                    except Exception:
                        pass

        # -----------------------------------------------------------------------------
        # FINAL AGGREGATION PER UNIQUE GOAT ID
        # -----------------------------------------------------------------------------
        final_summary: Dict[int, Dict[str, Any]] = {}
        excel_rows = []

        for goat_id, candidates in goat_records.items():
            if not candidates:
                continue

            # Pick highest scoring candidate as visual best frame
            best_cand = candidates[0]
            best_frame = best_cand["frame"]
            det = best_cand["detection"]
            view_eval = best_cand["evaluation"]
            meas = best_cand["measurements"]

            # Statistical Outlier Removal & Final Median Weight
            raw_weights = [c["predicted_weight"] for c in candidates]
            filtered_weights = filter_weight_outliers(raw_weights)
            final_weight_kg = round(float(np.median(filtered_weights)), 1)

            # Overall Confidence Calculation
            std_dev = float(np.std(filtered_weights)) if len(filtered_weights) > 1 else 0.0
            if view_eval.category == "GOOD" and len(candidates) >= 3 and std_dev < 2.5:
                confidence_str = "HIGH"
            elif view_eval.category in ["GOOD", "ACCEPTABLE"] and len(candidates) >= 2:
                confidence_str = "MEDIUM"
            else:
                confidence_str = "LOW"

            # Save best frame image in `{video_stem}_best_frames/` subfolder
            image_filename = f"goat_{goat_id:02d}_best_frame.jpg"
            save_image_path = best_frames_dir / image_filename

            annotated_best_img = self.visualizer.draw_annotated_frame(
                best_frame,
                [det],
                [view_eval],
                [final_weight_kg],
                [confidence_str],
                [meas],
            )
            cv2.imwrite(str(save_image_path), annotated_best_img)

            summary_item = {
                "goat_id": f"GOAT_{goat_id:02d}",
                "final_weight_kg": final_weight_kg,
                "best_view_score": round(view_eval.view_score, 1),
                "view_category": view_eval.category,
                "valid_frames_count": len(candidates),
                "best_frame_idx": best_cand["frame_idx"],
                "best_timestamp_sec": best_cand["timestamp_sec"],
                "confidence": confidence_str,
                "individual_weights": [round(w, 1) for w in raw_weights],
                "filtered_weights": [round(w, 1) for w in filtered_weights],
                "measurements": meas,
                "best_frame_path": str(save_image_path),
            }
            final_summary[goat_id] = summary_item

            excel_rows.append({
                "video_filename": input_file.name,
                "goat_id": f"GOAT_{goat_id:02d}",
                "final_weight_kg": final_weight_kg,
                "best_view_score": round(view_eval.view_score, 1),
                "view_category": view_eval.category,
                "weight_confidence": confidence_str,
                "frames_used_count": len(filtered_weights),
                "best_frame_idx": best_cand["frame_idx"],
                "video_timestamp_sec": best_cand["timestamp_sec"],
                "body_length_cm": meas.get("body_length", 0.0),
                "height_cm": meas.get("height", 0.0),
                "girth_cm": meas.get("girth", 0.0),
                "body_width_cm": meas.get("body_width", 0.0),
                "mask_area_cm2": meas.get("mask_area", 0.0),
                "best_frame_image": image_filename,
                "model_version": "v2.0-VideoAggregated",
            })

        # Write Excel & CSV Results Files in SAME FOLDER as input video
        if excel_rows:
            df_results = pd.DataFrame(excel_rows)
            df_results.to_excel(excel_results_path, index=False)
            df_results.to_csv(csv_results_path, index=False)

        # -----------------------------------------------------------------------------
        # APPEND FINAL VIDEO SUMMARY OVERLAY SCREEN (3 Seconds)
        # -----------------------------------------------------------------------------
        if final_summary and writer:
            summary_canvas = np.zeros((h, w, 3), dtype=np.uint8) + 30
            cv2.rectangle(summary_canvas, (30, 30), (w - 30, h - 30), (0, 220, 0), 3)

            cv2.putText(summary_canvas, "VIDEO PROCESSING COMPLETE - FINAL WEIGHT SUMMARY", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 120), 2)
            cv2.putText(summary_canvas, f"Total Unique Goats Measured: {len(final_summary)}", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)

            y_pos = 180
            for g_id, item in final_summary.items():
                line = f"{item['goat_id']}  |  FINAL WEIGHT: {item['final_weight_kg']:.1f} kg  |  View Score: {item['best_view_score']:.0f}/100 ({item['view_category']})  |  Confidence: {item['confidence']}"
                cv2.putText(summary_canvas, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
                y_pos += 40
                if y_pos > h - 60:
                    break

            # Write summary canvas for 3 seconds (fps * 3 frames)
            summary_frames_count = int(fps * 3)
            for _ in range(summary_frames_count):
                writer.write(summary_canvas)

        cap.release()
        if writer:
            writer.release()

        return {
            "input_video_path": str(input_file),
            "output_video_path": str(output_video_path),
            "excel_results_path": str(excel_results_path),
            "csv_results_path": str(csv_results_path),
            "best_frames_dir": str(best_frames_dir),
            "total_frames": total_frames,
            "total_goats": len(final_summary),
            "summary": final_summary,
        }

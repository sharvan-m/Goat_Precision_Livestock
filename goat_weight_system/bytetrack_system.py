"""
Goat Object Detection and Unique ID Tracking System using Roboflow supervision and ByteTrack.
Provides robust multi-object tracking across video frames, maintains consistent IDs through
occlusions/overlaps, produces frame-by-frame CSV tracking logs, and renders high-visibility HUD bounding boxes.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
import warnings
import time
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# Suppress supervision deprecation warnings cleanly
warnings.filterwarnings("ignore", category=FutureWarning)
import supervision as sv

from goat_weight_system.config import (
    DEFAULT_YOLO_WEIGHTS,
    YOLO_MODEL_PATH,
    COCO_GOAT_CLASSES,
)


class GoatByteTrackSystem:
    """
    Video processing pipeline for goat detection and persistent ByteTrack ID tracking.
    Links detections across frames to maintain stable unique IDs (Goat 1, Goat 2, ...).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.35,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 45,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        target_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Path to YOLO weights. Falls back to default weights if not found.
            conf_threshold: Detection confidence threshold (0.0 - 1.0).
            track_activation_threshold: ByteTrack confidence threshold for track activation.
            lost_track_buffer: Frames to keep lost tracks in memory during occlusions.
            minimum_matching_threshold: Minimum matching threshold for ByteTrack data association.
            minimum_consecutive_frames: Consecutive frames required before track is confirmed.
            target_classes: Target class IDs to detect/track.
            device: Compute device ('cuda' or 'cpu').
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve model weights
        weights = model_path if (model_path and Path(model_path).exists()) else (
            str(YOLO_MODEL_PATH) if Path(YOLO_MODEL_PATH).exists() else DEFAULT_YOLO_WEIGHTS
        )
        self.model = YOLO(weights)
        self.model.to(self.device)

        # Determine target goat classes
        if target_classes is not None:
            self.target_classes = target_classes
        else:
            # Check if model has a dedicated 'goat' class
            goat_indices = []
            for cid, cname in self.model.names.items():
                name_lower = cname.lower()
                if any(k in name_lower for k in ["goat", "sheep", "capra"]):
                    goat_indices.append(cid)
            self.target_classes = goat_indices if goat_indices else COCO_GOAT_CLASSES

        self.conf_threshold = conf_threshold
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.minimum_consecutive_frames = minimum_consecutive_frames

        # Roboflow Supervision Annotators
        # Newer supervision (>=0.21) dropped per-constructor color/thickness; handle both versions.
        try:
            self.box_annotator = sv.BoxAnnotator(
                thickness=3,
                color=sv.ColorPalette.DEFAULT,
            )
        except TypeError:
            self.box_annotator = sv.BoxAnnotator(thickness=3)

        try:
            self.label_annotator = sv.LabelAnnotator(
                text_scale=0.6,
                text_thickness=2,
                text_padding=6,
                color=sv.ColorPalette.DEFAULT,
            )
        except TypeError:
            self.label_annotator = sv.LabelAnnotator(
                text_scale=0.6,
                text_thickness=2,
                text_padding=6,
            )

    def _create_tracker(self, fps: float) -> sv.ByteTrack:
        """Instantiates a fresh ByteTrack instance configured with given video FPS."""
        return sv.ByteTrack(
            track_activation_threshold=self.track_activation_threshold,
            lost_track_buffer=int(self.lost_track_buffer),
            minimum_matching_threshold=float(self.minimum_matching_threshold),
            frame_rate=float(fps),
            minimum_consecutive_frames=int(self.minimum_consecutive_frames),
        )

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        output_csv_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, float, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Process video file: runs goat detection, associates ByteTrack IDs,
        draws bounding boxes & labels, and saves annotated video + frame CSV.

        Args:
            video_path: Path to input video file.
            output_video_path: Path to write annotated MP4. If None, saves in input video folder.
            output_csv_path: Path to write tracking CSV. If None, saves in input video folder.
            progress_callback: Optional callback(current_frame, total_frames, current_fps, active_goats).

        Returns:
            Dictionary with tracking metrics, paths, and per-goat summary statistics.
        """
        input_file = Path(video_path).resolve()
        if not input_file.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        input_dir = input_file.parent
        stem = input_file.stem

        if output_video_path is None:
            out_vid_path = input_dir / f"{stem}_bytetrack_output.mp4"
        else:
            out_vid_path = Path(output_video_path).resolve()

        if output_csv_path is None:
            out_csv_path = input_dir / f"{stem}_bytetrack_tracking.csv"
        else:
            out_csv_path = Path(output_csv_path).resolve()

        cap = cv2.VideoCapture(str(input_file))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if (fps and fps > 0) else 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (w, h))

        tracker = self._create_tracker(fps)

        csv_rows: List[Dict[str, Any]] = []
        # Per-goat metrics tracking: id -> list of confidences and frames
        goat_stats: Dict[int, Dict[str, Any]] = {}

        frame_idx = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # 1. Run YOLO inference on frame, filtering exclusively for goat class
            results = self.model(
                frame,
                conf=self.conf_threshold,
                classes=self.target_classes,
                device=self.device,
                verbose=False,
            )[0]

            # 2. Convert to Supervision Detections
            detections = sv.Detections.from_ultralytics(results)

            # 3. Update ByteTrack multi-object tracker
            # Maintains consistent tracking IDs even through partial occlusions and overlaps
            tracked_detections = tracker.update_with_detections(detections)

            # 4. Extract boxes, IDs, and confidences
            labels: List[str] = []
            active_ids_in_frame: List[int] = []

            if tracked_detections.tracker_id is not None and len(tracked_detections) > 0:
                for xyxy, conf, track_id in zip(
                    tracked_detections.xyxy,
                    tracked_detections.confidence,
                    tracked_detections.tracker_id,
                ):
                    tid = int(track_id)
                    conf_val = float(conf)
                    active_ids_in_frame.append(tid)

                    # Required format: Goat ID: 1 | Confidence: 0.94
                    label_str = f"Goat ID: {tid} | Confidence: {conf_val:.2f}"
                    labels.append(label_str)

                    # CSV Record: frame number, goat tracking ID, bounding-box coordinates, detection confidence
                    x1, y1, x2, y2 = [round(float(v), 2) for v in xyxy]
                    csv_rows.append({
                        "frame_number": frame_idx,
                        "goat_tracking_id": tid,
                        "class_name": "goat",
                        "detection_confidence": round(conf_val, 4),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "box_width": round(x2 - x1, 2),
                        "box_height": round(y2 - y1, 2),
                    })

                    # Accumulate stats
                    if tid not in goat_stats:
                        goat_stats[tid] = {
                            "goat_id": f"Goat {tid}",
                            "tracker_id": tid,
                            "first_frame": frame_idx,
                            "last_frame": frame_idx,
                            "total_detections": 1,
                            "confidences": [conf_val],
                        }
                    else:
                        goat_stats[tid]["last_frame"] = frame_idx
                        goat_stats[tid]["total_detections"] += 1
                        goat_stats[tid]["confidences"].append(conf_val)

                # 5. Draw bounding boxes and labels using Supervision
                annotated_frame = self.box_annotator.annotate(
                    scene=frame.copy(),
                    detections=tracked_detections,
                )
                annotated_frame = self.label_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections,
                    labels=labels,
                )
            else:
                annotated_frame = frame.copy()

            # Top HUD banner showing frame count and total active goats
            hud_text = f"Frame: {frame_idx}/{total_frames} | Active Goats: {len(active_ids_in_frame)} | ByteTrack"
            cv2.putText(
                annotated_frame,
                hud_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                hud_text,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )

            writer.write(annotated_frame)

            # Periodic progress callback
            if progress_callback and (frame_idx % 15 == 0 or frame_idx == total_frames):
                elapsed = time.time() - start_time
                current_fps = frame_idx / max(elapsed, 0.001)
                try:
                    progress_callback(frame_idx, total_frames, current_fps, len(goat_stats))
                except TypeError:
                    progress_callback(frame_idx, total_frames)

        cap.release()
        writer.release()

        # Save CSV file containing frame-by-frame goat detections and coordinates
        df_csv = pd.DataFrame(csv_rows)
        if df_csv.empty:
            df_csv = pd.DataFrame(columns=[
                "frame_number", "goat_tracking_id", "class_name",
                "detection_confidence", "x1", "y1", "x2", "y2", "box_width", "box_height"
            ])
        df_csv.to_csv(str(out_csv_path), index=False)

        # Compile per-goat final summary
        summary_list = []
        for tid, data in sorted(goat_stats.items(), key=lambda x: x[0]):
            confs = data["confidences"]
            summary_list.append({
                "Goat ID": f"Goat {tid}",
                "Tracker ID": tid,
                "Frames Seen": data["total_detections"],
                "First Frame": data["first_frame"],
                "Last Frame": data["last_frame"],
                "Duration (s)": round(data["total_detections"] / fps, 2),
                "Avg Confidence": round(float(np.mean(confs)), 3),
                "Max Confidence": round(float(np.max(confs)), 3),
            })

        return {
            "input_video_path": str(input_file),
            "output_video_path": str(out_vid_path),
            "output_csv_path": str(out_csv_path),
            "total_frames": total_frames,
            "total_unique_goats": len(goat_stats),
            "csv_rows_count": len(csv_rows),
            "goat_summary": summary_list,
        }

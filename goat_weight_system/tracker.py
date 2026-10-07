"""
Goat object tracking module.
Assigns and maintains persistent unique IDs for detected goats across video frames
using combined IoU and centroid proximity matching.
"""

from typing import List, Dict, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment
from goat_weight_system.detector import DetectionResult


class TrackedGoat:
    """Represents a single tracked goat instance across video frames."""

    def __init__(self, track_id: int, initial_detection: DetectionResult, frame_idx: int):
        self.track_id = track_id
        self.last_detection = initial_detection
        self.first_seen_frame = frame_idx
        self.last_seen_frame = frame_idx
        self.total_frames_seen = 1

    def update(self, detection: DetectionResult, frame_idx: int):
        self.last_detection = detection
        self.last_seen_frame = frame_idx
        self.total_frames_seen += 1
        detection.track_id = self.track_id


class GoatTracker:
    """
    Persistent Multi-Object Tracker for Goats.
    Uses Intersection over Union (IoU) and normalized centroid proximity matching to maintain stable Goat IDs.
    """

    def __init__(self, max_disappeared: int = 60, iou_threshold: float = 0.15, max_centroid_dist_px: float = 150.0):
        """
        Args:
            max_disappeared: Maximum consecutive frames a goat can be absent before its ID is retired.
            iou_threshold: Minimum IoU overlap to consider matching detections.
            max_centroid_dist_px: Maximum distance between centroids for proximity matching.
        """
        self.next_id = 1
        self.tracks: Dict[int, TrackedGoat] = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.max_centroid_dist_px = max_centroid_dist_px
        self.current_frame_idx = 0

    @staticmethod
    def _compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        """Calculate IoU between two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        denominator = float(boxAArea + boxBArea - interArea)
        if denominator <= 0:
            return 0.0
        return interArea / denominator

    @staticmethod
    def _compute_centroid_dist(boxA: np.ndarray, boxB: np.ndarray) -> float:
        """Calculate Euclidean distance between bounding box centroids."""
        cA = np.array([(boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0])
        cB = np.array([(boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0])
        return float(np.linalg.norm(cA - cB))

    def update(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Update tracker with current frame detections.

        Args:
            detections: List of DetectionResult objects in current frame.

        Returns:
            Updated list of DetectionResult objects with assigned track_id attributes.
        """
        self.current_frame_idx += 1

        active_track_ids = [
            tid for tid, t in self.tracks.items()
            if (self.current_frame_idx - t.last_seen_frame) <= self.max_disappeared
        ]

        if len(detections) == 0:
            return []

        if len(active_track_ids) == 0:
            for det in detections:
                det.track_id = self.next_id
                self.tracks[self.next_id] = TrackedGoat(self.next_id, det, self.current_frame_idx)
                self.next_id += 1
            return detections

        # Build cost matrix based on combined IoU and centroid distance
        cost_matrix = np.zeros((len(active_track_ids), len(detections)), dtype=np.float32)

        for i, tid in enumerate(active_track_ids):
            track_box = self.tracks[tid].last_detection.box
            for j, det in enumerate(detections):
                iou = self._compute_iou(track_box, det.box)
                dist = self._compute_centroid_dist(track_box, det.box)

                # Cost metric: 0.7 * (1 - IoU) + 0.3 * (norm_dist)
                norm_dist = min(1.0, dist / self.max_centroid_dist_px)
                cost_matrix[i, j] = 0.70 * (1.0 - iou) + 0.30 * norm_dist

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_detections = set()
        assigned_tracks = set()

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            tid = active_track_ids[r]
            det = detections[c]
            track_box = self.tracks[tid].last_detection.box
            iou = self._compute_iou(track_box, det.box)
            dist = self._compute_centroid_dist(track_box, det.box)

            if iou >= self.iou_threshold or dist <= self.max_centroid_dist_px:
                det.track_id = tid
                self.tracks[tid].update(det, self.current_frame_idx)
                assigned_tracks.add(tid)
                assigned_detections.add(c)

        for j, det in enumerate(detections):
            if j not in assigned_detections:
                det.track_id = self.next_id
                self.tracks[self.next_id] = TrackedGoat(self.next_id, det, self.current_frame_idx)
                self.next_id += 1

        return detections

    def get_track(self, track_id: int) -> Optional[TrackedGoat]:
        return self.tracks.get(track_id, None)

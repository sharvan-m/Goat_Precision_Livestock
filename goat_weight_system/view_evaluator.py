"""
Flexible view-quality evaluation engine for goat weight estimation.
Calculates a continuous 0 to 100 View Quality Score based on body visibility, orientation,
segmentation quality, goat resolution/size, image blur/confidence, and tracking stability.

Classifies views into:
  - GOOD (Score >= 75): GREEN annotation, reliable weight estimation.
  - ACCEPTABLE (Score 55-74): YELLOW annotation, usable view with medium confidence.
  - POOR (Score < 55): RED annotation, insufficient information for weight estimation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2

from goat_weight_system.config import (
    SCORE_THRESHOLD_GOOD,
    SCORE_THRESHOLD_ACCEPTABLE,
    MIN_BORDER_MARGIN_PX,
    MAX_INTERSECT_IOU,
)
from goat_weight_system.detector import DetectionResult


class ViewEvaluation:
    """Dataclass holding evaluation results and 0-100 View Quality Score."""

    def __init__(
        self,
        view_score: float,  # Range 0 to 100
        category: str,     # "GOOD", "ACCEPTABLE", or "POOR"
        can_estimate_weight: bool,
        orientation_angle: float,
        aspect_ratio: float,
        solidity: float,
        visibility_score: float,
        orientation_score: float,
        segmentation_score: float,
        size_score: float,
        image_quality_score: float,
        tracking_score: float,
        touches_border: bool,
        is_occluded: bool,
        rejection_reasons: List[str],
    ):
        self.view_score = float(np.clip(view_score, 0.0, 100.0))
        self.category = category
        self.can_estimate_weight = can_estimate_weight
        self.orientation_angle = orientation_angle
        self.aspect_ratio = aspect_ratio
        self.solidity = solidity
        self.visibility_score = visibility_score
        self.orientation_score = orientation_score
        self.segmentation_score = segmentation_score
        self.size_score = size_score
        self.image_quality_score = image_quality_score
        self.tracking_score = tracking_score
        self.touches_border = touches_border
        self.is_occluded = is_occluded
        self.rejection_reasons = rejection_reasons

        # Backward compatibility properties
        self.is_valid = can_estimate_weight
        self.status = category
        self.quality_score = self.view_score / 100.0


class BestViewEvaluator:
    """Evaluates goat view quality on a continuous 0 to 100 score scale."""

    def __init__(
        self,
        score_thresh_good: float = SCORE_THRESHOLD_GOOD,
        score_thresh_acceptable: float = SCORE_THRESHOLD_ACCEPTABLE,
        border_margin: int = MIN_BORDER_MARGIN_PX,
        max_overlap_iou: float = MAX_INTERSECT_IOU,
    ):
        self.score_thresh_good = score_thresh_good
        self.score_thresh_acceptable = score_thresh_acceptable
        self.border_margin = border_margin
        self.max_overlap_iou = max_overlap_iou

    @staticmethod
    def calculate_mask_orientation(mask: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate body orientation angle using ellipse fitting on mask contour.
        Returns angle relative to horizontal plane (0° to 90°).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 90.0, 0.0, 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        if len(largest_contour) < 5:
            x, y, w, h = cv2.boundingRect(largest_contour)
            angle = 0.0 if w >= h else 90.0
            return angle, float(max(w, h)), float(min(w, h))

        ellipse = cv2.fitEllipse(largest_contour)
        (cx, cy), (d1, d2), angle = ellipse

        if d1 < d2:
            major_axis = d2
            minor_axis = d1
            angle_from_horizontal = abs(angle - 90.0)
        else:
            major_axis = d1
            minor_axis = d2
            angle_from_horizontal = angle if angle <= 90 else 180.0 - angle

        angle_from_horizontal = float(np.clip(angle_from_horizontal, 0.0, 90.0))
        return angle_from_horizontal, float(major_axis), float(minor_axis)

    @staticmethod
    def calculate_solidity(mask: np.ndarray) -> float:
        """Calculate mask solidity: area / convex_hull_area."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)

        if hull_area <= 0:
            return 0.0
        return float(area / hull_area)

    @staticmethod
    def check_border_touch(box: np.ndarray, image_shape: Tuple[int, int], margin: int) -> bool:
        """Check if bounding box touches image boundaries."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = box
        return x1 <= margin or y1 <= margin or x2 >= (w - margin) or y2 >= (h - margin)

    @staticmethod
    def check_occlusion(target_box: np.ndarray, all_boxes: List[np.ndarray], max_iou: float) -> bool:
        """Check if target detection overlaps significantly with another animal box."""
        for other_box in all_boxes:
            if np.array_equal(target_box, other_box):
                continue
            xA = max(target_box[0], other_box[0])
            yA = max(target_box[1], other_box[1])
            xB = min(target_box[2], other_box[2])
            yB = min(target_box[3], other_box[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            targetArea = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
            if targetArea <= 0:
                continue
            if (interArea / float(targetArea)) > max_iou:
                return True
        return False

    @staticmethod
    def estimate_image_blur(frame: np.ndarray, box: np.ndarray) -> float:
        """Calculate Laplacian variance inside goat region for blur measurement."""
        x1, y1, x2, y2 = box.astype(int)
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return 50.0
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return variance

    def evaluate(
        self,
        detection: DetectionResult,
        image_shape: Tuple[int, int],
        all_detections: Optional[List[DetectionResult]] = None,
        frame: Optional[np.ndarray] = None,
        track_history_length: int = 1,
    ) -> ViewEvaluation:
        """
        Evaluate goat view quality and produce 0-100 View Quality Score.

        Components (Total 100 points):
          - Body Visibility (30 points)
          - Body Orientation (20 points)
          - Segmentation Quality (20 points)
          - Goat Resolution & Size (15 points)
          - Image Blur & Detection Confidence (10 points)
          - Tracking Stability (5 points)
        """
        box = detection.box
        mask = detection.mask
        h, w = image_shape[:2]
        rejection_reasons: List[str] = []

        # 1. Orientation Angle & Aspect Ratio
        angle, major_axis, minor_axis = self.calculate_mask_orientation(mask)
        bbox_w = float(box[2] - box[0])
        bbox_h = float(box[3] - box[1])
        aspect_ratio = bbox_w / max(bbox_h, 1.0)
        solidity = self.calculate_solidity(mask)

        touches_border = self.check_border_touch(box, image_shape, self.border_margin)
        all_boxes = [d.box for d in all_detections] if all_detections else []
        is_occluded = self.check_occlusion(box, all_boxes, self.max_overlap_iou)

        # -------------------------------------------------------------
        # Score Component 1: Body Visibility (30 points max)
        # -------------------------------------------------------------
        vis_score = 30.0
        if touches_border:
            vis_score -= 12.0
            rejection_reasons.append("Partially clipped at image boundary")
        if is_occluded:
            vis_score -= 15.0
            rejection_reasons.append("Occluded by overlapping animal")
        if solidity < 0.50:
            vis_score -= 8.0
        vis_score = float(np.clip(vis_score, 0.0, 30.0))

        # -------------------------------------------------------------
        # Score Component 2: Body Orientation (20 points max)
        # Accepts side views, diagonal side views, and moderate diagonal views.
        # Rejects only extreme frontal or extreme rear facing views.
        # -------------------------------------------------------------
        if angle <= 25.0:
            orient_score = 20.0  # Perfect side view
        elif angle <= 45.0:
            orient_score = 17.0  # Slightly diagonal side view
        elif angle <= 60.0:
            orient_score = 12.0  # Moderate diagonal view
        elif angle <= 75.0:
            orient_score = 6.0   # Severe diagonal view
        else:
            orient_score = 0.0   # Extreme frontal/rear view
            rejection_reasons.append(f"Frontal/rear orientation ({angle:.1f}° tilt)")

        if aspect_ratio < 0.80:
            orient_score *= 0.4
            rejection_reasons.append(f"Vertical profile (Aspect ratio {aspect_ratio:.2f})")
        orient_score = float(np.clip(orient_score, 0.0, 20.0))

        # -------------------------------------------------------------
        # Score Component 3: Segmentation Quality (20 points max)
        # -------------------------------------------------------------
        if solidity >= 0.80:
            seg_score = 20.0
        elif solidity >= 0.65:
            seg_score = 15.0
        elif solidity >= 0.50:
            seg_score = 10.0
        else:
            seg_score = 4.0
            rejection_reasons.append(f"Poor segmentation solidity ({solidity:.2f})")
        seg_score = float(np.clip(seg_score, 0.0, 20.0))

        # -------------------------------------------------------------
        # Score Component 4: Goat Resolution & Size (15 points max)
        # -------------------------------------------------------------
        bbox_area = bbox_w * bbox_h
        img_area = float(w * h)
        area_ratio = bbox_area / max(img_area, 1.0)

        if area_ratio >= 0.06:
            size_score = 15.0
        elif area_ratio >= 0.03:
            size_score = 11.0
        elif area_ratio >= 0.01:
            size_score = 6.0
        else:
            size_score = 2.0
            rejection_reasons.append("Goat too small in frame")
        size_score = float(np.clip(size_score, 0.0, 15.0))

        # -------------------------------------------------------------
        # Score Component 5: Image Quality & Blur (10 points max)
        # -------------------------------------------------------------
        conf_pts = detection.confidence * 6.0  # Up to 6 points
        blur_pts = 4.0
        if frame is not None:
            blur_var = self.estimate_image_blur(frame, box)
            if blur_var < 30.0:
                blur_pts = 1.0
                rejection_reasons.append("Blurry frame image")
            elif blur_var < 80.0:
                blur_pts = 2.5
        img_qual_score = float(np.clip(conf_pts + blur_pts, 0.0, 10.0))

        # -------------------------------------------------------------
        # Score Component 6: Tracking Stability (5 points max)
        # -------------------------------------------------------------
        track_score = min(5.0, float(track_history_length) * 1.5)

        # -------------------------------------------------------------
        # Final Total View Score (0 to 100)
        # -------------------------------------------------------------
        total_score = vis_score + orient_score + seg_score + size_score + img_qual_score + track_score
        total_score = float(np.clip(total_score, 0.0, 100.0))

        # Categorization based on View Score
        if total_score >= self.score_thresh_good:
            category = "GOOD"
            can_estimate = True
        elif total_score >= self.score_thresh_acceptable:
            category = "ACCEPTABLE"
            can_estimate = True
        else:
            category = "POOR"
            can_estimate = False
            if not rejection_reasons:
                rejection_reasons.append(f"Low view score ({total_score:.0f}/100)")

        return ViewEvaluation(
            view_score=total_score,
            category=category,
            can_estimate_weight=can_estimate,
            orientation_angle=angle,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            visibility_score=vis_score,
            orientation_score=orient_score,
            segmentation_score=seg_score,
            size_score=size_score,
            image_quality_score=img_qual_score,
            tracking_score=track_score,
            touches_border=touches_border,
            is_occluded=is_occluded,
            rejection_reasons=rejection_reasons,
        )

"""
Visualization HUD module for rendering GREEN (GOOD), YELLOW (ACCEPTABLE), and RED (POOR) view annotations.
Renders high-contrast overlays with 0-100 View Score, category, predicted weight, and confidence levels.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from goat_weight_system.config import (
    COLOR_GOOD,
    COLOR_ACCEPTABLE,
    COLOR_POOR,
    COLOR_TEXT,
    COLOR_MASK_GOOD,
    COLOR_MASK_ACCEPTABLE,
    COLOR_MASK_POOR,
)
from goat_weight_system.detector import DetectionResult
from goat_weight_system.view_evaluator import ViewEvaluation


# Distinct color palette for unique Goat IDs (BGR format)
GOAT_ID_COLORS = [
    (255, 140, 0),   # Deep Sky Blue
    (0, 215, 255),   # Golden Amber
    (255, 0, 200),   # Vibrant Violet / Magenta
    (0, 255, 128),   # Spring Emerald
    (0, 140, 255),   # Bright Orange
    (180, 105, 255), # Soft Pink
    (255, 255, 0),   # Electric Cyan
    (50, 205, 50),   # Lime Green
]


class GoatVisualizer:
    """Renders annotated frames with bounding boxes, masks, View Score, and weight HUD."""

    def __init__(self, alpha_mask: float = 0.40):
        self.alpha_mask = alpha_mask

    def draw_annotated_frame(
        self,
        frame: np.ndarray,
        detections: List[DetectionResult],
        evaluations: List[ViewEvaluation],
        weight_predictions: Optional[List[Optional[float]]] = None,
        confidences: Optional[List[Optional[str]]] = None,
        measurements_list: Optional[List[Optional[Dict[str, float]]]] = None,
    ) -> np.ndarray:
        """
        Draw annotations onto a copy of the input BGR image.
        Renders a prominent, high-contrast Unique Goat ID badge for each goat,
        segmentation mask, view quality color-coded bounding box, and informative HUD.

        Colors:
          - GREEN for GOOD view (Score >= 75)
          - YELLOW for ACCEPTABLE view (Score 55-74)
          - RED for POOR view (Score < 55)
        """
        annotated = frame.copy()
        h_img, w_img = annotated.shape[:2]

        for i, det in enumerate(detections):
            view_eval = evaluations[i] if i < len(evaluations) else None
            pred_weight = weight_predictions[i] if (weight_predictions and i < len(weight_predictions)) else None
            confidence_str = confidences[i] if (confidences and i < len(confidences)) else "LOW"
            measurements = measurements_list[i] if (measurements_list and i < len(measurements_list)) else None

            # Guaranteed Unique Goat ID number (never None, never 0)
            goat_id_num = det.track_id if (det.track_id is not None and det.track_id > 0) else (i + 1)
            det.track_id = goat_id_num
            goat_id_badge_text = f"GOAT #{goat_id_num:02d}"
            badge_color = GOAT_ID_COLORS[(goat_id_num - 1) % len(GOAT_ID_COLORS)]

            # Category & Color determination
            if view_eval is not None:
                category = view_eval.category
                v_score = view_eval.view_score
            else:
                category = "POOR"
                v_score = 0.0

            if category == "GOOD":
                color = COLOR_GOOD
                mask_color = COLOR_MASK_GOOD
            elif category == "ACCEPTABLE":
                color = COLOR_ACCEPTABLE
                mask_color = COLOR_MASK_ACCEPTABLE
            else:
                color = COLOR_POOR
                mask_color = COLOR_MASK_POOR

            # 1. Draw Segmentation Mask Overlay
            if det.mask is not None:
                mask_indices = det.mask > 0
                if np.any(mask_indices):
                    overlay = annotated.copy()
                    overlay[mask_indices] = mask_color
                    cv2.addWeighted(overlay, self.alpha_mask, annotated, 1.0 - self.alpha_mask, 0, annotated)

                    contours, _ = cv2.findContours((det.mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(annotated, contours, -1, color, 2)

            # 2. Draw Bounding Box
            x1, y1, x2, y2 = det.box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # 3. Draw High-Contrast Unique Goat ID Header Badge
            (b_w, b_h), _ = cv2.getTextSize(goat_id_badge_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
            badge_pad_x = 10
            badge_pad_y = 6
            badge_total_w = b_w + 2 * badge_pad_x
            badge_total_h = b_h + 2 * badge_pad_y

            # Place badge above box if space permits, otherwise inside top of box
            if y1 >= badge_total_h:
                badge_y1 = y1 - badge_total_h
                badge_y2 = y1
            else:
                badge_y1 = y1
                badge_y2 = y1 + badge_total_h

            badge_x1 = max(0, x1)
            badge_x2 = min(w_img, badge_x1 + badge_total_w)

            # Draw solid colored ID badge with dark outline
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), badge_color, -1)
            cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), (20, 20, 20), 1)

            # High-contrast bold text on badge
            cv2.putText(
                annotated,
                goat_id_badge_text,
                (badge_x1 + badge_pad_x, badge_y2 - badge_pad_y),
                cv2.FONT_HERSHEY_DUPLEX,
                0.65,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

            # 4-5. Informational HUD intentionally removed.
            # Weight, view score, confidence, measurements and other
            # per-goat information remain available for the final summary.
        return annotated


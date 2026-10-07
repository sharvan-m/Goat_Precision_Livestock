"""
Morphological measurement extraction module with perspective correction.
Extracts body length, withers height, chest girth proxy, body width, mask dimensions,
and perspective-corrected real-world dimensions along the goat's principal body axis.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

from goat_weight_system.config import DEFAULT_PIXEL_SCALE


class MeasurementExtractor:
    """Extracts morphological features from segmentation mask with perspective correction."""

    def __init__(self, pixel_scale_cm: float = DEFAULT_PIXEL_SCALE):
        """
        Args:
            pixel_scale_cm: Scale conversion factor (cm per pixel). Default 1.0 (uncalibrated px).
        """
        self.pixel_scale_cm = pixel_scale_cm

    def extract_features(
        self,
        mask: np.ndarray,
        box: np.ndarray,
        pixel_scale_override: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Extract complete morphological measurement dictionary from segmentation mask.

        Applies Perspective Correction along the goat's principal body orientation axis
        so that diagonal or angled postures yield accurate body length measurements.

        Args:
            mask: Binary uint8 mask (1 for goat pixel, 0 for background).
            box: Bounding box [x1, y1, x2, y2].
            pixel_scale_override: Optional scale override for current image frame.

        Returns:
            Dictionary containing extracted morphological features (both pixel and cm-scaled).
        """
        scale = pixel_scale_override if pixel_scale_override is not None else self.pixel_scale_cm

        binary_mask = (mask > 0).astype(np.uint8)

        # Bounding box measurements
        x1, y1, x2, y2 = box.astype(int)
        bbox_w = float(x2 - x1)
        bbox_h = float(y2 - y1)
        bbox_area = float(bbox_w * bbox_h)
        bbox_aspect_ratio = bbox_w / max(bbox_h, 1.0)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            body_length_px = bbox_w
            withers_height_px = bbox_h
            chest_girth_px = bbox_h * 0.85
            body_width_px = bbox_h * 0.70
            mask_area_px = bbox_area * 0.60
            perimeter_px = 2.0 * (bbox_w + bbox_h)
            hull_area_px = bbox_area
            solidity = 0.60
            major_axis_px = bbox_w
            minor_axis_px = bbox_h
            eccentricity = 0.80
            orientation_angle_deg = 0.0
        else:
            contour = max(contours, key=cv2.contourArea)
            mask_area_px = float(cv2.contourArea(contour))
            perimeter_px = float(cv2.arcLength(contour, True))

            hull = cv2.convexHull(contour)
            hull_area_px = float(cv2.contourArea(hull))
            solidity = float(mask_area_px / hull_area_px) if hull_area_px > 0 else 0.0

            # Rotated bounding box along principal body axis
            rect = cv2.minAreaRect(contour)
            (cx, cy), (w_rect, h_rect), rect_angle = rect
            raw_length_px = float(max(w_rect, h_rect))
            raw_width_px = float(min(w_rect, h_rect))

            # Withers height: vertical span of contour
            pts = contour[:, 0, :]
            y_min = np.min(pts[:, 1])
            y_max = np.max(pts[:, 1])
            withers_height_px = float(y_max - y_min)

            # Chest girth proxy: vertical slice across anterior 35% of horizontal length
            x_min = np.min(pts[:, 0])
            x_max = np.max(pts[:, 0])
            chest_x = int(x_min + 0.35 * (x_max - x_min))

            if 0 <= chest_x < binary_mask.shape[1]:
                col_indices = np.where(binary_mask[:, chest_x] > 0)[0]
                if len(col_indices) > 0:
                    chest_girth_px = float(col_indices[-1] - col_indices[0])
                else:
                    chest_girth_px = withers_height_px * 0.85
            else:
                chest_girth_px = withers_height_px * 0.85

            # Moments for ellipse axes & body tilt orientation
            if len(contour) >= 5:
                try:
                    (ellipse_cx, ellipse_cy), (d1, d2), angle = cv2.fitEllipse(contour)
                    major_axis_px = float(max(d1, d2))
                    minor_axis_px = float(min(d1, d2))

                    if d1 < d2:
                        orientation_angle_deg = abs(angle - 90.0)
                    else:
                        orientation_angle_deg = angle if angle <= 90 else 180.0 - angle
                except cv2.error:
                    major_axis_px = raw_length_px
                    minor_axis_px = raw_width_px
                    orientation_angle_deg = 0.0
            else:
                major_axis_px = raw_length_px
                minor_axis_px = raw_width_px
                orientation_angle_deg = 0.0

            # -------------------------------------------------------------
            # PERSPECTIVE CORRECTION:
            # Correct length for diagonal tilt angle shortening: L_corrected = L_raw / cos(theta)
            # -------------------------------------------------------------
            rad = np.radians(np.clip(orientation_angle_deg, 0.0, 60.0))
            cos_factor = max(np.cos(rad), 0.55)  # Cap shortening factor to avoid extreme inflation
            body_length_px = float(raw_length_px / cos_factor)
            body_width_px = float(raw_width_px)

            eccentricity = float(np.sqrt(1.0 - (minor_axis_px / max(major_axis_px, 1.0)) ** 2))

        # Metric conversions (cm / cm²)
        body_length_cm = body_length_px * scale
        withers_height_cm = withers_height_px * scale
        chest_girth_cm = chest_girth_px * scale
        body_width_cm = body_width_px * scale
        mask_area_cm2 = mask_area_px * (scale ** 2)

        return {
            # Pixel-space features
            "body_length_px": round(body_length_px, 2),
            "withers_height_px": round(withers_height_px, 2),
            "chest_girth_px": round(chest_girth_px, 2),
            "body_width_px": round(body_width_px, 2),
            "bbox_width_px": round(bbox_w, 2),
            "bbox_height_px": round(bbox_h, 2),
            "bbox_area_px": round(bbox_area, 2),
            "bbox_aspect_ratio": round(bbox_aspect_ratio, 3),
            "mask_area_px": round(mask_area_px, 2),
            "perimeter_px": round(perimeter_px, 2),
            "hull_area_px": round(hull_area_px, 2),
            "solidity": round(solidity, 4),
            "major_axis_px": round(major_axis_px, 2),
            "minor_axis_px": round(minor_axis_px, 2),
            "eccentricity": round(eccentricity, 4),
            "pixel_scale_cm": round(scale, 4),
            # Calibrated metric features (cm / cm²)
            "body_length": round(body_length_cm, 2),
            "height": round(withers_height_cm, 2),
            "girth": round(chest_girth_cm, 2),
            "body_width": round(body_width_cm, 2),
            "mask_area": round(mask_area_cm2, 2),
        }

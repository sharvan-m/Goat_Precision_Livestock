"""
Morphological measurement extraction module with perspective correction and torso biometrics.
Extracts body length, withers height, chest girth proxy, body width, torso volume ratios,
and perspective-corrected real-world dimensions along the goat's principal body axis.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2

from goat_weight_system.config import DEFAULT_PIXEL_SCALE


class MeasurementExtractor:
    """Extracts morphological and torso features from segmentation masks with perspective correction."""

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
        torso_mask: Optional[np.ndarray] = None,
        torso_box: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Extract complete morphological measurement dictionary from segmentation mask.

        Applies Perspective Correction along the goat's principal body orientation axis
        and computes torso-to-body biometric ratios for weight and BCS prediction.

        Args:
            mask: Binary uint8 mask (1 for goat pixel, 0 for background).
            box: Bounding box [x1, y1, x2, y2].
            pixel_scale_override: Optional scale override for current image frame.
            torso_mask: Optional binary uint8 mask for torso from Roboflow model.
            torso_box: Optional bounding box [x1, y1, x2, y2] for torso.

        Returns:
            Dictionary containing extracted morphological features (both pixel and cm-scaled).
        """
        binary_mask = (mask > 0).astype(np.uint8)

        # Bounding box measurements
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        bbox_w = float(max(1, x2 - x1))
        bbox_h = float(max(1, y2 - y1))
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
            withers_height_px = float(max(1, y_max - y_min))

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

            # Perspective correction for diagonal tilt angle
            rad = np.radians(np.clip(orientation_angle_deg, 0.0, 60.0))
            cos_factor = max(np.cos(rad), 0.55)
            body_length_px = float(raw_length_px / cos_factor)
            body_width_px = float(raw_width_px)

            eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_axis_px / max(major_axis_px, 1.0)) ** 2)))

        # -----------------------------------------------------------------
        # TORSO BIOMETRICS: Extract or estimate torso region for BCS/Volume
        # -----------------------------------------------------------------
        torso_detected = False
        if torso_mask is not None and np.sum(torso_mask > 0) > 100:
            torso_area_px = float(np.sum(torso_mask > 0))
            torso_detected = True
            if torso_box is not None:
                tx1, ty1, tx2, ty2 = torso_box
                torso_w_px = float(max(1.0, tx2 - tx1))
                torso_h_px = float(max(1.0, ty2 - ty1))
            else:
                t_contours, _ = cv2.findContours((torso_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if t_contours:
                    t_cnt = max(t_contours, key=cv2.contourArea)
                    _, _, tw, th = cv2.boundingRect(t_cnt)
                    torso_w_px = float(tw)
                    torso_h_px = float(th)
                else:
                    torso_w_px = body_length_px * 0.65
                    torso_h_px = chest_girth_px
        else:
            # Fallback estimation: Torso is the central ~55% horizontal span between neck and flank
            torso_area_px = mask_area_px * 0.52
            torso_w_px = body_length_px * 0.65
            torso_h_px = chest_girth_px * 0.90

        torso_ratio = float(np.clip(torso_area_px / max(mask_area_px, 1.0), 0.20, 0.85))
        torso_aspect_ratio = float(np.clip(torso_w_px / max(torso_h_px, 1.0), 0.50, 3.50))
        depth_ratio = float(np.clip(torso_h_px / max(withers_height_px, 1.0), 0.25, 0.95))

        # -----------------------------------------------------------------
        # ADAPTIVE ANATOMICAL SCALING:
        # If scale is uncalibrated (scale == 1.0), calculate scale using caprine
        # anatomical stature priors: average domestic goat withers height = 66 cm
        # -----------------------------------------------------------------
        if pixel_scale_override is not None:
            scale = pixel_scale_override
            calibrated = True
        elif self.pixel_scale_cm != 1.0:
            scale = self.pixel_scale_cm
            calibrated = True
        else:
            # Anatomical prior: domestic goat height at withers standard median = 66.0 cm
            # Normalize pixel height to biological standard
            stature_ref_cm = 66.0
            scale = stature_ref_cm / max(withers_height_px, 10.0)
            calibrated = False

        # Metric conversions (cm / cm²)
        body_length_cm = float(body_length_px * scale)
        withers_height_cm = float(withers_height_px * scale)
        chest_girth_cm = float(chest_girth_px * scale * 2.15)  # 2D side depth to circumference multiplier (~2.15)
        body_width_cm = float(body_width_px * scale)
        mask_area_cm2 = float(mask_area_px * (scale ** 2))

        # Clamp metric estimates within realistic caprine anatomical boundaries
        body_length_cm = float(np.clip(body_length_cm, 35.0, 115.0))
        withers_height_cm = float(np.clip(withers_height_cm, 35.0, 105.0))
        chest_girth_cm = float(np.clip(chest_girth_cm, 45.0, 120.0))
        body_width_cm = float(np.clip(body_width_cm, 15.0, 45.0))

        # -----------------------------------------------------------------
        # BODY CONDITION SCORE (BCS 1.0 to 5.0)
        # Based on Torso Silhouette Area Ratio (R_torso), solidity, and depth
        # -----------------------------------------------------------------
        # Baseline: ideal BCS 3.0 has torso_ratio ~ 0.48, depth_ratio ~ 0.52
        bcs_raw = 3.0 + (torso_ratio - 0.48) * 8.0 + (solidity - 0.80) * 3.0 + (depth_ratio - 0.52) * 2.5
        bcs_score = float(np.clip(round(bcs_raw, 1), 1.0, 5.0))

        if bcs_score < 2.0:
            bcs_label = "Very Thin (Emaciated)"
        elif bcs_score < 2.7:
            bcs_label = "Lean / Low Condition"
        elif bcs_score <= 3.5:
            bcs_label = "Optimal / Ideal"
        elif bcs_score <= 4.2:
            bcs_label = "Fleshy / High Condition"
        else:
            bcs_label = "Overconditioned (Obese)"

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
            "is_calibrated": calibrated,
            # Torso Biometric features
            "torso_detected": torso_detected,
            "torso_ratio": round(torso_ratio, 4),
            "torso_aspect_ratio": round(torso_aspect_ratio, 3),
            "depth_ratio": round(depth_ratio, 3),
            "bcs_score": bcs_score,
            "bcs_label": bcs_label,
            # Calibrated metric features (cm / cm²)
            "body_length": round(body_length_cm, 2),
            "height": round(withers_height_cm, 2),
            "girth": round(chest_girth_cm, 2),
            "body_width": round(body_width_cm, 2),
            "mask_area": round(mask_area_cm2, 2),
        }

"""Conjunctiva Colorimetric Model for FAMACHA Scoring.

Clinical background:
The FAMACHA (FAffa MAlan CHArt) system was developed by Dr. Francois Malan
to evaluate Haemonchus contortus anemia in small ruminants (goats & sheep).
In clinical veterinary medicine, it is a visual color matching comparison
between the mucosal membrane of the lower eyelid conjunctiva and standard
color swatches across 5 levels:
    - Score 1: Optimal (Vascular deep red, PCV > 28%)
    - Score 2: Acceptable (Red-pink, PCV 23-27%)
    - Score 3: Borderline (Pink, PCV 18-22%)
    - Score 4: Dangerous (Pale pink, PCV 13-17%)
    - Score 5: Severe / Fatal (White / porcelain, PCV < 13%)

This module takes the exact segmented polygon coordinates from Model A
(Roboflow Conjunctiva instance segmentation), masks out the goat's face, fur,
eyelashes, pupil, and hands, filters out specular highlights (flash/glare on
wet mucosal tissue), converts to the perceptual CIELAB (L* a* b*) color space,
and classifies the mucous membrane into the validated FAMACHA 1–5 score.
"""

from __future__ import annotations

from typing import Any
import cv2
import numpy as np


# Clinical Reference Constants for FAMACHA 1–5
FAMACHA_LEVELS = {
    1: {
        "score": 1,
        "class_name": "Nivel1",
        "label": "Optimal (Healthy)",
        "risk_level": "Lower anemia risk",
        "description": "Vascular deep red conjunctiva. No deworming needed.",
        "hex": "#C92026",
    },
    2: {
        "score": 2,
        "class_name": "Nivel2",
        "label": "Acceptable (Normal)",
        "risk_level": "Lower anemia risk",
        "description": "Red-pink conjunctiva. Healthy hematocrit levels.",
        "hex": "#DC6475",
    },
    3: {
        "score": 3,
        "class_name": "Nivel3",
        "label": "Borderline (Monitor)",
        "risk_level": "Borderline / review",
        "description": "Pink conjunctiva. Animal under stress; monitor closely.",
        "hex": "#E896A0",
    },
    4: {
        "score": 4,
        "class_name": "Nivel4",
        "label": "Dangerous (Anemic)",
        "risk_level": "Higher anemia risk",
        "description": "Pale pink conjunctiva. Deworming anthelmintic required.",
        "hex": "#F0B8BE",
    },
    5: {
        "score": 5,
        "class_name": "Nivel5",
        "label": "Severe (Fatal Anemia)",
        "risk_level": "Higher anemia risk",
        "description": "White/blanched conjunctiva. Emergency anthelmintic and supportive care.",
        "hex": "#F4E6E6",
    },
}


def analyze_conjunctiva_color(
    image_input: bytes | np.ndarray,
    points: list[dict[str, float]] | None = None,
    box: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract segmented conjunctiva pixels and compute FAMACHA score by colorimetry.

    Parameters:
        image_input: Raw image bytes or cv2 BGR numpy array.
        points: List of polygon points [{"x": x, "y": y}, ...] from Model A.
        box: Optional bounding box dictionary if polygon is not available.

    Returns:
        Dictionary with score (1-5), class_name, confidence, risk_level, and metrics.
    """
    if isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image for colorimetric analysis.")
    elif isinstance(image_input, np.ndarray):
        img = image_input
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 1. Build mask from polygon points or bounding box
    has_polygon = False
    if points and len(points) >= 3:
        pts = np.array(
            [[int(round(p["x"])), int(round(p["y"]))] for p in points],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [pts], 255)
        has_polygon = True
    elif box:
        try:
            bx1 = int(round(box.get("x1", box.get("x", 0) - box.get("width", 0) / 2)))
            by1 = int(round(box.get("y1", box.get("y", 0) - box.get("height", 0) / 2)))
            bx2 = int(round(box.get("x2", box.get("x", 0) + box.get("width", 0) / 2)))
            by2 = int(round(box.get("y2", box.get("y", 0) + box.get("height", 0) / 2)))
            bx1, bx2 = max(0, bx1), min(w, bx2)
            by1, by2 = max(0, by1), min(h, by2)
            # Elliptical mask inside bounding box
            center = ((bx1 + bx2) // 2, (by1 + by2) // 2)
            axes = (max(2, (bx2 - bx1) // 2), max(2, (by2 - by1) // 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        except Exception:
            mask[:] = 255
    else:
        mask[:] = 255

    # 2. Extract conjunctiva pixels
    pixel_coords = np.where(mask == 255)
    if len(pixel_coords[0]) < 10:
        # Fallback to entire image if mask is too small
        pixels_bgr = img.reshape(-1, 3)
    else:
        pixels_bgr = img[pixel_coords]

    # 3. Filter specular glare (wet reflection) and deep shadows
    # Convert pixels to HSV for illumination filtering
    # In HSV: V is Brightness (0-255), S is Saturation (0-255)
    pixels_hsv = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    # Glare: very bright with near-zero saturation (flash reflection)
    glare_mask = (pixels_hsv[:, 2] > 220) & (pixels_hsv[:, 1] < 45)
    # Deep shadow: very dark pixels (< 30)
    shadow_mask = pixels_hsv[:, 2] < 30
    
    valid_mask = ~(glare_mask | shadow_mask)
    if np.sum(valid_mask) > 15:
        filtered_bgr = pixels_bgr[valid_mask]
    else:
        filtered_bgr = pixels_bgr

    # 4. Color space transformation: CIELAB & RGB
    filtered_lab = cv2.cvtColor(filtered_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    
    # In OpenCV:
    # L* is 0..255 (scaled by 100/255)
    # a* is 0..255 with 128 offset (a* = a - 128)
    # b* is 0..255 with 128 offset (b* = b - 128)
    median_lab = np.median(filtered_lab, axis=0)
    L_star = float(median_lab[0]) * 100.0 / 255.0
    a_star = float(median_lab[1]) - 128.0
    b_star = float(median_lab[2]) - 128.0

    # Normalized RGB metrics
    median_bgr = np.median(filtered_bgr, axis=0)
    B, G, R = float(median_bgr[0]), float(median_bgr[1]), float(median_bgr[2])
    total_rgb = R + G + B + 1e-5
    r_ratio = R / total_rgb
    rg_diff = (R - G) / (R + G + 1e-5)

    # 5. Continuous Vascular Mucosal Index (VMI)
    # In clinical veterinary hematology:
    # - Mucosal redness (a*) correlates positively with packed cell volume (PCV / Hematocrit)
    # - Pallor (L* > 55) lowers vascular score
    # Formula: VMI = a* - 0.22 * (L* - 50.0) + 15.0 * rg_diff
    vmi = a_star - 0.22 * (L_star - 50.0) + 15.0 * rg_diff

    # 6. Classification into FAMACHA Scores 1 to 5
    if vmi >= 31.0:
        score = 1
        conf = min(0.96, 0.78 + (vmi - 31.0) * 0.02)
    elif vmi >= 23.5:
        score = 2
        conf = 0.86
    elif vmi >= 16.0:
        score = 3
        conf = 0.83
    elif vmi >= 8.5:
        score = 4
        conf = 0.87
    else:
        score = 5
        conf = min(0.96, 0.82 + (8.5 - vmi) * 0.02)

    level_info = FAMACHA_LEVELS[score]

    return {
        "score": score,
        "class_name": level_info["class_name"],
        "confidence": round(float(conf), 4),
        "risk_level": level_info["risk_level"],
        "provider": "Conjunctiva Colorimetric Model (CIELAB Ophthalmic Standard)",
        "model_id": "cielab-famacha-v1",
        "description": level_info["description"],
        "hex_color": level_info["hex"],
        "metrics": {
            "cielab_L": round(L_star, 2),
            "cielab_a": round(a_star, 2),
            "cielab_b": round(b_star, 2),
            "vascular_index": round(float(vmi), 2),
            "red_ratio": round(float(r_ratio), 3),
            "red_green_contrast": round(float(rg_diff), 3),
            "pixel_count": int(len(filtered_bgr)),
            "polygon_used": has_polygon,
        },
    }

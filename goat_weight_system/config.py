"""
Configuration settings and default parameters for the Goat Weight Estimation System.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

for d in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File Paths
CSV_LOG_PATH = DATA_DIR / "goat_weight_records.csv"
EXCEL_LOG_PATH = DATA_DIR / "goat_weight_records.xlsx"
DATASET_CSV_PATH = DATA_DIR / "dataset.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_weight_model.pkl"
YOLO_MODEL_PATH = MODELS_DIR / "goat_yolo_seg.pt"  # Fallback to yolov8n-seg.pt if custom not found

# YOLO Model Defaults
DEFAULT_YOLO_WEIGHTS = "yolov8n-seg.pt"
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# COCO animal class indices (In COCO: 16: cat, 17: dog, 18: sheep, 19: cow, 21: bear)
COCO_GOAT_CLASSES = [18, 19]  # sheep, cow (caprine/bovine shapes)

# Flexible View-Quality Evaluation Thresholds (0 to 100 Score)
SCORE_THRESHOLD_GOOD = 75.0         # Score >= 75 -> GOOD (GREEN, Estimate Weight: YES, Confidence: HIGH)
SCORE_THRESHOLD_ACCEPTABLE = 55.0   # Score 55-74 -> ACCEPTABLE (YELLOW, Estimate Weight: YES, Confidence: MEDIUM)
                                    # Score < 55 -> POOR (RED, Estimate Weight: NO, Confidence: LOW)

MIN_SOLIDITY = 0.60                # Minimum contour solidity before penalty
MIN_BORDER_MARGIN_PX = 10          # Margin to check frame clipping
MAX_INTERSECT_IOU = 0.20           # Max allowable overlap with other animals

# Category Colors (BGR format)
COLOR_GOOD = (0, 220, 0)           # GREEN
COLOR_ACCEPTABLE = (0, 215, 255)   # YELLOW
COLOR_POOR = (0, 0, 220)           # RED
COLOR_TEXT = (255, 255, 255)       # WHITE

COLOR_MASK_GOOD = (0, 255, 120)    # Semi-transparent light green
COLOR_MASK_ACCEPTABLE = (0, 235, 255) # Semi-transparent light yellow
COLOR_MASK_POOR = (0, 50, 255)    # Semi-transparent light red

# Legacy aliases for backwards compatibility
COLOR_BEST_VIEW = COLOR_GOOD
COLOR_INVALID_VIEW = COLOR_POOR

# Default Pixel Scale Calibration (pixels per cm)
DEFAULT_PIXEL_SCALE = 1.0          # 1.0 if uncalibrated (pixel measurements), > 0 if calibrated

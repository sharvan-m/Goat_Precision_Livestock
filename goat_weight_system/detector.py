"""
Goat detection and segmentation module using Ultralytics YOLO with automatic CUDA/GPU acceleration.
Isolates goats from input images/video frames and extracts segmentation masks and bounding boxes.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from goat_weight_system.config import (
    DEFAULT_YOLO_WEIGHTS,
    YOLO_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    COCO_GOAT_CLASSES,
)


class DetectionResult:
    """Dataclass holding detection information for a single detected goat."""

    def __init__(
        self,
        box: np.ndarray,          # [x1, y1, x2, y2]
        confidence: float,
        class_id: int,
        mask: Optional[np.ndarray] = None,  # Binary uint8 mask (same H, W as input frame)
        track_id: Optional[int] = None,
    ):
        self.box = box.astype(int)
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.mask = mask
        self.track_id = track_id

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def bbox_area(self) -> int:
        return self.width * self.height


class GoatDetector:
    """YOLO-based detector and segmenter optimized for goat detection with automatic GPU acceleration."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thresh: float = CONFIDENCE_THRESHOLD,
        target_classes: Optional[List[int]] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize GoatDetector.
        Args:
            model_path: Path to custom YOLO model weights (.pt). Defaults to YOLO_MODEL_PATH or DEFAULT_YOLO_WEIGHTS.
            conf_thresh: Minimum confidence score threshold.
            target_classes: List of class IDs to detect. If custom model, usually [0]. If COCO, COCO_GOAT_CLASSES.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.conf_thresh = conf_thresh

        # Auto-detect CUDA/GPU
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve model path
        if model_path and Path(model_path).exists():
            weights = str(model_path)
            self.is_custom = True
        elif YOLO_MODEL_PATH.exists():
            weights = str(YOLO_MODEL_PATH)
            self.is_custom = True
        else:
            weights = DEFAULT_YOLO_WEIGHTS
            self.is_custom = False

        self.model = YOLO(weights)
        try:
            self.model.to(self.device)
        except Exception:
            pass  # YOLO automatically manages device during predict

        # Set target classes
        if target_classes is not None:
            self.target_classes = target_classes
        else:
            if self.is_custom:
                self.target_classes = [0]
            else:
                self.target_classes = COCO_GOAT_CLASSES

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Perform detection and segmentation on a single image frame.

        Args:
            frame: BGR numpy image array.

        Returns:
            List of DetectionResult objects corresponding strictly to detected goats.
        """
        h, w = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            verbose=False,
            classes=self.target_classes,
            device=self.device,
        )

        detections: List[DetectionResult] = []

        if not results or len(results) == 0:
            return detections

        res = results[0]

        if res.boxes is None or len(res.boxes) == 0:
            return detections

        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy()

        has_masks = res.masks is not None and len(res.masks) > 0

        if has_masks:
            masks = res.masks.data.cpu().numpy()

        for i in range(len(boxes)):
            box = boxes[i]
            conf = confs[i]
            cls_id = int(cls_ids[i])

            mask_img = None
            if has_masks and i < len(masks):
                raw_mask = masks[i]
                mask_img = cv2.resize(
                    (raw_mask > 0.5).astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

            if mask_img is None:
                mask_img = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                mask_img[y1:y2, x1:x2] = 1

            detections.append(
                DetectionResult(
                    box=box,
                    confidence=conf,
                    class_id=cls_id,
                    mask=mask_img,
                )
            )

        return detections

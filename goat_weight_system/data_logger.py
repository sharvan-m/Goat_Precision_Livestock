"""
Data storage and logger module.
Logs goat measurement and weight prediction records to CSV and Excel files.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import pandas as pd

from goat_weight_system.config import CSV_LOG_PATH, EXCEL_LOG_PATH


class DataLogger:
    """Logs successfully estimated goat weights to CSV and Excel files."""

    COLUMNS = [
        "timestamp",
        "goat_id",
        "image_filename",
        "predicted_weight_kg",
        "actual_weight_kg",
        "absolute_error_kg",
        "view_score",
        "view_category",
        "confidence_level",
        "body_length_cm",
        "height_cm",
        "girth_cm",
        "body_width_cm",
        "segmentation_area_cm2",
        "model_version",
    ]

    def __init__(
        self,
        csv_path: Optional[str] = None,
        excel_path: Optional[str] = None,
        model_version: str = "v2.0-FlexibleView",
    ):
        self.csv_path = Path(csv_path) if csv_path else CSV_LOG_PATH
        self.excel_path = Path(excel_path) if excel_path else EXCEL_LOG_PATH
        self.model_version = model_version

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.excel_path.parent.mkdir(parents=True, exist_ok=True)

    def log_record(
        self,
        goat_id: str,
        image_filename: str,
        predicted_weight: float,
        measurements: Dict[str, float],
        view_score: float,
        view_category: str,
        confidence_level: str,
        actual_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Appends a new measurement record to CSV and Excel files."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        abs_error = None
        if actual_weight is not None and not pd.isna(actual_weight):
            abs_error = round(abs(predicted_weight - actual_weight), 2)

        record = {
            "timestamp": now_str,
            "goat_id": goat_id,
            "image_filename": image_filename,
            "predicted_weight_kg": round(predicted_weight, 2),
            "actual_weight_kg": round(actual_weight, 2) if actual_weight is not None else None,
            "absolute_error_kg": abs_error,
            "view_score": round(view_score, 1),
            "view_category": view_category,
            "confidence_level": confidence_level,
            "body_length_cm": round(measurements.get("body_length", 0.0), 2),
            "height_cm": round(measurements.get("height", 0.0), 2),
            "girth_cm": round(measurements.get("girth", 0.0), 2),
            "body_width_cm": round(measurements.get("body_width", 0.0), 2),
            "segmentation_area_cm2": round(measurements.get("mask_area", 0.0), 2),
            "model_version": self.model_version,
        }

        new_df = pd.DataFrame([record])

        # 1. CSV
        if self.csv_path.exists():
            new_df.to_csv(self.csv_path, mode="a", header=False, index=False)
        else:
            new_df.to_csv(self.csv_path, mode="w", header=True, index=False)

        # 2. Excel
        if self.excel_path.exists():
            try:
                existing_df = pd.read_excel(self.excel_path)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_excel(self.excel_path, index=False)
            except Exception:
                new_df.to_excel(self.excel_path, index=False)
        else:
            new_df.to_excel(self.excel_path, index=False)

        return record

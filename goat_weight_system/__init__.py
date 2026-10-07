"""
Goat Weight Estimation Package
"""

from goat_weight_system.config import (
    DEFAULT_YOLO_WEIGHTS,
    YOLO_MODEL_PATH,
    BEST_MODEL_PATH,
    CSV_LOG_PATH,
    EXCEL_LOG_PATH,
)
from goat_weight_system.detector import GoatDetector, DetectionResult
from goat_weight_system.tracker import GoatTracker, TrackedGoat
from goat_weight_system.view_evaluator import BestViewEvaluator, ViewEvaluation
from goat_weight_system.measurement_extractor import MeasurementExtractor
from goat_weight_system.dataset_manager import DatasetManager
from goat_weight_system.regression_models import GoatWeightRegressor, ModelBenchmarkEngine, compute_metrics
from goat_weight_system.evaluation import ConfidenceCalculator, format_error_display
from goat_weight_system.visualizer import GoatVisualizer
from goat_weight_system.data_logger import DataLogger
from goat_weight_system.video_processor import VideoProcessor
from goat_weight_system.bytetrack_system import GoatByteTrackSystem

__all__ = [
    "GoatDetector",
    "DetectionResult",
    "GoatTracker",
    "TrackedGoat",
    "BestViewEvaluator",
    "ViewEvaluation",
    "MeasurementExtractor",
    "DatasetManager",
    "GoatWeightRegressor",
    "ModelBenchmarkEngine",
    "compute_metrics",
    "ConfidenceCalculator",
    "format_error_display",
    "GoatVisualizer",
    "DataLogger",
    "VideoProcessor",
    "GoatByteTrackSystem",
]

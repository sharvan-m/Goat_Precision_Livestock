"""Tests for local vs cloud detector selection and comparison mode."""

import cv2
import numpy as np
import pytest

from backend.services import anemia_pipeline as pipe


def make_test_image() -> bytes:
    img = np.full((120, 120, 3), 150, np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    return enc.tobytes()


def test_local_detector_gating_with_bounding_box():
    """Verify that a local detection with bounding box and class 'conjunctiva' passes gate."""
    box_pred = {
        "class": "conjunctiva",
        "confidence": 0.85,
        "x": 60.0,
        "y": 60.0,
        "width": 40.0,
        "height": 30.0,
        "points": [
            {"x": 40.0, "y": 45.0},
            {"x": 80.0, "y": 45.0},
            {"x": 80.0, "y": 75.0},
            {"x": 40.0, "y": 75.0},
        ],
    }

    best, reason = pipe.evaluate_conjunctiva({"predictions": [box_pred]})
    assert reason is None
    assert best is not None
    assert best["class"] == "conjunctiva"
    assert best["confidence"] == 0.85


def test_analyze_with_custom_local_fn():
    """Verify analyze with detector='local' using a mocked local detector."""
    def fake_local_detector(_bytes):
        return {
            "model_id": "local:conjunctiva_best.pt",
            "provider": "Our Local Model (YOLO11n)",
            "model_type": "bounding_box",
            "predictions": [
                {
                    "class": "conjunctiva",
                    "confidence": 0.92,
                    "x": 50.0,
                    "y": 50.0,
                    "width": 30.0,
                    "height": 20.0,
                    "points": [{"x": 35.0, "y": 40.0}, {"x": 65.0, "y": 40.0}, {"x": 65.0, "y": 60.0}, {"x": 35.0, "y": 60.0}],
                }
            ],
            "prediction_count": 1,
            "annotated_image": "data:image/jpeg;base64,LOCAL_FAKE",
        }

    out = pipe.analyze(make_test_image(), conjunctiva_fn=fake_local_detector)
    assert out["status"] == "ok"
    assert out["conjunctiva"]["detected"] is True
    assert out["conjunctiva"]["best_confidence"] == 0.92
    assert out["conjunctiva"]["annotated_image"] == "data:image/jpeg;base64,LOCAL_FAKE"

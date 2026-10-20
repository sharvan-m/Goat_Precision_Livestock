"""Tests for the two-model anemia pipeline. Both Roboflow models are faked."""

import cv2
import numpy as np
import pytest

from backend.services import anemia_pipeline as pipe
from backend.services import famacha_roboflow as fam

POLY = [{"x": 20, "y": 20}, {"x": 80, "y": 25}, {"x": 70, "y": 70}, {"x": 25, "y": 60}]


def make_image() -> bytes:
    img = np.full((100, 100, 3), 120, np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    return enc.tobytes()


def pred(conf=0.8, cls="sheep-conjunctiva", points=POLY):
    p = {"class": cls, "confidence": conf, "class_id": 0}
    if points is not None:
        p["points"] = points
    return p


def model_a_returning(predictions, calls=None):
    def fn(_bytes):
        if calls is not None:
            calls.append(1)
        return {
            "model_id": "fake-a/1",
            "provider": "fake",
            "predictions": predictions,
            "prediction_count": len(predictions),
            "annotated_image": "data:image/jpeg;base64,AAAA",
        }

    return fn


def model_b_ok(calls=None, score=3):
    def fn(data):
        if calls is not None:
            calls.append(data)
        return {"model_id": "fake-b/1", "provider": "fake", "class_name": "x", "confidence": 0.9, "score": score}

    return fn


@pytest.fixture(autouse=True)
def default_gate(monkeypatch):
    monkeypatch.setattr(pipe, "CONJUNCTIVA_CLASS", "sheep-conjunctiva")
    monkeypatch.delenv("CONJUNCTIVA_MIN_CONFIDENCE", raising=False)


# ---------------------------------------------------------------- gating

@pytest.mark.parametrize(
    "predictions",
    [
        [],
        [pred(cls="goat-eye")],
        [pred(conf=0.30)],
        [pred(points=None)],
        [pred(points=POLY[:2])],
    ],
    ids=["none", "wrong-class", "low-confidence", "no-mask", "tiny-mask"],
)
def test_no_valid_detection_means_model_b_not_called_and_no_score(predictions):
    b_calls = []
    out = pipe.analyze(make_image(), model_a_returning(predictions), model_b_ok(b_calls), "full")

    assert b_calls == []
    assert out["conjunctiva"]["detected"] is False
    assert out["famacha"]["attempted"] is False
    assert out["famacha_score"] is None
    assert out["predictions"] == []
    assert out["message"] == pipe.MSG_NOT_DETECTED


def test_valid_detection_calls_each_model_once_and_returns_score():
    a_calls, b_calls = [], []
    out = pipe.analyze(
        make_image(), model_a_returning([pred(0.4), pred(0.9)], a_calls), model_b_ok(b_calls, 4), "full"
    )

    assert len(a_calls) == 1 and len(b_calls) == 1
    assert out["conjunctiva"]["detected"] is True
    assert out["conjunctiva"]["best_confidence"] == 0.9
    assert out["famacha"]["success"] is True
    assert out["famacha_score"] == 4
    assert out["risk_level"] == "Higher anemia risk"
    assert out["predictions"][0]["confidence"] == 90.0


def test_low_confidence_extra_prediction_is_ignored_best_is_used():
    out = pipe.analyze(make_image(), model_a_returning([pred(0.64), pred(0.48)]), model_b_ok(), "full")
    assert out["conjunctiva"]["best_confidence"] == 0.64
    assert out["famacha_score"] == 3


# ---------------------------------------------------------------- failures

def test_model_a_exception_means_no_score_and_model_b_not_called():
    def boom(_):
        raise RuntimeError("network down")

    b_calls = []
    out = pipe.analyze(make_image(), boom, model_b_ok(b_calls), "full")

    assert b_calls == []
    assert out["status"] == "error"
    assert out["conjunctiva"]["success"] is False
    assert "network down" in out["conjunctiva"]["error"]
    assert out["famacha_score"] is None
    assert out["message"] == pipe.MSG_DETECTION_FAILED


def test_model_b_exception_means_no_score_but_detection_stays_visible():
    def boom(_):
        raise fam.FamachaRoboflowError("scoring exploded")

    out = pipe.analyze(make_image(), model_a_returning([pred()]), boom, "full")

    assert out["conjunctiva"]["detected"] is True
    assert out["annotated_image"] == "data:image/jpeg;base64,AAAA"
    assert out["famacha"]["success"] is False
    assert out["famacha"]["error"] == "scoring exploded"
    assert out["famacha_score"] is None
    assert out["predictions"] == []
    assert out["message"] == pipe.MSG_SCORING_FAILED


def test_model_b_local_model_produces_score(monkeypatch):
    for name in ("FAMACHA_ROBOFLOW_MODEL_ID", "FAMACHA_INPUT_MODE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(fam, "CLASS_TO_SCORE", {})

    out = pipe.analyze(make_image(), model_a_returning([pred()]))

    assert out["famacha_score"] is not None
    assert 1 <= out["famacha_score"] <= 5
    assert out["famacha"]["success"] is True


def test_each_request_is_independent_no_stale_score():
    ok = pipe.analyze(make_image(), model_a_returning([pred()]), model_b_ok(), "full")
    bad = pipe.analyze(make_image(), model_a_returning([]), model_b_ok(), "full")
    assert ok["famacha_score"] == 3
    assert bad["famacha_score"] is None


# ---------------------------------------------------------------- crop mode

def test_crop_mode_sends_cropped_image_and_full_mode_sends_original():
    original = make_image()
    crop_calls, full_calls = [], []
    pipe.analyze(original, model_a_returning([pred()]), model_b_ok(crop_calls), "crop")
    pipe.analyze(original, model_a_returning([pred()]), model_b_ok(full_calls), "full")

    assert full_calls[0] == original
    cropped = cv2.imdecode(np.frombuffer(crop_calls[0], np.uint8), cv2.IMREAD_COLOR)
    assert cropped is not None
    assert cropped.shape[0] < 100 and cropped.shape[1] < 100


# ---------------------------------------------------------------- Model B parsing

def test_parse_single_label_classification():
    assert fam.parse_top_prediction({"top": "A", "confidence": 0.7, "predictions": []}) == ("A", 0.7)


def test_parse_list_and_dict_shapes_pick_highest_confidence():
    assert fam.parse_top_prediction(
        {"predictions": [{"class": "A", "confidence": 0.2}, {"class": "B", "confidence": 0.6}]}
    ) == ("B", 0.6)
    assert fam.parse_top_prediction(
        {"predictions": {"A": {"confidence": 0.9}, "B": {"confidence": 0.1}}}
    ) == ("A", 0.9)


@pytest.mark.parametrize("raw", [{}, {"predictions": []}, {"top": "A", "confidence": 1.7}, "nope"])
def test_parse_rejects_empty_or_invalid_output(raw):
    with pytest.raises(fam.FamachaRoboflowError):
        fam.parse_top_prediction(raw)


def test_class_mapping_uses_real_names_and_rejects_unknown(monkeypatch):
    monkeypatch.setattr(fam, "CLASS_TO_SCORE", {"famacha_2": 2})
    assert fam.map_class_to_score(" FAMACHA_2 ") == 2
    with pytest.raises(fam.FamachaRoboflowError):
        fam.map_class_to_score("famacha_9")


def test_run_famacha_model_refuses_to_run_when_not_configured(monkeypatch):
    monkeypatch.delenv("FAMACHA_ROBOFLOW_MODEL_ID", raising=False)
    with pytest.raises(fam.FamachaRoboflowError):
        fam.run_famacha_model(b"x")


def test_status_reports_both_models():
    s = pipe.status()
    assert s["available"] is True
    assert "conjunctiva" in s and "famacha" in s
    assert s["famacha"]["configured"] is True

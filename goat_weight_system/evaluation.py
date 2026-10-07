"""
Evaluation metrics and confidence calculation module.
Calculates prediction confidence levels (HIGH, MEDIUM, LOW) based on 0-100 View Score and category.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from goat_weight_system.view_evaluator import ViewEvaluation


class ConfidenceCalculator:
    """Calculates weight prediction confidence rating (HIGH, MEDIUM, LOW) and formats output displays."""

    @staticmethod
    def calculate_confidence(
        view_eval: ViewEvaluation,
        predicted_weight: float,
        detection_confidence: float = 0.85,
    ) -> Tuple[str, float]:
        """
        Calculate confidence category and normalized confidence score (0.0 to 1.0).

        Categories:
          - GOOD view (Score >= 75): HIGH confidence (>= 0.80)
          - ACCEPTABLE view (Score 55-74): MEDIUM confidence (0.55 - 0.79)
          - POOR view (Score < 55): LOW confidence (< 0.55)

        Returns:
            Tuple of (confidence_str, score_float) where confidence_str is "HIGH", "MEDIUM", or "LOW".
        """
        if not view_eval.can_estimate_weight or view_eval.category == "POOR":
            return "LOW", round(view_eval.view_score / 200.0, 2)

        # Base confidence calculated from 0-100 View Score + detection confidence
        raw_score = 0.70 * (view_eval.view_score / 100.0) + 0.30 * detection_confidence

        # Weight range sanity check penalty
        if predicted_weight < 10.0 or predicted_weight > 130.0:
            raw_score *= 0.75

        final_score = float(np.clip(raw_score, 0.0, 1.0))

        if view_eval.category == "GOOD" and final_score >= 0.75:
            category = "HIGH"
        elif view_eval.category in ["GOOD", "ACCEPTABLE"] and final_score >= 0.50:
            category = "MEDIUM"
        else:
            category = "LOW"

        return category, round(final_score, 2)


def format_error_display(
    predicted_weight: float,
    actual_weight: Optional[float] = None,
) -> Dict[str, str]:
    """
    Formats actual weight, predicted weight, and absolute error text.

    Example output:
        Actual Weight: 43.0 kg
        Predicted Weight: 42.6 kg
        Error: ±0.4 kg
    """
    pred_str = f"{predicted_weight:.1f} kg"

    if actual_weight is not None and not np.isnan(actual_weight):
        act_str = f"{actual_weight:.1f} kg"
        abs_err = abs(predicted_weight - actual_weight)
        err_str = f"±{abs_err:.1f} kg"
    else:
        act_str = "N/A"
        err_str = "N/A"

    return {
        "actual_weight_display": act_str,
        "predicted_weight_display": pred_str,
        "error_display": err_str,
    }

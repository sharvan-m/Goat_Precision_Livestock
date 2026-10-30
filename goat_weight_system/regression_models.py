"""
Regression model suite and benchmarking engine for goat weight estimation.
Compares Random Forest, XGBoost, LightGBM, Gradient Boosting, and Neural Network (MLP) regressors
augmented with veterinary body condition scoring (BCS) and robust missing-feature handling.
"""

from typing import Dict, Any, Tuple, Optional, List
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error as _rmse_fn
except ImportError:
    import math
    def _rmse_fn(y_true, y_pred):
        return math.sqrt(mean_squared_error(y_true, y_pred))

import xgboost as xgb
import lightgbm as lgb

from goat_weight_system.config import BEST_MODEL_PATH
from goat_weight_system.dataset_manager import DatasetManager


FEATURE_DEFAULTS = {
    "body_length": 65.0,
    "height": 62.0,
    "girth": 68.0,
    "body_width": 24.0,
    "mask_area": 3500.0,
    "solidity": 0.82,
    "bbox_aspect_ratio": 1.25,
    "major_axis_px": 500.0,
    "minor_axis_px": 450.0,
    "eccentricity": 0.75,
    "torso_ratio": 0.48,
    "torso_aspect_ratio": 1.50,
    "depth_ratio": 0.52,
}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R2, MAPE, and MPE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(_rmse_fn(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    non_zero = y_true != 0
    mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
    mpe = float(np.mean((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]) * 100.0)

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
        "MPE": round(mpe, 2),
    }


class GoatWeightRegressor:
    """Wrapper class for goat weight regression model with preprocessing and BCS derivation."""

    def __init__(self, model_name: str, estimator: Any):
        self.model_name = model_name
        self.estimator = estimator
        self.scaler = StandardScaler()
        self.feature_names = DatasetManager.FEATURE_COLUMNS
        self.is_fitted = False
        self.rmse: float = 1.85  # Model standard error for confidence intervals

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit scaler and regression model."""
        X_df = X.copy()
        for col in self.feature_names:
            if col not in X_df.columns:
                X_df[col] = FEATURE_DEFAULTS.get(col, 0.0)
        X_feats = X_df[self.feature_names]
        X_scaled = self.scaler.fit_transform(X_feats)
        self.estimator.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: Any) -> np.ndarray:
        """Predict continuous live weight (kg) from features DataFrame or Dict."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")

        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        else:
            X_df = X.copy()

        for col in self.feature_names:
            if col not in X_df.columns:
                X_df[col] = FEATURE_DEFAULTS.get(col, 0.0)

        X_feats = X_df[self.feature_names]
        X_scaled = self.scaler.transform(X_feats)
        preds = self.estimator.predict(X_scaled)
        return np.clip(preds, 5.0, 150.0)

    def predict_full(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute continuous live weight, Body Condition Score (BCS), and confidence margin.
        """
        pred_arr = self.predict(measurements)
        weight_kg = round(float(pred_arr[0]), 2)

        # Derive Body Condition Score (BCS 1.0 - 5.0)
        # toros_ratio ~ 0.48 is ideal BCS 3.0
        torso_ratio = float(measurements.get("torso_ratio", FEATURE_DEFAULTS["torso_ratio"]))
        solidity = float(measurements.get("solidity", FEATURE_DEFAULTS["solidity"]))
        depth_ratio = float(measurements.get("depth_ratio", FEATURE_DEFAULTS["depth_ratio"]))

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

        # 95% Confidence margin (± 1.96 * RMSE)
        margin_kg = round(float(1.96 * self.rmse), 1)

        return {
            "weight_kg": weight_kg,
            "margin_kg": margin_kg,
            "bcs_score": bcs_score,
            "bcs_label": bcs_label,
            "model_name": self.model_name,
        }

    def save(self, filepath: Optional[str] = None):
        """Save model pipeline to disk."""
        target_path = Path(filepath) if filepath else BEST_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "GoatWeightRegressor":
        """Load trained model pipeline from disk."""
        target_path = Path(filepath) if filepath else BEST_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at: {target_path}")
        with open(target_path, "rb") as f:
            model = pickle.load(f)
        return model


class ModelBenchmarkEngine:
    """Trains and benchmarks 5 regression models on train/val/test splits."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, GoatWeightRegressor] = {
            "Random Forest": GoatWeightRegressor(
                "Random Forest",
                RandomForestRegressor(n_estimators=180, max_depth=12, random_state=random_state)
            ),
            "XGBoost": GoatWeightRegressor(
                "XGBoost",
                xgb.XGBRegressor(n_estimators=180, max_depth=6, learning_rate=0.04, random_state=random_state)
            ),
            "LightGBM": GoatWeightRegressor(
                "LightGBM",
                lgb.LGBMRegressor(n_estimators=180, max_depth=6, learning_rate=0.04, random_state=random_state, verbose=-1)
            ),
            "Gradient Boosting": GoatWeightRegressor(
                "Gradient Boosting",
                GradientBoostingRegressor(n_estimators=180, max_depth=5, learning_rate=0.04, random_state=random_state)
            ),
            "Neural Network (MLP)": GoatWeightRegressor(
                "Neural Network (MLP)",
                MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=600, random_state=random_state)
            ),
        }
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[GoatWeightRegressor] = None

    def benchmark(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Train all models, evaluate metrics on Validation and Test sets, and select the best model based on Validation MAE.
        """
        results = []
        target_col = DatasetManager.TARGET_COLUMN

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        best_val_mae = float("inf")

        for name, model in self.models.items():
            model.fit(train_df, y_train)

            val_preds = model.predict(val_df)
            test_preds = model.predict(test_df)

            val_metrics = compute_metrics(y_val, val_preds)
            test_metrics = compute_metrics(y_test, test_preds)

            model.rmse = test_metrics["RMSE"]

            results.append({
                "Model": name,
                "Val MAE (kg)": val_metrics["MAE"],
                "Val RMSE (kg)": val_metrics["RMSE"],
                "Val R²": val_metrics["R2"],
                "Val MAPE (%)": val_metrics["MAPE"],
                "Test MAE (kg)": test_metrics["MAE"],
                "Test RMSE (kg)": test_metrics["RMSE"],
                "Test R²": test_metrics["R2"],
                "Test MAPE (%)": test_metrics["MAPE"],
            })

            if val_metrics["MAE"] < best_val_mae:
                best_val_mae = val_metrics["MAE"]
                self.best_model_name = name
                self.best_model = model

        if self.best_model:
            self.best_model.save(BEST_MODEL_PATH)

        comparison_df = pd.DataFrame(results).sort_values("Val MAE (kg)").reset_index(drop=True)
        return comparison_df

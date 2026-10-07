"""
Regression model suite and benchmarking engine for goat weight estimation.
Compares Random Forest, XGBoost, LightGBM, Gradient Boosting, and Neural Network (MLP) regressors.
"""

from typing import Dict, Any, Tuple, Optional, List
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R2, MAPE, and MPE."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(_rmse_fn(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Avoid division by zero in MAPE/MPE
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
    """Wrapper class for goat weight regression model with preprocessing."""

    def __init__(self, model_name: str, estimator: Any):
        self.model_name = model_name
        self.estimator = estimator
        self.scaler = StandardScaler()
        self.feature_names = DatasetManager.FEATURE_COLUMNS
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit scaler and regression model."""
        X_feats = X[self.feature_names]
        X_scaled = self.scaler.fit_transform(X_feats)
        self.estimator.fit(X_scaled, y)
        self.is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict weight (kg) from features DataFrame or Dict."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")

        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        else:
            X_df = X

        X_feats = X_df[self.feature_names]
        X_scaled = self.scaler.transform(X_feats)
        preds = self.estimator.predict(X_scaled)
        return np.clip(preds, 5.0, 150.0)

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
                RandomForestRegressor(n_estimators=150, max_depth=10, random_state=random_state)
            ),
            "XGBoost": GoatWeightRegressor(
                "XGBoost",
                xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=random_state)
            ),
            "LightGBM": GoatWeightRegressor(
                "LightGBM",
                lgb.LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=random_state, verbose=-1)
            ),
            "Gradient Boosting": GoatWeightRegressor(
                "Gradient Boosting",
                GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=random_state)
            ),
            "Neural Network (MLP)": GoatWeightRegressor(
                "Neural Network (MLP)",
                MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state)
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

        Returns:
            Comparison DataFrame summarizing metrics across models.
        """
        results = []
        feature_cols = DatasetManager.FEATURE_COLUMNS
        target_col = DatasetManager.TARGET_COLUMN

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        best_val_mae = float("inf")

        for name, model in self.models.items():
            # Fit model on training data
            model.fit(train_df, y_train)

            # Predict on val and test
            val_preds = model.predict(val_df)
            test_preds = model.predict(test_df)

            val_metrics = compute_metrics(y_val, val_preds)
            test_metrics = compute_metrics(y_test, test_preds)

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

            # Model selection based on measured validation performance (lowest MAE)
            if val_metrics["MAE"] < best_val_mae:
                best_val_mae = val_metrics["MAE"]
                self.best_model_name = name
                self.best_model = model

        # Save winning best model
        if self.best_model:
            self.best_model.save(BEST_MODEL_PATH)

        comparison_df = pd.DataFrame(results).sort_values("Val MAE (kg)").reset_index(drop=True)
        return comparison_df

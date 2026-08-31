"""
Baseline Models for F1 Race Outcome Prediction.
Includes Historical Average baseline and Ridge Regression with standard scaling.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.base import BaseF1Model


class HistoricalAverageBaseline(BaseF1Model):
    """Simple baseline using exponentially weighted moving average finish and grid."""

    def __init__(self, name: str = "HistoricalAverageBaseline"):
        super().__init__(name=name)

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "HistoricalAverageBaseline":
        self.feature_columns = list(X.columns)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Heuristic blend of recent form and grid position."""
        form = X["driver_form_ewm_finish"].values if "driver_form_ewm_finish" in X else 10.0
        grid = X["grid_position"].values if "grid_position" in X else 10.0
        c_finish = X["constructor_recent_avg_finish_5"].values if "constructor_recent_avg_finish_5" in X else 10.0
        pred = 0.40 * grid + 0.35 * form + 0.25 * c_finish
        return np.clip(pred, 1.0, 22.0)


class RidgeRegressionModel(BaseF1Model):
    """Regularized linear regression baseline."""

    def __init__(self, alpha: float = 1.0, name: str = "RidgeRegression"):
        super().__init__(name=name)
        self.alpha = alpha
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=self.alpha, random_state=42)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "RidgeRegressionModel":
        self.feature_columns = list(X.columns)
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.pipeline.predict(X[self.feature_columns])
        return np.clip(preds, 1.0, 22.0)

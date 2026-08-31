"""
DNF (Did Not Finish) Probability Classifier.
Predicts calibrated probability of mechanical or incident race retirement.
"""

from typing import Optional, List, Dict
import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

from src.models.base import BaseF1Model


class DNFProbabilityModel(BaseF1Model):
    """Calibrated binary classifier for race retirement probability P(DNF)."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.04,
        max_depth: int = 4,
        random_state: int = 42,
        name: str = "DNFModel",
    ):
        super().__init__(name=name)
        self.base_classifier = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            scale_pos_weight=3.0,
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        )
        self.calibrated_model: Optional[CalibratedClassifierCV] = None

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "DNFProbabilityModel":
        """
        Fit calibrated DNF classifier.
        `y` is binary indicator (1 = DNF, 0 = Finished).
        """
        self.feature_columns = list(X.columns)

        # Calibrate probabilities with sigmoid (Platt scaling) cross-validation
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_classifier,
            method="sigmoid",
            cv=3,
        )
        self.calibrated_model.fit(X[self.feature_columns], y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated P(DNF) probabilities for each driver."""
        if not self.is_fitted or self.calibrated_model is None:
            raise ValueError("Model is not fitted.")
        probs = self.calibrated_model.predict_proba(X[self.feature_columns])[:, 1]
        # In modern F1, DNF rates typically range between 4% and 25%
        return np.clip(probs, 0.02, 0.45)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary prediction using default 0.50 threshold."""
        return (self.predict_proba(X) >= 0.50).astype(int)

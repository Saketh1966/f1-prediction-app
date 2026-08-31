"""
Probability Calibration and Reliability Analysis.
Provides Platt scaling, Isotonic regression, and reliability curve generation.
"""

from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """Calibrates raw model probabilities using Platt Scaling or Isotonic Regression."""

    def __init__(self, method: str = "sigmoid"):
        self.method = method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, y_raw_probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """Fit probability calibrator on validation predictions."""
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(y_raw_probs, y_true)
        else:  # Platt scaling (logistic sigmoid)
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_raw_probs.reshape(-1, 1), y_true)

        self.is_fitted = True
        return self

    def predict(self, y_raw_probs: np.ndarray) -> np.ndarray:
        """Transform raw probabilities into calibrated probabilities."""
        if not self.is_fitted:
            return y_raw_probs
        if self.method == "isotonic":
            return np.clip(self.calibrator.predict(y_raw_probs), 0.0, 1.0)
        else:
            return self.calibrator.predict_proba(y_raw_probs.reshape(-1, 1))[:, 1]


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, List[float]]:
    """Compute empirical calibration curve points for visualization."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return {
        "prob_true": [float(x) for x in prob_true],
        "prob_pred": [float(x) for x in prob_pred],
    }

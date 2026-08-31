"""
Base Model Abstract Classes and Interfaces for F1 Prediction.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import os
import joblib
import pandas as pd
import numpy as np


class BaseF1Model(ABC):
    """Abstract base class for all F1 race outcome prediction models."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.feature_columns: list = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "BaseF1Model":
        """Fit model on training feature matrix and target."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict finishing positions or performance scores."""
        pass

    def save(self, filepath: str) -> None:
        """Serialize model artifact to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BaseF1Model":
        """Deserialize model artifact from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model artifact '{filepath}' not found.")
        return joblib.load(filepath)

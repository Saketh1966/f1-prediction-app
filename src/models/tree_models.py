"""
Tree-Based Ensemble Models: Random Forest, LightGBM, and XGBoost.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

from src.models.base import BaseF1Model


class RandomForestModel(BaseF1Model):
    """Random Forest Regressor for finishing position prediction."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 10, random_state: int = 42, name: str = "RandomForest"):
        super().__init__(name=name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "RandomForestModel":
        self.feature_columns = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_columns])
        return np.clip(preds, 1.0, 22.0)

    def get_feature_importances(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)


class LightGBMModel(BaseF1Model):
    """LightGBM Gradient Boosted Decision Trees Regressor."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_state: int = 42,
        name: str = "LightGBM",
    ):
        super().__init__(name=name)
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=15,
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "LightGBMModel":
        self.feature_columns = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_columns])
        return np.clip(preds, 1.0, 22.0)

    def get_feature_importances(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)


class XGBoostModel(BaseF1Model):
    """XGBoost Gradient Boosted Trees Regressor."""

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.03,
        max_depth: int = 5,
        random_state: int = 42,
        name: str = "XGBoost",
    ):
        super().__init__(name=name)
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "XGBoostModel":
        self.feature_columns = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X[self.feature_columns])
        return np.clip(preds, 1.0, 22.0)

    def get_feature_importances(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        return pd.DataFrame({
            "feature": self.feature_columns,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

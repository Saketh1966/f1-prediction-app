"""
Race Ranking Model (Listwise/Pairwise LambdaRank Objective).
Models Formula 1 finishing order as an intra-event ranking problem.
"""

from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import lightgbm as lgb

from src.models.base import BaseF1Model


class F1RankingModel(BaseF1Model):
    """
    LightGBM LambdaRank model optimizing NDCG across race groups.
    Target relevance: 23 - positionOrder (1st place gets 22, 22nd place gets 1).
    """

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        random_state: int = 42,
        name: str = "LambdaRank",
    ):
        super().__init__(name=name)
        self.model = lgb.LGBMRanker(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[1, 3, 5, 10],
            verbosity=-1,
            n_jobs=-1,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "F1RankingModel":
        """
        Fit LambdaRank model.
        `y` is positionOrder (1..22). Converted to relevance score (23 - y).
        `groups` is the raceId series.
        """
        self.feature_columns = list(X.columns)

        if groups is None:
            raise ValueError("F1RankingModel requires `groups` (raceId) for grouping query sessions.")

        # Group counts must match order of rows sorted by raceId
        df_fit = X.copy()
        df_fit["_y"] = y.values
        df_fit["_raceId"] = groups.values
        df_fit = df_fit.sort_values("_raceId").reset_index(drop=True)

        group_counts = df_fit.groupby("_raceId", sort=False).size().values
        relevance = np.maximum(1, np.round(23 - df_fit["_y"].values)).astype(int)
        X_sorted = df_fit[self.feature_columns]

        self.model.fit(X_sorted, relevance, group=group_counts)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict latent relevance scores, then invert to finishing position estimates.
        Higher relevance => lower position number (better finish).
        """
        scores = self.model.predict(X[self.feature_columns])
        # Map relevance score to 1..22 scale
        # For a single race, sorting descending gives rank
        min_s, max_s = scores.min(), scores.max()
        if max_s > min_s:
            norm_scores = (scores - min_s) / (max_s - min_s + 1e-8)
            # 1.0 (best) -> 1st position, 0.0 (worst) -> 20th position
            return np.clip(1.0 + (1.0 - norm_scores) * 19.0, 1.0, 22.0)
        return np.full_like(scores, 10.0)

    def predict_latent_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw latent ranking scores for Monte Carlo simulation."""
        return self.model.predict(X[self.feature_columns])

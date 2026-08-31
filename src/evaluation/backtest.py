import sys
import os
import json
import logging
from typing import Dict, List, Any

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.models.baseline import HistoricalAverageBaseline, RidgeRegressionModel
from src.models.tree_models import RandomForestModel, LightGBMModel, XGBoostModel
from src.models.ranking_model import F1RankingModel
from src.models.neural_net import PyTorchEntityEmbeddingModel
from src.models.dnf_model import DNFProbabilityModel
from src.evaluation.metrics import (
    evaluate_finishing_position,
    evaluate_ranking_accuracy,
    evaluate_probability_calibration,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """Orchestrates temporal expanding-window validation over historical seasons."""

    def __init__(
        self,
        test_seasons: List[int] = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        processed_data_dir: str = "data/processed",
        output_dir: str = "models",
    ):
        self.test_seasons = test_seasons
        self.processed_data_dir = processed_data_dir
        self.output_dir = output_dir

    def run_backtest(self) -> Dict[str, Any]:
        """Execute full walk-forward validation across all models and test seasons."""
        os.makedirs(self.output_dir, exist_ok=True)
        train_path = os.path.join(self.processed_data_dir, "train_features.csv")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Feature dataset '{train_path}' not found. Run feature build first.")

        df = pd.read_csv(train_path)

        exclude_cols = [
            "raceId", "driverId", "constructorId", "circuitId", "year", "round",
            "positionOrder", "points", "statusId", "is_dnf", "is_win", "is_podium",
            "is_top5", "is_top10", "actual_grid", "driverRef", "code", "forename",
            "surname", "nationality", "constructorRef", "constructor_name"
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        model_factories = {
            "Historical Average": lambda: HistoricalAverageBaseline(),
            "Ridge Regression": lambda: RidgeRegressionModel(alpha=5.0),
            "Random Forest": lambda: RandomForestModel(n_estimators=150, max_depth=8),
            "LightGBM Regressor": lambda: LightGBMModel(n_estimators=200, learning_rate=0.03, max_depth=5),
            "XGBoost Regressor": lambda: XGBoostModel(n_estimators=180, learning_rate=0.03, max_depth=4),
            "LambdaRank Ranking": lambda: F1RankingModel(n_estimators=180, learning_rate=0.03),
            "PyTorch Embedding NN": lambda: PyTorchEntityEmbeddingModel(epochs=25, lr=0.002),
        }

        season_results = []
        all_oof_predictions = []

        logger.info(f"Starting walk-forward validation for seasons: {self.test_seasons}")

        for season in self.test_seasons:
            train_mask = df["year"] < season
            test_mask = df["year"] == season

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                logger.warning(f"Skipping season {season}: Insufficient train ({train_mask.sum()}) or test ({test_mask.sum()}) samples.")
                continue

            X_train = df.loc[train_mask, feature_cols]
            y_train = df.loc[train_mask, "positionOrder"]
            groups_train = df.loc[train_mask, "raceId"]

            X_test = df.loc[test_mask, feature_cols]
            y_test = df.loc[test_mask, "positionOrder"]
            test_subset = df.loc[test_mask, ["raceId", "year", "round", "driverId", "positionOrder", "is_dnf"]].copy()

            # Train DNF model for this fold
            dnf_clf = DNFProbabilityModel(n_estimators=100)
            dnf_clf.fit(X_train, df.loc[train_mask, "is_dnf"])
            dnf_pred_proba = dnf_clf.predict_proba(X_test)
            dnf_metrics = evaluate_probability_calibration(df.loc[test_mask, "is_dnf"].values, dnf_pred_proba)

            for model_name, factory in model_factories.items():
                model = factory()
                try:
                    if model_name in ["PyTorch Embedding NN"]:
                        # NN needs driverId, constructorId, circuitId
                        X_train_nn = df.loc[train_mask, feature_cols + ["driverId", "constructorId", "circuitId"]]
                        X_test_nn = df.loc[test_mask, feature_cols + ["driverId", "constructorId", "circuitId"]]
                        model.fit(X_train_nn, y_train)
                        preds = model.predict(X_test_nn)
                    elif model_name == "LambdaRank Ranking":
                        model.fit(X_train, y_train, groups=groups_train)
                        preds = model.predict(X_test)
                    else:
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                    # Position metrics
                    pos_metrics = evaluate_finishing_position(y_test.values, preds)

                    # Ranking accuracy within races
                    test_eval_df = test_subset.copy()
                    test_eval_df["pred_pos"] = preds
                    rank_metrics = evaluate_ranking_accuracy(test_eval_df)

                    season_results.append({
                        "season": season,
                        "model": model_name,
                        "mae": pos_metrics["mae"],
                        "rmse": pos_metrics["rmse"],
                        "spearman_rho": pos_metrics["spearman_rho"],
                        "kendall_tau": pos_metrics["kendall_tau"],
                        "top1_accuracy": rank_metrics["top1_accuracy"],
                        "top3_accuracy": rank_metrics["top3_accuracy"],
                        "race_kendall_tau": rank_metrics["mean_race_kendall_tau"],
                        "dnf_brier_score": dnf_metrics["brier_score"],
                        "dnf_roc_auc": dnf_metrics["roc_auc"],
                    })

                    test_eval_df["model"] = model_name
                    all_oof_predictions.append(test_eval_df)

                except Exception as e:
                    logger.error(f"Error training {model_name} on season {season}: {e}")

        res_df = pd.DataFrame(season_results)
        summary_table = res_df.groupby("model").agg({
            "mae": "mean",
            "rmse": "mean",
            "spearman_rho": "mean",
            "kendall_tau": "mean",
            "top1_accuracy": "mean",
            "top3_accuracy": "mean",
            "race_kendall_tau": "mean",
        }).reset_index().sort_values("mae")

        # Save results
        res_df.to_csv(os.path.join(self.output_dir, "walk_forward_season_results.csv"), index=False)
        summary_table.to_csv(os.path.join(self.output_dir, "benchmark_summary.csv"), index=False)

        summary_dict = {
            "models_evaluated": list(model_factories.keys()),
            "test_seasons": self.test_seasons,
            "overall_summary": summary_table.to_dict(orient="records"),
            "best_model_by_mae": summary_table.iloc[0]["model"],
        }
        with open(os.path.join(self.output_dir, "benchmark_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        logger.info("\n=== Walk-Forward Cross-Validation Summary ===")
        logger.info("\n" + summary_table.to_string(index=False))

        return summary_dict


if __name__ == "__main__":
    backtester = WalkForwardBacktester()
    backtester.run_backtest()

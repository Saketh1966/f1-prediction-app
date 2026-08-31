import sys
import os
import json
import logging
import argparse

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import joblib

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.models.baseline import HistoricalAverageBaseline, RidgeRegressionModel
from src.models.tree_models import RandomForestModel, LightGBMModel, XGBoostModel
from src.models.ranking_model import F1RankingModel
from src.models.neural_net import PyTorchEntityEmbeddingModel
from src.models.dnf_model import DNFProbabilityModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_production_models(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    target_race_id: int = 1181,
) -> None:
    """Train and persist all final models."""
    os.makedirs(models_dir, exist_ok=True)
    train_path = os.path.join(processed_dir, "train_features.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset '{train_path}' not found.")

    df = pd.read_csv(train_path)

    exclude_cols = [
        "raceId", "driverId", "constructorId", "circuitId", "year", "round",
        "positionOrder", "points", "statusId", "is_dnf", "is_win", "is_podium",
        "is_top5", "is_top10", "actual_grid", "driverRef", "code", "forename",
        "surname", "nationality", "constructorRef", "constructor_name"
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols]
    y_pos = df["positionOrder"]
    y_dnf = df["is_dnf"]
    groups = df["raceId"]

    logger.info(f"Training final production models on {len(df)} samples across {df['raceId'].nunique()} Grand Prix events...")

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("f1_monza_race_prediction")
            mlflow.start_run(run_name="production_training_monza_2026")
            mlflow.log_params({
                "n_samples": len(df),
                "n_features": len(feature_cols),
                "target_race_id": target_race_id,
            })
        except Exception as e:
            logger.warning(f"MLflow initialization warning: {e}")

    # 1. Baseline
    logger.info("Training Historical Baseline & Ridge Regression...")
    baseline = HistoricalAverageBaseline().fit(X, y_pos)
    baseline.save(os.path.join(models_dir, "baseline_model.joblib"))

    ridge = RidgeRegressionModel(alpha=5.0).fit(X, y_pos)
    ridge.save(os.path.join(models_dir, "ridge_model.joblib"))

    # 2. Random Forest
    logger.info("Training Random Forest...")
    rf = RandomForestModel(n_estimators=200, max_depth=9).fit(X, y_pos)
    rf.save(os.path.join(models_dir, "random_forest_model.joblib"))

    # 3. LightGBM Regressor (Primary Regression Model)
    logger.info("Training LightGBM Regressor...")
    lgb_model = LightGBMModel(n_estimators=250, learning_rate=0.03, max_depth=6).fit(X, y_pos)
    lgb_model.save(os.path.join(models_dir, "lightgbm_model.joblib"))

    # 4. XGBoost Regressor
    logger.info("Training XGBoost Regressor...")
    xgb_model = XGBoostModel(n_estimators=220, learning_rate=0.03, max_depth=5).fit(X, y_pos)
    xgb_model.save(os.path.join(models_dir, "xgboost_model.joblib"))

    # 5. LambdaRank Model (Intra-Race Ranking)
    logger.info("Training LambdaRank Model...")
    rank_model = F1RankingModel(n_estimators=220, learning_rate=0.03).fit(X, y_pos, groups=groups)
    rank_model.save(os.path.join(models_dir, "lambdarank_model.joblib"))

    # 6. PyTorch Entity Embedding NN
    logger.info("Training PyTorch Entity Embedding Neural Network...")
    X_nn = df[feature_cols + ["driverId", "constructorId", "circuitId"]]
    nn_model = PyTorchEntityEmbeddingModel(epochs=35, lr=0.0015).fit(X_nn, y_pos)
    nn_model.save(os.path.join(models_dir, "neural_net_model.joblib"))

    # 7. Calibrated DNF Model
    logger.info("Training Calibrated DNF Probability Model...")
    dnf_model = DNFProbabilityModel(n_estimators=150, learning_rate=0.04).fit(X, y_dnf)
    dnf_model.save(os.path.join(models_dir, "dnf_model.joblib"))

    # Save feature metadata
    with open(os.path.join(models_dir, "feature_columns.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    if MLFLOW_AVAILABLE:
        try:
            mlflow.end_run()
        except Exception:
            pass

    logger.info("[SUCCESS] All production models successfully trained and serialized to 'models/'.")


if __name__ == "__main__":
    train_production_models()

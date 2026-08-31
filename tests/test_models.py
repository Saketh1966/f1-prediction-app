"""
Unit Tests for Machine Learning Models (Baseline, RF, LightGBM, XGBoost, Ranker, PyTorch NN, DNF).
"""

import pytest
import pandas as pd
import numpy as np

from src.models.baseline import HistoricalAverageBaseline, RidgeRegressionModel
from src.models.tree_models import RandomForestModel, LightGBMModel, XGBoostModel
from src.models.ranking_model import F1RankingModel
from src.models.neural_net import PyTorchEntityEmbeddingModel
from src.models.dnf_model import DNFProbabilityModel


@pytest.fixture
def dummy_train_data():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "raceId": np.repeat(np.arange(1, 11), 20),
        "driverId": np.tile(np.arange(1, 21), 10),
        "constructorId": np.tile(np.repeat(np.arange(1, 11), 2), 10),
        "circuitId": np.full(n, 14),
        "driver_form_ewm_finish": np.random.uniform(1, 20, n),
        "driver_rolling_finish_last3": np.random.uniform(1, 20, n),
        "grid_position": np.random.uniform(1, 20, n),
        "constructor_recent_avg_finish_5": np.random.uniform(1, 20, n),
        "driver_career_podium_rate": np.random.uniform(0, 0.8, n),
        "driver_circuit_avg_finish": np.random.uniform(1, 20, n),
    })
    y = np.clip(0.6 * df["grid_position"] + 0.4 * df["driver_form_ewm_finish"] + np.random.normal(0, 1, n), 1, 20)
    is_dnf = (np.random.uniform(0, 1, n) < 0.15).astype(int)
    return df, pd.Series(y), pd.Series(is_dnf)


def test_baseline_and_ridge(dummy_train_data):
    X, y, _ = dummy_train_data
    base = HistoricalAverageBaseline().fit(X, y)
    preds = base.predict(X)
    assert len(preds) == len(X)
    assert (preds >= 1.0).all() and (preds <= 22.0).all()

    ridge = RidgeRegressionModel().fit(X, y)
    r_preds = ridge.predict(X)
    assert len(r_preds) == len(X)


def test_tree_models(dummy_train_data):
    X, y, _ = dummy_train_data
    rf = RandomForestModel(n_estimators=10).fit(X, y)
    assert len(rf.predict(X)) == len(X)

    lgb = LightGBMModel(n_estimators=10).fit(X, y)
    assert len(lgb.predict(X)) == len(X)

    xgb_m = XGBoostModel(n_estimators=10).fit(X, y)
    assert len(xgb_m.predict(X)) == len(X)


def test_lambdarank_model(dummy_train_data):
    X, y, _ = dummy_train_data
    ranker = F1RankingModel(n_estimators=10).fit(X, y, groups=X["raceId"])
    preds = ranker.predict(X)
    assert len(preds) == len(X)
    assert (preds >= 1.0).all() and (preds <= 22.0).all()


def test_pytorch_embedding_nn(dummy_train_data):
    X, y, _ = dummy_train_data
    nn_model = PyTorchEntityEmbeddingModel(epochs=2, batch_size=32).fit(X, y)
    preds = nn_model.predict(X)
    assert len(preds) == len(X)
    assert (preds >= 1.0).all() and (preds <= 22.0).all()


def test_dnf_model(dummy_train_data):
    X, _, is_dnf = dummy_train_data
    dnf_m = DNFProbabilityModel(n_estimators=10).fit(X, is_dnf)
    probs = dnf_m.predict_proba(X)
    assert len(probs) == len(X)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

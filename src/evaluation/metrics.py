"""
Comprehensive F1 Race Evaluation Metrics Suite.
Calculates regression, ranking, classification, and probabilistic calibration metrics.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_finishing_position(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression and rank-order correlation metrics for finishing position."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Rank correlations
    s_corr, _ = spearmanr(y_true, y_pred)
    k_tau, _ = kendalltau(y_true, y_pred)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "spearman_rho": float(s_corr) if not np.isnan(s_corr) else 0.0,
        "kendall_tau": float(k_tau) if not np.isnan(k_tau) else 0.0,
    }


def evaluate_ranking_accuracy(
    df_eval: pd.DataFrame,
    group_col: str = "raceId",
    pred_col: str = "pred_pos",
    true_col: str = "positionOrder",
) -> Dict[str, float]:
    """Calculate intra-race top-k classification accuracy and NDCG."""
    top1_hits, top3_hits, total_races = 0, 0, 0
    kendall_taus = []

    for _, race_group in df_eval.groupby(group_col):
        total_races += 1
        sorted_pred = race_group.sort_values(pred_col)
        sorted_true = race_group.sort_values(true_col)

        pred_winner = sorted_pred.iloc[0]["driverId"]
        true_winner = sorted_true.iloc[0]["driverId"]
        if pred_winner == true_winner:
            top1_hits += 1

        pred_podium = set(sorted_pred.iloc[:3]["driverId"])
        true_podium = set(sorted_true.iloc[:3]["driverId"])
        top3_hits += len(pred_podium.intersection(true_podium)) / 3.0

        kt, _ = kendalltau(race_group[true_col], race_group[pred_col])
        if not np.isnan(kt):
            kendall_taus.append(kt)

    return {
        "top1_accuracy": float(top1_hits / total_races) if total_races > 0 else 0.0,
        "top3_accuracy": float(top3_hits / total_races) if total_races > 0 else 0.0,
        "mean_race_kendall_tau": float(np.mean(kendall_taus)) if kendall_taus else 0.0,
    }


def evaluate_probability_calibration(
    y_true_binary: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Calculate probabilistic evaluation metrics and Expected Calibration Error (ECE)."""
    brier = brier_score_loss(y_true_binary, y_pred_proba)

    # Avoid log loss inf/nan on edge 0/1
    eps = 1e-15
    clipped_probs = np.clip(y_pred_proba, eps, 1.0 - eps)
    ll = log_loss(y_true_binary, clipped_probs)

    # ROC AUC if both classes exist
    if len(np.unique(y_true_binary)) > 1:
        roc_auc = roc_auc_score(y_true_binary, clipped_probs)
    else:
        roc_auc = 0.50

    # Expected Calibration Error (ECE)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (clipped_probs >= bins[i]) & (clipped_probs < bins[i + 1])
        if np.sum(bin_mask) > 0:
            bin_acc = np.mean(y_true_binary[bin_mask])
            bin_conf = np.mean(clipped_probs[bin_mask])
            bin_weight = np.sum(bin_mask) / len(y_true_binary)
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return {
        "brier_score": float(brier),
        "log_loss": float(ll),
        "roc_auc": float(roc_auc),
        "expected_calibration_error": float(ece),
    }

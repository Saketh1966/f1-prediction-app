"""
Explainability Engine using SHAP (SHapley Additive exPlanations).
Generates global feature attributions and localized driver factor breakdowns.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import shap

from src.models.tree_models import LightGBMModel


class F1Explainer:
    """Computes SHAP explanations for tree-based F1 prediction models."""

    def __init__(self, model: LightGBMModel, background_data: pd.DataFrame):
        self.model = model
        self.feature_columns = model.feature_columns
        self.explainer = shap.TreeExplainer(model.model)
        exp_val = self.explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            self.expected_value = float(np.ravel(exp_val)[0])
        else:
            self.expected_value = float(exp_val)

    def explain_instance(self, driver_row: pd.Series) -> Dict[str, Any]:
        """
        Explain predictions for an individual driver in the upcoming race.
        Returns top positive (factors that improve finish) and negative factors (factors that worsen finish).
        Note: For positionOrder, negative SHAP delta is beneficial (lower position number = better finish).
        """
        X_inst = pd.DataFrame([driver_row[self.feature_columns]])
        shap_vals = self.explainer.shap_values(X_inst)[0]

        feature_impacts = []
        for feat, val, s_val in zip(self.feature_columns, X_inst.iloc[0], shap_vals):
            feature_impacts.append({
                "feature": feat,
                "value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                "shap_value": float(s_val),
            })

        # Sort by magnitude of contribution
        # Negative shap_value means it pushes predicted position LOWER (better finish, positive factor for driver)
        sorted_by_benefit = sorted(feature_impacts, key=lambda x: x["shap_value"])

        # Human-readable labels
        feature_labels = {
            "driver_form_ewm_finish": "Recent Driver Race Form",
            "driver_rolling_finish_last3": "Last 3 Races Finishing Avg",
            "driver_rolling_finish_last5": "Last 5 Races Finishing Avg",
            "driver_season_avg_finish": "Current Season Average Finish",
            "driver_championship_stand_pos": "Championship Standings Rank",
            "driver_championship_points": "Championship Points Total",
            "driver_circuit_recency_weighted_finish": "Monza Track Record (Time-Decayed)",
            "driver_circuit_podiums": "Monza Career Podiums",
            "driver_circuit_avg_finish": "Monza Career Average Finish",
            "driver_quali_rolling_avg_last5": "Recent Qualifying Pace",
            "driver_quali_vs_teammate_diff": "Teammate Qualifying Advantage",
            "grid_position": "Starting Grid Position",
            "constructor_recent_avg_finish_5": "Constructor Recent Race Pace",
            "constructor_rolling_points_last5": "Constructor Points Momentum",
            "constructor_season_rank": "Constructor Championship Rank",
            "constructor_circuit_podium_rate": "Constructor Monza Track Record",
            "constructor_reliability_dnf_rate": "Constructor Reliability Track Record",
            "driver_career_dnf_rate": "Driver Career DNF Hazard",
            "driver_recent_dnf_rate_5": "Recent Mechanical Retirement Rate",
            "constructor_pit_duration_mean": "Constructor Pit Stop Efficiency",
            "driver_historical_pace_median": "Historical Race Pace Relative to Field",
        }

        top_positive = []
        for item in sorted_by_benefit:
            if item["shap_value"] < -0.10 and len(top_positive) < 4:
                lbl = feature_labels.get(item["feature"], item["feature"])
                top_positive.append({
                    "factor": lbl,
                    "impact": f"{-item['shap_value']:.2f} positions better",
                    "feature_name": item["feature"],
                    "shap": item["shap_value"],
                })

        top_negative = []
        for item in reversed(sorted_by_benefit):
            if item["shap_value"] > 0.10 and len(top_negative) < 4:
                lbl = feature_labels.get(item["feature"], item["feature"])
                top_negative.append({
                    "factor": lbl,
                    "impact": f"+{item['shap_value']:.2f} positions worse",
                    "feature_name": item["feature"],
                    "shap": item["shap_value"],
                })

        return {
            "driver_id": int(driver_row.get("driverId", 0)),
            "driver_name": str(driver_row.get("driver_name", driver_row.get("surname", "Driver"))),
            "base_value": self.expected_value,
            "predicted_finish": float(self.expected_value + np.sum(shap_vals)),
            "top_positive_factors": top_positive,
            "top_negative_factors": top_negative,
            "all_shap_values": {item["feature"]: item["shap_value"] for item in feature_impacts},
        }

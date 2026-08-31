"""
Monza 2026 Race Predictor.
High-level inference and simulation pipeline for the 2026 Formula 1 Italian Grand Prix at Monza.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import joblib

from src.models.tree_models import LightGBMModel
from src.models.ranking_model import F1RankingModel
from src.models.dnf_model import DNFProbabilityModel
from src.simulation.monte_carlo import MonteCarloRaceSimulator
from src.prediction.explainability import F1Explainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MonzaPredictor:
    """End-to-end predictor and race simulator for the 2026 Italian GP at Monza."""

    def __init__(
        self,
        models_dir: str = "models",
        processed_dir: str = "data/processed",
        artifacts_dir: str = "artifacts",
    ):
        self.models_dir = models_dir
        self.processed_dir = processed_dir
        self.artifacts_dir = artifacts_dir
        os.makedirs(artifacts_dir, exist_ok=True)

        self.model: Optional[LightGBMModel] = None
        self.rank_model: Optional[F1RankingModel] = None
        self.dnf_model: Optional[DNFProbabilityModel] = None
        self.explainer: Optional[F1Explainer] = None
        self.feature_columns: List[str] = []
        self.monza_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained models and target Monza feature set."""
        monza_path = os.path.join(self.processed_dir, "monza_2026_target_features.csv")
        train_path = os.path.join(self.processed_dir, "train_features.csv")
        feat_path = os.path.join(self.models_dir, "feature_columns.json")

        if os.path.exists(monza_path):
            self.monza_df = pd.read_csv(monza_path)
        if os.path.exists(train_path):
            self.train_df = pd.read_csv(train_path)
        if os.path.exists(feat_path):
            with open(feat_path, "r", encoding="utf-8") as f:
                self.feature_columns = json.load(f)

        lgb_path = os.path.join(self.models_dir, "lightgbm_model.joblib")
        rank_path = os.path.join(self.models_dir, "lambdarank_model.joblib")
        dnf_path = os.path.join(self.models_dir, "dnf_model.joblib")

        if os.path.exists(lgb_path):
            self.model = LightGBMModel.load(lgb_path)
        if os.path.exists(rank_path):
            self.rank_model = F1RankingModel.load(rank_path)
        if os.path.exists(dnf_path):
            self.dnf_model = DNFProbabilityModel.load(dnf_path)

        if self.model and self.train_df is not None:
            self.explainer = F1Explainer(self.model, self.train_df[self.feature_columns].sample(min(200, len(self.train_df))))

    def predict_monza(
        self,
        n_simulations: int = 10000,
        is_post_qualifying: bool = False,
        custom_grid: Optional[Dict[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete prediction and Monte Carlo simulation for 2026 Italian GP.
        """
        if self.monza_df is None or self.model is None or self.dnf_model is None:
            self._load_artifacts()
            if self.monza_df is None:
                raise ValueError("Monza target data or models not loaded. Ensure models and features are built.")

        df_target = self.monza_df.copy()

        # Update grid position if post-qualifying / custom grid provided
        if custom_grid:
            df_target["grid_position"] = df_target["driverId"].map(custom_grid).fillna(df_target["grid_position"])
            df_target["is_post_qualifying_feature"] = 1.0

        X_target = df_target[self.feature_columns]

        # Model predictions
        predicted_finishes = self.model.predict(X_target)
        predicted_dnf_prob = self.dnf_model.predict_proba(X_target)

        # Monte Carlo Simulation
        simulator = MonteCarloRaceSimulator(n_simulations=n_simulations, grid_importance=0.55)
        sim_results = simulator.run_simulation(
            driver_df=df_target,
            predicted_positions=predicted_finishes,
            dnf_probabilities=predicted_dnf_prob,
            custom_grid=custom_grid,
        )

        summary_df = sim_results["summary_table"]

        # Formulate final report dictionary
        top_winner = summary_df.iloc[0]
        podium_drivers = summary_df.iloc[:3]

        prediction_payload = {
            "grand_prix": "2026 Formula 1 Italian Grand Prix",
            "circuit": "Autodromo Nazionale di Monza",
            "round": 13,
            "season": 2026,
            "mode": "Post-Qualifying Grid Scenario" if (is_post_qualifying or custom_grid) else "Pre-Qualifying Latent Forecast",
            "n_simulations": n_simulations,
            "predicted_winner": {
                "driver_name": top_winner["driver_name"],
                "driver_code": top_winner["driver_code"],
                "constructor": top_winner["constructor"],
                "win_probability": f"{top_winner['win_probability'] * 100:.1f}%",
                "expected_finish": round(top_winner["expected_finish"], 2),
            },
            "predicted_podium": [
                {
                    "position": i + 1,
                    "driver_name": row["driver_name"],
                    "driver_code": row["driver_code"],
                    "constructor": row["constructor"],
                    "podium_probability": f"{row['podium_probability'] * 100:.1f}%",
                    "expected_finish": round(row["expected_finish"], 2),
                }
                for i, (_, row) in enumerate(podium_drivers.iterrows())
            ],
            "full_grid_predictions": summary_df.to_dict(orient="records"),
        }

        # Persist predictions artifact
        out_json = os.path.join(self.artifacts_dir, "monza_2026_prediction.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(prediction_payload, f, indent=2)

        return prediction_payload

    def get_driver_explanation(self, driver_id: int) -> Dict[str, Any]:
        """Generate SHAP explanation for a specific driver."""
        if self.explainer is None:
            self._load_artifacts()
            if self.explainer is None:
                raise ValueError("Explainer not ready.")

        d_rows = self.monza_df[self.monza_df["driverId"] == driver_id]
        if len(d_rows) == 0:
            raise ValueError(f"Driver ID {driver_id} not found in 2026 Monza grid.")

        return self.explainer.explain_instance(d_rows.iloc[0])


if __name__ == "__main__":
    predictor = MonzaPredictor()
    results = predictor.predict_monza(n_simulations=10000)
    print("\nPredicted Winner:", results["predicted_winner"])
    print("\nPredicted Podium:", results["predicted_podium"])

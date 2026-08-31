"""
Driver Temporal Feature Engineering.
Calculates historical rolling performance, championship momentum, and reliability metrics
strictly prior to the target race.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class DriverFeatureEngineer:
    """Computes driver-level historical and rolling features without temporal leakage."""

    def __init__(self, ewm_alpha: float = 0.35):
        self.ewm_alpha = ewm_alpha

    def compute_features_for_race(
        self,
        historical_results: pd.DataFrame,
        historical_standings: pd.DataFrame,
        target_driver_ids: List[int],
        target_season: int,
        target_round: int,
    ) -> pd.DataFrame:
        """
        Compute features for target drivers strictly using historical_results (where race < target_race).
        """
        records = []

        for driver_id in target_driver_ids:
            d_results = historical_results[historical_results["driverId"] == driver_id].sort_values(
                ["year", "round"]
            )

            n_starts = len(d_results)
            if n_starts == 0:
                # Rookie or driver without prior starts in historical window
                records.append({
                    "driverId": driver_id,
                    "driver_career_starts": 0,
                    "driver_career_avg_finish": 15.0,
                    "driver_form_ewm_finish": 15.0,
                    "driver_rolling_finish_last3": 15.0,
                    "driver_rolling_finish_last5": 15.0,
                    "driver_season_avg_finish": 15.0,
                    "driver_recent_points_sum5": 0.0,
                    "driver_career_podium_rate": 0.0,
                    "driver_career_win_rate": 0.0,
                    "driver_career_top10_rate": 0.10,
                    "driver_career_dnf_rate": 0.15,
                    "driver_recent_dnf_rate_5": 0.15,
                    "driver_avg_grid_pos_career": 15.0,
                    "driver_grid_to_finish_gain_mean": 0.0,
                    "driver_championship_stand_pos": 18.0,
                    "driver_championship_points": 0.0,
                })
                continue

            finishes = d_results["positionOrder"].values
            grids = d_results["grid"].values
            points = d_results["points"].values
            is_dnf = (d_results["statusId"].isin([1, 11, 12, 13, 14, 15, 16, 17, 18, 19]) == False).astype(float).values

            # Season subset
            season_results = d_results[d_results["year"] == target_season]
            season_avg_finish = season_results["positionOrder"].mean() if len(season_results) > 0 else finishes.mean()

            # Rolling metrics
            last3_finish = finishes[-3:].mean() if n_starts >= 3 else finishes.mean()
            last5_finish = finishes[-5:].mean() if n_starts >= 5 else finishes.mean()
            recent_points5 = points[-5:].sum() if n_starts >= 5 else points.sum()
            recent_dnf5 = is_dnf[-5:].mean() if n_starts >= 5 else is_dnf.mean()

            # EWM finish
            ewm_finish = pd.Series(finishes).ewm(alpha=self.ewm_alpha, adjust=False).mean().iloc[-1]

            # Career rates
            podium_rate = (finishes <= 3).mean()
            win_rate = (finishes == 1).mean()
            top10_rate = (finishes <= 10).mean()
            career_dnf_rate = is_dnf.mean()
            grid_gain = (grids - finishes).mean()

            # Standings entering race
            d_standings = historical_standings[historical_standings["driverId"] == driver_id]
            if len(d_standings) > 0:
                latest_stand = d_standings.iloc[-1]
                stand_pos = float(latest_stand.get("position", 15.0))
                stand_pts = float(latest_stand.get("points", 0.0))
            else:
                stand_pos = 18.0
                stand_pts = 0.0

            records.append({
                "driverId": driver_id,
                "driver_career_starts": n_starts,
                "driver_career_avg_finish": float(finishes.mean()),
                "driver_form_ewm_finish": float(ewm_finish),
                "driver_rolling_finish_last3": float(last3_finish),
                "driver_rolling_finish_last5": float(last5_finish),
                "driver_season_avg_finish": float(season_avg_finish),
                "driver_recent_points_sum5": float(recent_points5),
                "driver_career_podium_rate": float(podium_rate),
                "driver_career_win_rate": float(win_rate),
                "driver_career_top10_rate": float(top10_rate),
                "driver_career_dnf_rate": float(career_dnf_rate),
                "driver_recent_dnf_rate_5": float(recent_dnf5),
                "driver_avg_grid_pos_career": float(grids.mean()),
                "driver_grid_to_finish_gain_mean": float(grid_gain),
                "driver_championship_stand_pos": stand_pos,
                "driver_championship_points": stand_pts,
            })

        return pd.DataFrame(records)

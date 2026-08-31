"""
Constructor Temporal Feature Engineering.
Calculates team-level performance, championship standing, qualifying strength,
and mechanical reliability metrics strictly prior to the target race.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class ConstructorFeatureEngineer:
    """Computes constructor-level historical and rolling features without temporal leakage."""

    def __init__(self, ewm_alpha: float = 0.30):
        self.ewm_alpha = ewm_alpha

    def compute_features_for_race(
        self,
        historical_results: pd.DataFrame,
        historical_constructor_standings: pd.DataFrame,
        target_constructor_ids: List[int],
        target_season: int,
        target_round: int,
    ) -> pd.DataFrame:
        """
        Compute constructor features strictly using past results.
        """
        records = []

        for constructor_id in target_constructor_ids:
            c_results = historical_results[historical_results["constructorId"] == constructor_id].sort_values(
                ["year", "round"]
            )

            n_starts = len(c_results)
            if n_starts == 0:
                # New constructor entering the sport (e.g., Cadillac/Audi predecessor baseline)
                records.append({
                    "constructorId": constructor_id,
                    "constructor_career_starts": 0,
                    "constructor_career_avg_finish": 14.0,
                    "constructor_rolling_points_last5": 2.0,
                    "constructor_recent_avg_finish_5": 14.0,
                    "constructor_career_podium_rate": 0.02,
                    "constructor_career_win_rate": 0.0,
                    "constructor_reliability_dnf_rate": 0.15,
                    "constructor_recent_dnf_rate_10": 0.15,
                    "constructor_season_points": 0.0,
                    "constructor_season_rank": 10.0,
                })
                continue

            finishes = c_results["positionOrder"].values
            points = c_results["points"].values
            is_dnf = (c_results["statusId"].isin([1, 11, 12, 13, 14, 15, 16, 17, 18, 19]) == False).astype(float).values

            # Last 5 and last 10 entries (note each race usually has 2 entries per team)
            last10_finish = finishes[-10:].mean() if n_starts >= 10 else finishes.mean()
            recent_points5_races = points[-10:].sum() if n_starts >= 10 else points.sum()
            recent_dnf10 = is_dnf[-10:].mean() if n_starts >= 10 else is_dnf.mean()

            podium_rate = (finishes <= 3).mean()
            win_rate = (finishes == 1).mean()
            career_dnf = is_dnf.mean()

            # Standings entering race
            c_standings = historical_constructor_standings[
                historical_constructor_standings["constructorId"] == constructor_id
            ]
            if len(c_standings) > 0:
                latest_stand = c_standings.iloc[-1]
                stand_pos = float(latest_stand.get("position", 10.0))
                stand_pts = float(latest_stand.get("points", 0.0))
            else:
                stand_pos = 10.0
                stand_pts = 0.0

            records.append({
                "constructorId": constructor_id,
                "constructor_career_starts": n_starts,
                "constructor_career_avg_finish": float(finishes.mean()),
                "constructor_rolling_points_last5": float(recent_points5_races),
                "constructor_recent_avg_finish_5": float(last10_finish),
                "constructor_career_podium_rate": float(podium_rate),
                "constructor_career_win_rate": float(win_rate),
                "constructor_reliability_dnf_rate": float(career_dnf),
                "constructor_recent_dnf_rate_10": float(recent_dnf10),
                "constructor_season_points": stand_pts,
                "constructor_season_rank": stand_pos,
            })

        return pd.DataFrame(records)

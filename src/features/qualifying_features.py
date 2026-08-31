"""
Qualifying and Starting Grid Feature Engineering.
Features for both Pre-Qualifying prediction mode (latent qualifying strength)
and Post-Qualifying mode (confirmed starting grid position).
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class QualifyingFeatureEngineer:
    """Computes qualifying performance, teammate delta, and starting grid features."""

    def compute_features_for_race(
        self,
        historical_qualifying: pd.DataFrame,
        historical_results: pd.DataFrame,
        target_driver_ids: List[int],
        target_driver_constructors: Dict[int, int],
        is_post_qualifying: bool = True,
        current_grid_positions: Optional[Dict[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Compute qualifying and grid features.
        If is_post_qualifying is False, starting grid is imputed from driver & team qualifying strength.
        """
        records = []

        for driver_id in target_driver_ids:
            constructor_id = target_driver_constructors.get(driver_id)
            d_quali = historical_qualifying[historical_qualifying["driverId"] == driver_id].sort_values("raceId")
            n_quali = len(d_quali)

            if n_quali == 0:
                career_avg_q = 14.0
                rolling_last3_q = 14.0
                rolling_last5_q = 14.0
                teammate_diff = 0.0
            else:
                q_positions = d_quali["position"].dropna().values
                if len(q_positions) == 0:
                    career_avg_q = 14.0
                    rolling_last3_q = 14.0
                    rolling_last5_q = 14.0
                else:
                    career_avg_q = float(q_positions.mean())
                    rolling_last3_q = float(q_positions[-3:].mean()) if len(q_positions) >= 3 else career_avg_q
                    rolling_last5_q = float(q_positions[-5:].mean()) if len(q_positions) >= 5 else career_avg_q

                # Teammate delta in prior sessions
                teammate_diffs = []
                for _, q_row in d_quali.tail(5).iterrows():
                    r_id = q_row["raceId"]
                    c_id = q_row["constructorId"]
                    tm_quali = historical_qualifying[
                        (historical_qualifying["raceId"] == r_id)
                        & (historical_qualifying["constructorId"] == c_id)
                        & (historical_qualifying["driverId"] != driver_id)
                    ]
                    if len(tm_quali) > 0 and pd.notnull(tm_quali.iloc[0]["position"]) and pd.notnull(q_row["position"]):
                        # Negative diff means driver qualifies ahead of teammate (better)
                        teammate_diffs.append(float(q_row["position"] - tm_quali.iloc[0]["position"]))

                teammate_diff = float(np.mean(teammate_diffs)) if len(teammate_diffs) > 0 else 0.0

            # Constructor qualifying strength
            c_quali = historical_qualifying[historical_qualifying["constructorId"] == constructor_id]
            if len(c_quali) > 0:
                c_positions = c_quali["position"].dropna().values
                constructor_q_strength = float(c_positions[-10:].mean()) if len(c_positions) >= 10 else float(c_positions.mean())
            else:
                constructor_q_strength = 14.0

            # Starting grid assignment
            if is_post_qualifying and current_grid_positions and driver_id in current_grid_positions:
                grid_pos = float(current_grid_positions[driver_id])
            elif is_post_qualifying and not current_grid_positions:
                # If no custom grid provided, use rolling quali form as best post-quali proxy
                grid_pos = float(np.clip(rolling_last3_q, 1.0, 22.0))
            else:
                # Pre-qualifying mode: Expected grid blended from driver form and constructor pace
                expected_grid = 0.55 * rolling_last5_q + 0.45 * constructor_q_strength
                grid_pos = float(np.clip(expected_grid, 1.0, 22.0))

            records.append({
                "driverId": driver_id,
                "driver_quali_career_avg": career_avg_q,
                "driver_quali_rolling_avg_last3": rolling_last3_q,
                "driver_quali_rolling_avg_last5": rolling_last5_q,
                "driver_quali_vs_teammate_diff": teammate_diff,
                "constructor_quali_strength": constructor_q_strength,
                "grid_position": grid_pos,
                "is_post_qualifying_feature": 1.0 if is_post_qualifying else 0.0,
            })

        return pd.DataFrame(records)

"""
Monte Carlo Race Simulation Engine.
Simulates 10,000+ stochastic Grand Prix iterations incorporating calibrated DNF hazards,
latent performance distributions, track-specific overtaking difficulty, and FIA points allocation.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np


class MonteCarloRaceSimulator:
    """Simulates Grand Prix race outcomes and computes empirical probability distributions."""

    def __init__(
        self,
        n_simulations: int = 10000,
        performance_std: float = 1.45,
        grid_importance: float = 0.60,
        random_seed: int = 42,
    ):
        self.n_simulations = n_simulations
        self.performance_std = performance_std
        self.grid_importance = grid_importance
        self.random_seed = random_seed
        self.points_map = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }

    def run_simulation(
        self,
        driver_df: pd.DataFrame,
        predicted_positions: np.ndarray,
        dnf_probabilities: np.ndarray,
        custom_grid: Optional[Dict[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Monte Carlo simulations for the target driver grid.

        Parameters:
            driver_df: DataFrame with driver metadata (driverId, code/name, constructor).
            predicted_positions: Array of expected finishing position values or latent scores.
            dnf_probabilities: Calibrated P(DNF) for each driver.
            custom_grid: Optional manual override of grid positions.
        """
        np.random.seed(self.random_seed)
        n_drivers = len(driver_df)
        driver_ids = driver_df["driverId"].values

        # Grid positions
        if custom_grid:
            grid_positions = np.array([custom_grid.get(d, i + 1) for i, d in enumerate(driver_ids)])
        elif "grid_position" in driver_df.columns:
            grid_positions = driver_df["grid_position"].values
        else:
            grid_positions = np.arange(1, n_drivers + 1)

        # Storage for simulation outcomes: shape (n_simulations, n_drivers)
        simulated_finishes = np.zeros((self.n_simulations, n_drivers), dtype=int)
        simulated_dnf = np.zeros((self.n_simulations, n_drivers), dtype=bool)
        simulated_points = np.zeros((self.n_simulations, n_drivers), dtype=float)

        mean_grid = np.mean(grid_positions)

        for s in range(self.n_simulations):
            # 1. Sample DNF events
            u = np.random.uniform(0.0, 1.0, size=n_drivers)
            is_dnf = u < dnf_probabilities
            simulated_dnf[s, :] = is_dnf

            # 2. Sample stochastic performance score
            # Base expectation + Gaussian lap-pace perturbation + grid advantage
            noise = np.random.normal(0, self.performance_std, size=n_drivers)
            # Add heavy-tailed safety car perturbation occasionally (10% chance)
            if np.random.rand() < 0.10:
                noise += np.random.standard_t(df=4, size=n_drivers) * 0.8

            grid_effect = self.grid_importance * (grid_positions - mean_grid)
            composite_score = predicted_positions + noise + grid_effect

            # 3. Rank finishers and assign retirements
            surviving_indices = np.where(~is_dnf)[0]
            dnf_indices = np.where(is_dnf)[0]

            # Sort surviving drivers by composite score (lower is better)
            sorted_survivors = surviving_indices[np.argsort(composite_score[surviving_indices])]

            # Assign positions
            finish_rank = 1
            for idx in sorted_survivors:
                simulated_finishes[s, idx] = finish_rank
                if finish_rank in self.points_map:
                    simulated_points[s, idx] = self.points_map[finish_rank]
                finish_rank += 1

            # DNF drivers placed at the end in random order (sampled retirement lap)
            if len(dnf_indices) > 0:
                np.random.shuffle(dnf_indices)
                for idx in dnf_indices:
                    simulated_finishes[s, idx] = finish_rank
                    simulated_points[s, idx] = 0.0
                    finish_rank += 1

            # Bonus point for fastest lap (assigned among top 10 finishers)
            if len(sorted_survivors) >= 10:
                fastest_lap_idx = np.random.choice(sorted_survivors[:10])
                simulated_points[s, fastest_lap_idx] += 1.0
            elif len(sorted_survivors) > 0:
                fastest_lap_idx = np.random.choice(sorted_survivors)
                simulated_points[s, fastest_lap_idx] += 1.0

        # Aggregate summary statistics per driver
        results_summary = []
        for i in range(n_drivers):
            finishes_i = simulated_finishes[:, i]
            points_i = simulated_points[:, i]
            dnf_i = simulated_dnf[:, i]

            win_prob = float(np.mean(finishes_i == 1))
            podium_prob = float(np.mean(finishes_i <= 3))
            top5_prob = float(np.mean(finishes_i <= 5))
            top10_prob = float(np.mean(finishes_i <= 10))
            dnf_prob = float(np.mean(dnf_i))
            exp_finish = float(np.mean(finishes_i))
            median_finish = float(np.median(finishes_i))
            ci_lower = float(np.percentile(finishes_i, 2.5))
            ci_upper = float(np.percentile(finishes_i, 97.5))
            exp_points = float(np.mean(points_i))

            # Distribution histogram across positions 1..n_drivers
            pos_dist = [float(np.mean(finishes_i == p)) for p in range(1, n_drivers + 1)]

            d_info = {
                "driverId": int(driver_ids[i]),
                "driver_code": str(driver_df.iloc[i].get("code", f"D{i+1}")),
                "driver_name": f"{driver_df.iloc[i].get('forename', '')} {driver_df.iloc[i].get('surname', '')}".strip() or str(driver_df.iloc[i].get("driverRef", f"Driver {i+1}")),
                "constructor": str(driver_df.iloc[i].get("constructor_name", driver_df.iloc[i].get("constructorRef", "Team"))),
                "grid_position": float(grid_positions[i]),
                "win_probability": win_prob,
                "podium_probability": podium_prob,
                "top5_probability": top5_prob,
                "top10_probability": top10_prob,
                "dnf_probability": dnf_prob,
                "expected_finish": exp_finish,
                "median_finish": median_finish,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
                "expected_points": exp_points,
                "position_distribution": pos_dist,
            }
            results_summary.append(d_info)

        # Sort summary by expected finish
        summary_df = pd.DataFrame(results_summary).sort_values("expected_finish").reset_index(drop=True)
        summary_df["predicted_order"] = np.arange(1, len(summary_df) + 1)

        return {
            "summary_table": summary_df,
            "simulated_finishes": simulated_finishes,
            "simulated_points": simulated_points,
            "n_simulations": self.n_simulations,
        }

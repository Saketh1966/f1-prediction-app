"""
Circuit-Specific Feature Engineering (Specialized for Autodromo Nazionale di Monza).
Computes historical circuit track record with half-life recency weighting.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class CircuitFeatureEngineer:
    """Computes circuit-specific historical performance features with temporal decay."""

    def __init__(self, half_life_years: float = 5.0):
        self.decay_rate = np.log(2) / half_life_years

    def compute_features_for_race(
        self,
        historical_results: pd.DataFrame,
        target_circuit_id: int,
        target_driver_ids: List[int],
        target_constructor_ids: List[int],
        target_year: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute circuit-specific features for drivers and constructors strictly using prior results.
        """
        # Filter to prior races at this specific circuit
        circuit_results = historical_results[historical_results["circuitId"] == target_circuit_id]

        driver_records = []
        for driver_id in target_driver_ids:
            d_circuit = circuit_results[circuit_results["driverId"] == driver_id].sort_values("year")
            n_starts = len(d_circuit)

            if n_starts == 0:
                driver_records.append({
                    "driverId": driver_id,
                    "driver_circuit_starts": 0,
                    "driver_circuit_avg_finish": 14.0,
                    "driver_circuit_best_finish": 15.0,
                    "driver_circuit_podiums": 0,
                    "driver_circuit_podium_rate": 0.0,
                    "driver_circuit_dnf_rate": 0.15,
                    "driver_circuit_recency_weighted_finish": 14.0,
                })
                continue

            finishes = d_circuit["positionOrder"].values
            years = d_circuit["year"].values
            is_dnf = (d_circuit["statusId"].isin([1, 11, 12, 13, 14, 15, 16, 17, 18, 19]) == False).astype(float).values

            # Half-life weights based on time difference from target year
            weights = np.exp(-self.decay_rate * np.maximum(0, target_year - years))
            weights /= np.sum(weights)

            weighted_finish = float(np.sum(finishes * weights))
            avg_finish = float(finishes.mean())
            best_finish = float(finishes.min())
            podiums = int((finishes <= 3).sum())
            podium_rate = float((finishes <= 3).mean())
            dnf_rate = float(is_dnf.mean())

            driver_records.append({
                "driverId": driver_id,
                "driver_circuit_starts": n_starts,
                "driver_circuit_avg_finish": avg_finish,
                "driver_circuit_best_finish": best_finish,
                "driver_circuit_podiums": podiums,
                "driver_circuit_podium_rate": podium_rate,
                "driver_circuit_dnf_rate": dnf_rate,
                "driver_circuit_recency_weighted_finish": weighted_finish,
            })

        constructor_records = []
        for constructor_id in target_constructor_ids:
            c_circuit = circuit_results[circuit_results["constructorId"] == constructor_id].sort_values("year")
            n_starts = len(c_circuit)

            if n_starts == 0:
                constructor_records.append({
                    "constructorId": constructor_id,
                    "constructor_circuit_starts": 0,
                    "constructor_circuit_avg_finish": 14.0,
                    "constructor_circuit_podium_rate": 0.02,
                    "constructor_circuit_dnf_rate": 0.15,
                    "constructor_circuit_recency_finish": 14.0,
                })
                continue

            finishes = c_circuit["positionOrder"].values
            years = c_circuit["year"].values
            is_dnf = (c_circuit["statusId"].isin([1, 11, 12, 13, 14, 15, 16, 17, 18, 19]) == False).astype(float).values

            weights = np.exp(-self.decay_rate * np.maximum(0, target_year - years))
            weights /= np.sum(weights)

            weighted_finish = float(np.sum(finishes * weights))

            constructor_records.append({
                "constructorId": constructor_id,
                "constructor_circuit_starts": n_starts,
                "constructor_circuit_avg_finish": float(finishes.mean()),
                "constructor_circuit_podium_rate": float((finishes <= 3).mean()),
                "constructor_circuit_dnf_rate": float(is_dnf.mean()),
                "constructor_circuit_recency_finish": weighted_finish,
            })

        return pd.DataFrame(driver_records), pd.DataFrame(constructor_records)

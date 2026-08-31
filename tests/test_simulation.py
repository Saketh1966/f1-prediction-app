"""
Monte Carlo Simulation Engine Unit Tests.
"""

import pytest
import pandas as pd
import numpy as np
from src.simulation.monte_carlo import MonteCarloRaceSimulator


def test_monte_carlo_simulation_integrity():
    n_drivers = 20
    driver_df = pd.DataFrame({
        "driverId": np.arange(1, n_drivers + 1),
        "code": [f"D{i}" for i in range(1, n_drivers + 1)],
        "surname": [f"Driver{i}" for i in range(1, n_drivers + 1)],
        "constructor_name": [f"Team{i//2}" for i in range(1, n_drivers + 1)],
        "grid_position": np.arange(1, n_drivers + 1),
    })

    pred_positions = np.arange(1.0, float(n_drivers) + 1.0)
    dnf_probs = np.full(n_drivers, 0.10)

    sim = MonteCarloRaceSimulator(n_simulations=2000, random_seed=42)
    res = sim.run_simulation(driver_df, pred_positions, dnf_probs)

    summary = res["summary_table"]
    assert len(summary) == n_drivers

    # Win probability sum should be approximately 1.0
    total_win_prob = summary["win_probability"].sum()
    assert abs(total_win_prob - 1.0) < 0.05

    # Podium probability sum should be approximately 3.0
    total_pod_prob = summary["podium_probability"].sum()
    assert abs(total_pod_prob - 3.0) < 0.10

    # Top-10 probability sum should be approximately 10.0
    total_top10_prob = summary["top10_probability"].sum()
    assert abs(total_top10_prob - 10.0) < 0.25

    # Check that points allocated match FIA scoring distribution (sum per race = 25+18+15+12+10+8+6+4+2+1 + 1 fastest lap = 102)
    total_points = summary["expected_points"].sum()
    assert abs(total_points - 102.0) < 2.0

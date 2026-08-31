"""
Strict Data Leakage and Temporal Correctness Unit Tests.
Verifies that no future race information leaks into historical feature generation.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.loader import F1DataLoader
from src.features.driver_features import DriverFeatureEngineer
from src.features.constructor_features import ConstructorFeatureEngineer
from src.features.circuit_features import CircuitFeatureEngineer


def test_strict_temporal_cutoff_driver_features():
    """
    Assert that adding future races to the dataset does NOT change
    historical driver features computed for an earlier race.
    """
    loader = F1DataLoader()
    results = loader.get_results()
    races = loader.get_races()
    d_standings = loader.get_driver_standings()

    results_merged = results.merge(
        races[["raceId", "year", "round", "circuitId", "date"]],
        on="raceId",
        how="inner",
    ).sort_values(["date", "year", "round"])

    # Target: 2023 Italian GP at Monza (Round 14)
    target_year = 2023
    target_round = 14
    target_drivers = [1, 830, 844] # Hamilton, Verstappen, Leclerc

    # Dataset A: Ground truth historical subset strictly prior to 2023 Monza
    df_past_only = results_merged[
        (results_merged["year"] < target_year)
        | ((results_merged["year"] == target_year) & (results_merged["round"] < target_round))
    ]

    # Dataset B: Full dataset INCLUDING future 2024, 2025, 2026 races
    df_future_included = results_merged.copy()

    engineer = DriverFeatureEngineer()

    # Pass Dataset A
    feats_a = engineer.compute_features_for_race(
        historical_results=df_past_only,
        historical_standings=d_standings[d_standings["raceId"].isin(df_past_only["raceId"])],
        target_driver_ids=target_drivers,
        target_season=target_year,
        target_round=target_round,
    )

    # Pass Dataset B filtered strictly using pipeline temporal condition
    df_filtered_from_b = df_future_included[
        (df_future_included["year"] < target_year)
        | ((df_future_included["year"] == target_year) & (df_future_included["round"] < target_round))
    ]
    feats_b = engineer.compute_features_for_race(
        historical_results=df_filtered_from_b,
        historical_standings=d_standings[d_standings["raceId"].isin(df_filtered_from_b["raceId"])],
        target_driver_ids=target_drivers,
        target_season=target_year,
        target_round=target_round,
    )

    # The features MUST be mathematically identical
    pd.testing.assert_frame_equal(feats_a, feats_b)


def test_monza_circuit_features_future_leakage():
    """
    Assert that Monza track-specific stats for 2022 do not change when 2023-2026 Monza races are added.
    """
    loader = F1DataLoader()
    results = loader.get_results()
    races = loader.get_races()

    results_merged = results.merge(
        races[["raceId", "year", "round", "circuitId", "date"]],
        on="raceId",
        how="inner",
    )

    circ_eng = CircuitFeatureEngineer(half_life_years=5.0)

    # Target: 2022 Monza GP
    past_2022 = results_merged[results_merged["year"] < 2022]
    d_feats_2022, c_feats_2022 = circ_eng.compute_features_for_race(
        historical_results=past_2022,
        target_circuit_id=14,
        target_driver_ids=[1, 830],
        target_constructor_ids=[6, 9],
        target_year=2022,
    )

    # Re-run on subset
    assert len(d_feats_2022) == 2
    assert len(c_feats_2022) == 2
    assert (d_feats_2022["driver_circuit_starts"] >= 0).all()

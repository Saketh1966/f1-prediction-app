"""
Unit tests for Driver, Constructor, and Qualifying Feature Engineering.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.driver_features import DriverFeatureEngineer
from src.features.constructor_features import ConstructorFeatureEngineer
from src.features.qualifying_features import QualifyingFeatureEngineer


def test_driver_features_shape_and_bounds():
    dummy_results = pd.DataFrame({
        "driverId": [1, 1, 1, 2, 2],
        "year": [2022, 2022, 2023, 2022, 2023],
        "round": [1, 2, 1, 1, 1],
        "positionOrder": [1, 2, 3, 10, 12],
        "grid": [1, 2, 1, 8, 11],
        "points": [25.0, 18.0, 15.0, 1.0, 0.0],
        "statusId": [1, 1, 1, 1, 1],
    })
    dummy_standings = pd.DataFrame({
        "driverId": [1, 2],
        "position": [1.0, 10.0],
        "points": [58.0, 1.0],
    })

    eng = DriverFeatureEngineer()
    df_feats = eng.compute_features_for_race(
        historical_results=dummy_results,
        historical_standings=dummy_standings,
        target_driver_ids=[1, 2, 999], # 999 is rookie with no starts
        target_season=2023,
        target_round=2,
    )

    assert len(df_feats) == 3
    assert df_feats.loc[df_feats["driverId"] == 1, "driver_career_starts"].values[0] == 3
    assert df_feats.loc[df_feats["driverId"] == 999, "driver_career_starts"].values[0] == 0
    assert (df_feats["driver_career_avg_finish"] >= 1.0).all()
    assert (df_feats["driver_career_avg_finish"] <= 22.0).all()


def test_qualifying_features_pre_and_post_quali():
    dummy_q = pd.DataFrame({
        "raceId": [1, 1, 2, 2],
        "driverId": [1, 2, 1, 2],
        "constructorId": [10, 10, 10, 10],
        "position": [1.0, 2.0, 3.0, 4.0],
    })
    dummy_r = pd.DataFrame()

    q_eng = QualifyingFeatureEngineer()

    # Pre-qualifying mode
    df_pre = q_eng.compute_features_for_race(
        historical_qualifying=dummy_q,
        historical_results=dummy_r,
        target_driver_ids=[1, 2],
        target_driver_constructors={1: 10, 2: 10},
        is_post_qualifying=False,
    )
    assert df_pre["is_post_qualifying_feature"].iloc[0] == 0.0

    # Post-qualifying mode with confirmed grid
    df_post = q_eng.compute_features_for_race(
        historical_qualifying=dummy_q,
        historical_results=dummy_r,
        target_driver_ids=[1, 2],
        target_driver_constructors={1: 10, 2: 10},
        is_post_qualifying=True,
        current_grid_positions={1: 1, 2: 5},
    )
    assert df_post.loc[df_post["driverId"] == 1, "grid_position"].values[0] == 1.0
    assert df_post.loc[df_post["driverId"] == 2, "grid_position"].values[0] == 5.0

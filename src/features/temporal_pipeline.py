"""
Temporal Feature Engineering Pipeline (Zero Data Leakage Engine).
Iterates chronologically through Grand Prix events, creating historical features
using strictly pre-race information.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.loader import F1DataLoader
from src.features.driver_features import DriverFeatureEngineer
from src.features.constructor_features import ConstructorFeatureEngineer
from src.features.circuit_features import CircuitFeatureEngineer
from src.features.qualifying_features import QualifyingFeatureEngineer
from src.features.lap_pit_features import LapAndPitFeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TemporalFeaturePipeline:
    """End-to-end temporal feature generation pipeline enforcing strict time boundaries."""

    def __init__(self, data_loader: Optional[F1DataLoader] = None, start_year: int = 1995):
        self.loader = data_loader or F1DataLoader()
        self.start_year = start_year
        self.driver_eng = DriverFeatureEngineer()
        self.constructor_eng = ConstructorFeatureEngineer()
        self.circuit_eng = CircuitFeatureEngineer()
        self.quali_eng = QualifyingFeatureEngineer()
        self.lap_pit_eng = LapAndPitFeatureEngineer()

    def build_all_features(
        self,
        output_dir: str = "data/processed",
        target_monza_race_id: int = 1181,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Construct features chronologically for all races from start_year to present.
        Returns (historical_feature_matrix, monza_2026_target_features).
        """
        os.makedirs(output_dir, exist_ok=True)
        races = self.loader.get_races()
        results = self.loader.get_results()
        qualifying = self.loader.get_qualifying()
        lap_times = self.loader.get_lap_times()
        pit_stops = self.loader.get_pit_stops()
        d_standings = self.loader.get_driver_standings()
        c_standings = self.loader.get_constructor_standings()
        drivers = self.loader.get_drivers()
        constructors = self.loader.get_constructors()

        # Merge race date and year into results for fast temporal slicing
        results_merged = results.merge(
            races[["raceId", "year", "round", "circuitId", "date"]],
            on="raceId",
            how="inner",
        ).sort_values(["date", "year", "round"])

        # Filter races to process from start_year
        eligible_races = races[races["year"] >= self.start_year].sort_values(["date", "year", "round"])

        all_race_features = []
        monza_target_df = pd.DataFrame()

        logger.info(f"Generating temporal features for {len(eligible_races)} races starting from {self.start_year}...")

        # Pre-cache 2026 driver-constructor mappings for the upcoming race
        latest_race_results = results_merged[results_merged["year"] == 2026].sort_values(["round", "positionOrder"])
        if len(latest_race_results) > 0:
            last_round_results = latest_race_results[
                latest_race_results["round"] == latest_race_results["round"].max()
            ]
            active_2026_grid = last_round_results[["driverId", "constructorId"]].drop_duplicates()
        else:
            active_2026_grid = pd.DataFrame()

        for idx, race_row in tqdm(eligible_races.iterrows(), total=len(eligible_races), desc="Processing Races"):
            race_id = int(race_row["raceId"])
            year = int(race_row["year"])
            round_num = int(race_row["round"])
            circuit_id = int(race_row["circuitId"])
            race_date = race_row["date"]

            # Strict temporal filter: strictly past races
            past_results = results_merged[
                (results_merged["date"] < race_date)
                | ((results_merged["year"] < year) | ((results_merged["year"] == year) & (results_merged["round"] < round_num)))
            ]
            past_race_ids = set(past_results["raceId"].unique())

            past_standings_d = d_standings[d_standings["raceId"].isin(past_race_ids)]
            past_standings_c = c_standings[c_standings["raceId"].isin(past_race_ids)]
            past_qualifying = qualifying[qualifying["raceId"].isin(past_race_ids)]
            past_lap_times = lap_times[lap_times["raceId"].isin(past_race_ids)]
            past_pit_stops = pit_stops[pit_stops["raceId"].isin(past_race_ids)]

            # Check if this race has results (historical) or is upcoming (target Monza)
            race_actual_results = results[results["raceId"] == race_id]

            if len(race_actual_results) > 0:
                target_driver_ids = race_actual_results["driverId"].tolist()
                target_constructor_map = dict(zip(race_actual_results["driverId"], race_actual_results["constructorId"]))
                target_constructor_ids = list(set(race_actual_results["constructorId"].tolist()))
                target_grid_map = dict(zip(race_actual_results["driverId"], race_actual_results["grid"]))
                is_target_race = False
            elif race_id == target_monza_race_id or (year == 2026 and round_num == 13):
                # 2026 Monza Target Grid
                target_driver_ids = active_2026_grid["driverId"].tolist()
                target_constructor_map = dict(zip(active_2026_grid["driverId"], active_2026_grid["constructorId"]))
                target_constructor_ids = list(set(active_2026_grid["constructorId"].tolist()))
                target_grid_map = {}
                is_target_race = True
            else:
                continue

            # 1. Driver features
            df_driver = self.driver_eng.compute_features_for_race(
                past_results, past_standings_d, target_driver_ids, year, round_num
            )

            # 2. Constructor features
            df_constructor = self.constructor_eng.compute_features_for_race(
                past_results, past_standings_c, target_constructor_ids, year, round_num
            )

            # 3. Circuit features
            df_circuit_d, df_circuit_c = self.circuit_eng.compute_features_for_race(
                past_results, circuit_id, target_driver_ids, target_constructor_ids, year
            )

            # 4. Qualifying / Grid features
            df_quali = self.quali_eng.compute_features_for_race(
                past_qualifying,
                past_results,
                target_driver_ids,
                target_constructor_map,
                is_post_qualifying=not is_target_race,
                current_grid_positions=target_grid_map if not is_target_race else None,
            )

            # 5. Lap & Pit features
            df_lap_d, df_pit_c = self.lap_pit_eng.compute_features_for_race(
                past_lap_times, past_pit_stops, target_driver_ids, target_constructor_ids
            )

            # Assemble driver-level record
            race_df = df_driver.merge(df_circuit_d, on="driverId", how="left")
            race_df = race_df.merge(df_quali, on="driverId", how="left")
            race_df = race_df.merge(df_lap_d, on="driverId", how="left")

            # Map constructor ID
            race_df["constructorId"] = race_df["driverId"].map(target_constructor_map)

            # Merge constructor-level features
            race_df = race_df.merge(df_constructor, on="constructorId", how="left")
            race_df = race_df.merge(df_circuit_c, on="constructorId", how="left")
            race_df = race_df.merge(df_pit_c, on="constructorId", how="left")

            # Add race context
            race_df["raceId"] = race_id
            race_df["year"] = year
            race_df["round"] = round_num
            race_df["circuitId"] = circuit_id
            race_df["is_monza"] = 1 if circuit_id == 14 else 0

            # Add labels if historical
            if not is_target_race and len(race_actual_results) > 0:
                race_actual = race_actual_results[
                    ["driverId", "positionOrder", "points", "statusId", "grid"]
                ].rename(columns={"grid": "actual_grid"})

                race_df = race_df.merge(race_actual, on="driverId", how="inner")
                race_df["is_dnf"] = (
                    ~race_df["statusId"].isin([1, 11, 12, 13, 14, 15, 16, 17, 18, 19])
                ).astype(int)
                race_df["is_win"] = (race_df["positionOrder"] == 1).astype(int)
                race_df["is_podium"] = (race_df["positionOrder"] <= 3).astype(int)
                race_df["is_top5"] = (race_df["positionOrder"] <= 5).astype(int)
                race_df["is_top10"] = (race_df["positionOrder"] <= 10).astype(int)

                all_race_features.append(race_df)
            elif is_target_race:
                # Merge bio info for target presentation
                race_df = race_df.merge(
                    drivers[["driverId", "driverRef", "code", "forename", "surname", "nationality"]],
                    on="driverId",
                    how="left",
                ).merge(
                    constructors[["constructorId", "constructorRef", "name"]].rename(
                        columns={"name": "constructor_name"}
                    ),
                    on="constructorId",
                    how="left",
                )
                monza_target_df = race_df

        train_matrix = pd.concat(all_race_features, ignore_index=True)

        # Fill any residual NAs with column medians
        exclude_cols = [
            "raceId", "driverId", "constructorId", "circuitId", "year", "round",
            "positionOrder", "points", "statusId", "is_dnf", "is_win", "is_podium",
            "is_top5", "is_top10", "actual_grid", "driverRef", "code", "forename", "surname",
            "nationality", "constructorRef", "constructor_name"
        ]
        feature_cols = [c for c in train_matrix.columns if c not in exclude_cols]

        train_matrix[feature_cols] = train_matrix[feature_cols].fillna(train_matrix[feature_cols].median())
        if len(monza_target_df) > 0:
            target_feat_cols = [c for c in feature_cols if c in monza_target_df.columns]
            monza_target_df[target_feat_cols] = monza_target_df[target_feat_cols].fillna(train_matrix[target_feat_cols].median())

        # Save artifacts
        train_csv = os.path.join(output_dir, "train_features.csv")
        train_parquet = os.path.join(output_dir, "train_features.parquet")
        target_csv = os.path.join(output_dir, "monza_2026_target_features.csv")
        meta_json = os.path.join(output_dir, "feature_metadata.json")

        train_matrix.to_csv(train_csv, index=False)
        try:
            train_matrix.to_parquet(train_parquet, index=False)
        except Exception:
            pass

        if len(monza_target_df) > 0:
            monza_target_df.to_csv(target_csv, index=False)

        metadata = {
            "num_samples": len(train_matrix),
            "num_races": int(train_matrix["raceId"].nunique()),
            "years_covered": [int(train_matrix["year"].min()), int(train_matrix["year"].max())],
            "feature_columns": feature_cols,
            "target_monza_drivers": len(monza_target_df),
        }
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Feature matrix built successfully: {train_matrix.shape[0]} rows, {len(feature_cols)} features.")
        logger.info(f"Target 2026 Monza grid: {len(monza_target_df)} drivers.")

        return train_matrix, monza_target_df


if __name__ == "__main__":
    pipeline = TemporalFeaturePipeline(start_year=2000)
    pipeline.build_all_features()

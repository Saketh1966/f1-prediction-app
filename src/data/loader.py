"""
F1 Data Loader and Schema Validator.
Loads all historical F1 CSV files with robust type parsing, missing value handling,
and referential integrity verification.
"""

import os
import glob
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


class F1DataLoader:
    """Loads and validates historical Formula 1 tabular datasets."""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.tables: Dict[str, pd.DataFrame] = {}
        self._load_all_tables()

    def _load_all_tables(self) -> None:
        """Load all CSV files from raw data directory with standard NA parsing."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found.")

        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in '{self.data_dir}'.")

        for file_path in csv_files:
            table_name = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_csv(
                file_path,
                na_values=["\\N", "null", "None", "", "nan", "NaN"],
                low_memory=False,
            )
            self.tables[table_name] = df

        self._clean_and_cast_types()

    def _clean_and_cast_types(self) -> None:
        """Clean data types, dates, and identifier columns across tables."""
        # Races
        if "races" in self.tables:
            races = self.tables["races"]
            races["date"] = pd.to_datetime(races["date"], errors="coerce")
            races["year"] = pd.to_numeric(races["year"], errors="coerce").astype(int)
            races["round"] = pd.to_numeric(races["round"], errors="coerce").astype(int)
            races["circuitId"] = pd.to_numeric(races["circuitId"], errors="coerce").astype(int)
            self.tables["races"] = races.sort_values(["date", "year", "round"]).reset_index(drop=True)

        # Results
        if "results" in self.tables:
            res = self.tables["results"]
            res["raceId"] = pd.to_numeric(res["raceId"], errors="coerce").astype(int)
            res["driverId"] = pd.to_numeric(res["driverId"], errors="coerce").astype(int)
            res["constructorId"] = pd.to_numeric(res["constructorId"], errors="coerce").astype(int)
            res["grid"] = pd.to_numeric(res["grid"], errors="coerce").fillna(20).astype(int)
            res["positionOrder"] = pd.to_numeric(res["positionOrder"], errors="coerce").astype(int)
            res["points"] = pd.to_numeric(res["points"], errors="coerce").fillna(0.0)
            res["laps"] = pd.to_numeric(res["laps"], errors="coerce").fillna(0).astype(int)
            res["statusId"] = pd.to_numeric(res["statusId"], errors="coerce").fillna(0).astype(int)
            self.tables["results"] = res

        # Qualifying
        if "qualifying" in self.tables:
            q = self.tables["qualifying"]
            q["raceId"] = pd.to_numeric(q["raceId"], errors="coerce").astype(int)
            q["driverId"] = pd.to_numeric(q["driverId"], errors="coerce").astype(int)
            q["constructorId"] = pd.to_numeric(q["constructorId"], errors="coerce").astype(int)
            q["position"] = pd.to_numeric(q["position"], errors="coerce")
            self.tables["qualifying"] = q

        # Lap Times
        if "lap_times" in self.tables:
            lt = self.tables["lap_times"]
            lt["raceId"] = pd.to_numeric(lt["raceId"], errors="coerce").astype(int)
            lt["driverId"] = pd.to_numeric(lt["driverId"], errors="coerce").astype(int)
            lt["lap"] = pd.to_numeric(lt["lap"], errors="coerce").astype(int)
            lt["milliseconds"] = pd.to_numeric(lt["milliseconds"], errors="coerce")
            self.tables["lap_times"] = lt

        # Pit Stops
        if "pit_stops" in self.tables:
            ps = self.tables["pit_stops"]
            ps["raceId"] = pd.to_numeric(ps["raceId"], errors="coerce").astype(int)
            ps["driverId"] = pd.to_numeric(ps["driverId"], errors="coerce").astype(int)
            ps["stop"] = pd.to_numeric(ps["stop"], errors="coerce").astype(int)
            ps["milliseconds"] = pd.to_numeric(ps["milliseconds"], errors="coerce")
            self.tables["pit_stops"] = ps

        # Standings
        if "driver_standings" in self.tables:
            ds = self.tables["driver_standings"]
            ds["raceId"] = pd.to_numeric(ds["raceId"], errors="coerce").astype(int)
            ds["driverId"] = pd.to_numeric(ds["driverId"], errors="coerce").astype(int)
            ds["points"] = pd.to_numeric(ds["points"], errors="coerce").fillna(0.0)
            ds["position"] = pd.to_numeric(ds["position"], errors="coerce").fillna(20.0)
            ds["wins"] = pd.to_numeric(ds["wins"], errors="coerce").fillna(0).astype(int)
            self.tables["driver_standings"] = ds

        if "constructor_standings" in self.tables:
            cs = self.tables["constructor_standings"]
            cs["raceId"] = pd.to_numeric(cs["raceId"], errors="coerce").astype(int)
            cs["constructorId"] = pd.to_numeric(cs["constructorId"], errors="coerce").astype(int)
            cs["points"] = pd.to_numeric(cs["points"], errors="coerce").fillna(0.0)
            cs["position"] = pd.to_numeric(cs["position"], errors="coerce").fillna(10.0)
            cs["wins"] = pd.to_numeric(cs["wins"], errors="coerce").fillna(0).astype(int)
            self.tables["constructor_standings"] = cs

    def get_table(self, name: str) -> pd.DataFrame:
        """Retrieve a specific table by name."""
        if name not in self.tables:
            raise KeyError(f"Table '{name}' not found. Available: {list(self.tables.keys())}")
        return self.tables[name].copy()

    def get_races(self) -> pd.DataFrame:
        return self.get_table("races")

    def get_results(self) -> pd.DataFrame:
        return self.get_table("results")

    def get_drivers(self) -> pd.DataFrame:
        return self.get_table("drivers")

    def get_constructors(self) -> pd.DataFrame:
        return self.get_table("constructors")

    def get_circuits(self) -> pd.DataFrame:
        return self.get_table("circuits")

    def get_qualifying(self) -> pd.DataFrame:
        return self.get_table("qualifying")

    def get_lap_times(self) -> pd.DataFrame:
        return self.get_table("lap_times")

    def get_pit_stops(self) -> pd.DataFrame:
        return self.get_table("pit_stops")

    def get_status(self) -> pd.DataFrame:
        return self.get_table("status")

    def get_driver_standings(self) -> pd.DataFrame:
        return self.get_table("driver_standings")

    def get_constructor_standings(self) -> pd.DataFrame:
        return self.get_table("constructor_standings")

    def validate_relationships(self) -> Dict[str, bool]:
        """Verify referential integrity across all key foreign keys."""
        results = self.tables["results"]
        races = self.tables["races"]
        drivers = self.tables["drivers"]
        constructors = self.tables["constructors"]

        validations = {
            "results_raceId_valid": results["raceId"].isin(races["raceId"]).all(),
            "results_driverId_valid": results["driverId"].isin(drivers["driverId"]).all(),
            "results_constructorId_valid": results["constructorId"].isin(constructors["constructorId"]).all(),
            "races_circuitId_valid": races["circuitId"].isin(self.tables["circuits"]["circuitId"]).all(),
        }
        return validations


if __name__ == "__main__":
    loader = F1DataLoader()
    print("Tables loaded successfully:", list(loader.tables.keys()))
    print("Referential integrity check:", loader.validate_relationships())

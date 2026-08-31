"""
Data Dictionary Generator.
Generates comprehensive Markdown documentation detailing all input tables,
relationships, column types, and engineered feature definitions.
"""

import os
from typing import Dict, List
import pandas as pd
from src.data.loader import F1DataLoader


def generate_data_dictionary(output_path: str = "docs/data_dictionary.md") -> None:
    """Generate Markdown Data Dictionary file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    loader = F1DataLoader()

    doc = []
    doc.append("# Formula 1 Race Prediction System: Comprehensive Data Dictionary\n")
    doc.append("This document outlines the raw historical F1 dataset schemas, entity relationships, primary/foreign keys, and engineered machine learning features.\n")
    doc.append("## 1. Raw Dataset Tables Overview\n")
    doc.append("| Table Name | Rows | Columns | Primary Key | Description |")
    doc.append("| :--- | :--- | :--- | :--- | :--- |")

    table_descriptions = {
        "circuits": "Grand Prix circuit metadata (coordinates, altitude, country, track ref)",
        "constructors": "F1 teams / constructors (name, nationality, constructorRef)",
        "constructor_results": "Constructor race point tallies and DNF statuses per event",
        "constructor_standings": "Constructor Championship points and standings after each round",
        "drivers": "Driver bio data (names, driverRef, DOB, nationality, code, number)",
        "driver_standings": "Driver Championship points, rank, and win tallies after each round",
        "lap_times": "Lap-by-lap timing records (lap, position, milliseconds) across races",
        "pit_stops": "Pit stop event records (stop number, lap, duration, milliseconds)",
        "qualifying": "Qualifying classification records (Q1, Q2, Q3 lap times, grid position)",
        "races": "Grand Prix event schedule, circuit references, dates, and session times",
        "results": "Official Grand Prix finishing classification, points, grid, laps, status",
        "seasons": "F1 championship season list (1950 to present)",
        "sprint_results": "Sprint race classification and points records",
        "status": "Race finishing status reference (Finished, Engine, Collision, etc.)",
    }

    for name, df in sorted(loader.tables.items()):
        pk = f"{name[:-1] if name.endswith('s') else name}Id"
        if pk not in df.columns:
            pk = df.columns[0]
        desc = table_descriptions.get(name, "F1 raw data table")
        doc.append(f"| `{name}.csv` | {len(df):,} | {len(df.columns)} | `{pk}` | {desc} |")

    doc.append("\n## 2. Table Schemas & Foreign Key Relationships\n")
    for name, df in sorted(loader.tables.items()):
        doc.append(f"### `{name}.csv`")
        doc.append(f"**Shape:** `{df.shape[0]:,}` rows × `{df.shape[1]}` columns\n")
        doc.append("| Column | Data Type | Non-Null % | Key Type | Sample Values |")
        doc.append("| :--- | :--- | :--- | :--- | :--- |")

        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = f"{(df[col].notnull().mean() * 100):.1f}%"
            key_type = "Normal"
            if col.lower().endswith("id"):
                if col.lower() == f"{name[:-1] if name.endswith('s') else name}id".lower():
                    key_type = "**Primary Key**"
                else:
                    key_type = "*Foreign Key*"
            samples = ", ".join([str(x) for x in df[col].dropna().unique()[:3]])
            if len(samples) > 40:
                samples = samples[:37] + "..."
            doc.append(f"| `{col}` | `{dtype}` | {non_null} | {key_type} | {samples} |")
        doc.append("\n")

    doc.append("## 3. Engineered Machine Learning Feature Dictionary\n")
    doc.append("All engineered features are strictly temporal: for any race at date $T$, only data strictly prior to $T$ ($t < T$) is used.\n")
    doc.append("| Feature Name | Group | Type | Description | Leakage Prevention Rule |")
    doc.append("| :--- | :--- | :--- | :--- | :--- |")

    features_info = [
        ("driver_form_ewm_finish", "Driver Form", "float", "Exponentially weighted moving average of finishing position", "Calculated strictly on prior races (shift 1)"),
        ("driver_rolling_finish_last3", "Driver Form", "float", "Mean finishing position across previous 3 races", "Prior races only"),
        ("driver_rolling_finish_last5", "Driver Form", "float", "Mean finishing position across previous 5 races", "Prior races only"),
        ("driver_season_avg_finish", "Driver Form", "float", "Season-to-date average finishing position", "Rounds $1 \\dots (k-1)$ of current season"),
        ("driver_career_avg_finish", "Driver Form", "float", "Career-to-date average finishing position", "All historical career starts prior to current round"),
        ("driver_recent_points_sum5", "Driver Form", "float", "Total championship points scored in last 5 races", "Prior 5 races"),
        ("driver_championship_stand_pos", "Driver Form", "float", "Championship standing position entering the weekend", "Previous race standings table"),
        ("driver_championship_points", "Driver Form", "float", "Championship points accumulated entering the weekend", "Previous race standings table"),
        ("driver_career_podium_rate", "Driver Track Record", "float", "Proportion of career starts resulting in top-3 finish", "Strictly past races"),
        ("driver_career_win_rate", "Driver Track Record", "float", "Proportion of career starts resulting in P1 victory", "Strictly past races"),
        ("driver_career_top10_rate", "Driver Track Record", "float", "Proportion of career starts resulting in top-10 finish", "Strictly past races"),
        ("driver_career_dnf_rate", "Driver Reliability", "float", "Historical rate of race retirements (status != Finished/Lapped)", "Past starts"),
        ("driver_recent_dnf_rate_5", "Driver Reliability", "float", "Rate of retirements in the last 5 races", "Prior 5 starts"),
        ("driver_monza_starts", "Monza Circuit", "int", "Total career race starts at Autodromo Nazionale di Monza", "Prior Monza GPs"),
        ("driver_monza_avg_finish", "Monza Circuit", "float", "Average finishing position specifically at Monza", "Prior Monza GPs (imputed if 0 starts)"),
        ("driver_monza_podiums", "Monza Circuit", "int", "Career podium finishes at Monza", "Prior Monza GPs"),
        ("driver_monza_dnf_rate", "Monza Circuit", "float", "Historical DNF rate specifically at Monza", "Prior Monza GPs"),
        ("driver_monza_recency_weighted_finish", "Monza Circuit", "float", "Half-life weighted Monza performance (recent years heavier)", "Time-decay on prior Monza GPs"),
        ("constructor_rolling_points_last5", "Constructor Pace", "float", "Constructor championship points in last 5 rounds", "Prior constructor race points"),
        ("constructor_recent_avg_finish", "Constructor Pace", "float", "Constructor average finishing position across both cars (last 5)", "Prior races"),
        ("constructor_season_points", "Constructor Pace", "float", "Current constructor championship points entering race", "Prior standings"),
        ("constructor_season_rank", "Constructor Pace", "float", "Constructor championship rank entering race", "Prior standings"),
        ("constructor_career_podium_rate", "Constructor Pace", "float", "Historical constructor podium rate", "Prior races"),
        ("constructor_monza_avg_finish", "Constructor Circuit", "float", "Constructor historical average finish at Monza", "Prior Monza races"),
        ("constructor_monza_podium_rate", "Constructor Circuit", "float", "Constructor historical podium rate at Monza", "Prior Monza races"),
        ("constructor_reliability_dnf_rate", "Constructor Reliability", "float", "Constructor mechanical/overall DNF rate over last 10 races", "Prior 10 constructor starts"),
        ("driver_quali_rolling_avg_last5", "Qualifying Form", "float", "Average qualifying position over prior 5 sessions", "Prior qualifying sessions"),
        ("driver_quali_vs_teammate_diff", "Qualifying Form", "float", "Driver average qualifying position delta against direct teammate", "Prior qualifying sessions"),
        ("grid_position", "Grid (Post-Quali)", "int", "Starting grid position on race day (2026 Monza confirmed/simulated)", "Set to expected grid pre-quali, actual post-quali"),
        ("driver_lap_pace_rel_median", "Lap Pace", "float", "Historical race pace relative to field median from `lap_times.csv`", "Prior aggregated race laps"),
        ("driver_lap_pace_std", "Lap Pace", "float", "Historical lap-time standard deviation (consistency)", "Prior aggregated race laps"),
        ("constructor_pit_duration_mean", "Pit Stop", "float", "Constructor mean pit stop stationary duration (seconds)", "Prior pit stops from `pit_stops.csv`"),
        ("constructor_pit_duration_std", "Pit Stop", "float", "Constructor pit stop consistency / variance", "Prior pit stops from `pit_stops.csv`"),
    ]

    for name, group, dtype, desc, leak_rule in features_info:
        doc.append(f"| `{name}` | {group} | `{dtype}` | {desc} | {leak_rule} |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc) + "\n")

    print(f"Data dictionary generated at: {output_path}")


if __name__ == "__main__":
    generate_data_dictionary()

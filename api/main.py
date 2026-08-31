"""
FastAPI REST Application for Formula 1 Monza Race Prediction & Simulation.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query, Path
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

from api.schemas import (
    HealthResponse,
    DriverProfile,
    RacePredictionResponse,
    SimulationRequest,
    DriverExplanationResponse,
    ModelBenchmarkSummary,
    CircuitProfileResponse,
)
from src.prediction.monza_predictor import MonzaPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Formula 1 Monza GP Prediction & Simulation API",
    description="Production-grade ML prediction, ranking, and Monte Carlo race simulation for the Italian GP at Monza.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Predictor Instance
_predictor: Optional[MonzaPredictor] = None


def get_predictor() -> MonzaPredictor:
    global _predictor
    if _predictor is None:
        _predictor = MonzaPredictor()
    return _predictor


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health check endpoint confirming API availability and target race."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "target_race": "2026 Formula 1 Italian Grand Prix at Monza (Round 13)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/drivers", response_model=List[DriverProfile], tags=["Grid"])
def get_drivers():
    """Retrieve the participating driver lineup for the 2026 Italian GP."""
    target_csv = "data/processed/monza_2026_target_features.csv"
    if not os.path.exists(target_csv):
        raise HTTPException(status_code=503, detail="Target dataset not yet generated. Run feature build.")

    df = pd.read_csv(target_csv)
    drivers = []
    for _, row in df.iterrows():
        drivers.append({
            "driver_id": int(row["driverId"]),
            "code": str(row.get("code", "DRV")),
            "name": f"{row.get('forename', '')} {row.get('surname', '')}".strip() or str(row.get("driverRef", "Driver")),
            "constructor": str(row.get("constructor_name", "Team")),
            "nationality": str(row.get("nationality", "Unknown")),
            "championship_points": float(row.get("driver_championship_points", 0.0)),
            "championship_rank": float(row.get("driver_championship_stand_pos", 20.0)),
        })
    return sorted(drivers, key=lambda x: x["championship_rank"])


@app.get("/prediction", response_model=RacePredictionResponse, tags=["Prediction"])
def get_race_prediction(
    is_post_qualifying: bool = Query(False, description="Whether to incorporate confirmed qualifying grid"),
    n_simulations: int = Query(10000, ge=100, le=50000, description="Monte Carlo simulation iterations"),
):
    """
    Generate race predictions, probabilities, and 10,000 Monte Carlo race simulations
    for the upcoming Italian Grand Prix at Monza.
    """
    try:
        predictor = get_predictor()
        res = predictor.predict_monza(
            n_simulations=n_simulations,
            is_post_qualifying=is_post_qualifying,
        )
        return res
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prediction/{driver_id}", response_model=Dict[str, Any], tags=["Prediction"])
def get_driver_prediction_and_explanation(
    driver_id: int = Path(..., description="Unique F1 Driver ID")
):
    """
    Retrieve predicted outcome and local SHAP explainability feature attributions for an individual driver.
    """
    try:
        predictor = get_predictor()
        sim_res = predictor.predict_monza(n_simulations=5000)
        driver_stats = next(
            (d for d in sim_res["full_grid_predictions"] if d["driverId"] == driver_id),
            None
        )
        if not driver_stats:
            raise HTTPException(status_code=404, detail=f"Driver ID {driver_id} not found in target grid.")

        explanation = predictor.get_driver_explanation(driver_id)
        return {
            "driver_profile": driver_stats,
            "shap_explanation": explanation,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching driver {driver_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulation", response_model=RacePredictionResponse, tags=["Simulation"])
def run_custom_simulation(request: SimulationRequest):
    """
    Execute custom Monte Carlo race simulations with optional starting grid position overrides.
    """
    try:
        predictor = get_predictor()
        res = predictor.predict_monza(
            n_simulations=request.n_simulations,
            is_post_qualifying=request.is_post_qualifying,
            custom_grid=request.grid_overrides,
        )
        return res
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-performance", response_model=Dict[str, Any], tags=["Evaluation"])
def get_model_performance():
    """
    Retrieve historical walk-forward backtesting metrics across models and test seasons.
    """
    summary_json = "models/benchmark_summary.json"
    if not os.path.exists(summary_json):
        raise HTTPException(status_code=503, detail="Model benchmarks not available. Run backtesting first.")

    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@app.get("/circuit/monza", response_model=CircuitProfileResponse, tags=["Circuit"])
def get_monza_circuit_profile():
    """
    Retrieve track telemetry characteristics, overtaking difficulty, and historical Monza stats.
    """
    return {
        "circuit_id": 14,
        "name": "Autodromo Nazionale di Monza (Temple of Speed)",
        "location": "Monza",
        "country": "Italy",
        "track_length_km": 5.793,
        "laps": 53,
        "race_distance_km": 306.720,
        "lap_record": "1:21.046 (Rubens Barrichello, Ferrari, 2004)",
        "corners": 11,
        "drs_zones": 2,
        "overtaking_index": "High (Slipstream & DRS on Rettifilo and Curva Grande)",
        "monza_historical_stats": {
            "pole_to_win_rate": "54.2%",
            "top3_grid_to_podium_rate": "78.6%",
            "safety_car_probability": "55.0%",
            "avg_pit_stationary_loss_sec": 24.2,
            "full_throttle_percentage": "76%",
            "top_speed_kmh": 355.0,
        },
    }

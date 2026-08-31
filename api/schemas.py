"""
Pydantic Schemas for FastAPI REST Endpoints.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str
    target_race: str
    timestamp: str


class DriverProfile(BaseModel):
    driver_id: int
    code: str
    name: str
    constructor: str
    nationality: str
    championship_points: float
    championship_rank: float


class DriverPredictionItem(BaseModel):
    driverId: int
    driver_code: str
    driver_name: str
    constructor: str
    grid_position: float
    win_probability: float
    podium_probability: float
    top5_probability: float
    top10_probability: float
    dnf_probability: float
    expected_finish: float
    median_finish: float
    ci_95_lower: float
    ci_95_upper: float
    expected_points: float
    predicted_order: int


class RacePredictionResponse(BaseModel):
    grand_prix: str
    circuit: str
    round: int
    season: int
    mode: str
    n_simulations: int
    predicted_winner: Dict[str, Any]
    predicted_podium: List[Dict[str, Any]]
    full_grid_predictions: List[DriverPredictionItem]


class SimulationRequest(BaseModel):
    n_simulations: int = Field(10000, ge=100, le=50000)
    grid_overrides: Optional[Dict[int, int]] = None
    is_post_qualifying: bool = False


class DriverExplanationResponse(BaseModel):
    driver_id: int
    driver_name: str
    base_value: float
    predicted_finish: float
    top_positive_factors: List[Dict[str, Any]]
    top_negative_factors: List[Dict[str, Any]]


class ModelBenchmarkSummary(BaseModel):
    models_evaluated: List[str]
    test_seasons: List[int]
    overall_summary: List[Dict[str, Any]]
    best_model_by_mae: str


class CircuitProfileResponse(BaseModel):
    circuit_id: int
    name: str
    location: str
    country: str
    track_length_km: float
    laps: int
    race_distance_km: float
    lap_record: str
    corners: int
    drs_zones: int
    overtaking_index: str
    monza_historical_stats: Dict[str, Any]

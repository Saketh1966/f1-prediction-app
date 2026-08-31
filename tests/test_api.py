"""
FastAPI REST API Endpoints Integration Tests.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Monza" in data["target_race"]


def test_api_circuit_endpoint():
    response = client.get("/circuit/monza")
    assert response.status_code == 200
    data = response.json()
    assert data["circuit_id"] == 14
    assert "Monza" in data["name"]
    assert data["laps"] == 53

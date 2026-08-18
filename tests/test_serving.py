# tests/test_serving.py
import pytest
from fastapi.testclient import TestClient
from src.serving.app import app

@pytest.fixture(scope="module")
def client():
    """Initializes TestClient and runs lifespan startup on real data."""
    with TestClient(app) as test_client:
        yield test_client

def test_health_endpoint_on_real_data(client):
    """Verifies that the /health endpoint confirms model and real data are loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["data_source"] != "None"

def test_predict_real_trip_payload(client):
    """Verifies scoring of a real Midtown-to-Upper East Side trip."""
    payload = {
        "trip_distance": 2.8,
        "fare_amount": 14.50,
        "tolls_amount": 0.0,
        "passenger_count": 1.0,
        "PULocationID": "161",
        "DOLocationID": "236",
        "payment_type": "1",
        "RatecodeID": "1"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["high_tip_prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert data["latency_ms"] > 0.0

def test_predict_invalid_fare_boundary_rejected(client):
    """Verifies that invalid fare amounts (< 2.50) are rejected with HTTP 422."""
    bad_payload = {
        "trip_distance": 2.8,
        "fare_amount": 1.00,  # Below legal minimum!
        "tolls_amount": 0.0,
        "passenger_count": 1.0,
        "PULocationID": "161",
        "DOLocationID": "236",
        "payment_type": "1",
        "RatecodeID": "1"
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
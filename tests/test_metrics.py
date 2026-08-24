# tests/test_metrics.py
import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture(scope="module")
def client():
    """Initializes TestClient and executes lifespan startup."""
    with TestClient(app) as test_client:
        yield test_client


def test_metrics_endpoint_returns_200(client):
    """Verifies that GET /metrics returns valid Prometheus exposition text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    content = response.text
    # Verify core operational metrics are registered
    assert "http_requests_total" in content
    assert "http_request_duration_seconds" in content
    # Verify ML domain metrics are registered
    assert "taxi_predictions_total" in content
    assert "taxi_model_load_status" in content
    assert "taxi_feature_fare_amount_dollars" in content
    assert "taxi_feature_trip_distance_miles" in content


def test_metrics_updated_on_prediction(client):
    """Verifies that executing a prediction increments custom ML telemetry metrics."""
    payload = {
        "trip_distance": 4.2,
        "fare_amount": 22.50,
        "tolls_amount": 0.0,
        "passenger_count": 2.0,
        "PULocationID": "161",
        "DOLocationID": "236",
        "payment_type": "1",
        "RatecodeID": "1"
    }

    # Execute inference
    predict_response = client.post("/predict", json=payload)
    assert predict_response.status_code == 200

    # Fetch Prometheus metrics
    metrics_response = client.get("/metrics")
    metrics_text = metrics_response.text

    # Verify prediction counter was incremented
    assert 'taxi_predictions_total{model_version=' in metrics_text
    # Verify fare amount histogram recorded observation
    assert "taxi_feature_fare_amount_dollars_count" in metrics_text
    # Verify model load status gauge is 1 (healthy)
    assert "taxi_model_load_status{" in metrics_text

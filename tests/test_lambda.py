# tests/test_lambda.py
import json

from src.serving.lambda_handler import handler


def test_lambda_healthcheck():
    """Verifies that Lambda handler responds to GET /health requests."""
    event = {
        "rawPath": "/health",
        "requestContext": {
            "http": {
                "method": "GET"
            }
        }
    }
    response = handler(event)
    assert response["statusCode"] == 200

    body = json.loads(response["body"])
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["runtime"] == "AWS Lambda Serverless"


def test_lambda_valid_prediction():
    """Verifies that Lambda scores a valid NYC Taxi payload and returns probability."""
    payload = {
        "trip_distance": 3.4,
        "fare_amount": 18.50,
        "tolls_amount": 0.0,
        "passenger_count": 1.0,
        "PULocationID": "161",
        "DOLocationID": "236",
        "payment_type": "1",
        "RatecodeID": "1"
    }

    event = {
        "rawPath": "/predict",
        "requestContext": {
            "http": {
                "method": "POST"
            }
        },
        "body": json.dumps(payload)
    }

    response = handler(event)
    assert response["statusCode"] == 200

    body = json.loads(response["body"])
    assert body["high_tip_prediction"] in [0, 1]
    assert 0.0 <= body["probability"] <= 1.0
    assert body["latency_ms"] > 0.0
    assert body["serverless_engine"] == "AWS Lambda"


def test_lambda_invalid_fare_boundary():
    """Verifies that invalid fare amounts (< 2.50) trigger a 422 Validation Error."""
    bad_payload = {
        "trip_distance": 3.4,
        "fare_amount": 1.00,  # Illegal fare amount (< 2.50)
        "tolls_amount": 0.0,
        "passenger_count": 1.0,
        "PULocationID": "161",
        "DOLocationID": "236",
        "payment_type": "1",
        "RatecodeID": "1"
    }

    event = {
        "rawPath": "/predict",
        "requestContext": {
            "http": {
                "method": "POST"
            }
        },
        "body": json.dumps(bad_payload)
    }

    response = handler(event)
    assert response["statusCode"] == 422

    body = json.loads(response["body"])
    assert body["error"] == "Validation Error"

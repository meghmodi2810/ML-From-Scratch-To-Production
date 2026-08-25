# tests/test_loadtest.py
# =====================================================================
# NYC TAXI MLOPS - LOAD TEST PERSONA & GENERATOR UNIT TESTS
# =====================================================================


from src.serving.schemas import TaxiPredictionRequest
from tests.locustfile import (
    generate_airport_payload,
    generate_boundary_payload,
    generate_standard_payload,
)


def test_standard_payload_generator_conforms_to_schema():
    """Verifies that 100 randomly generated standard payloads strictly conform to TaxiPredictionRequest."""
    for _ in range(100):
        payload = generate_standard_payload()
        req = TaxiPredictionRequest(**payload)
        assert 0.8 <= req.trip_distance <= 6.5
        assert 2.5 <= req.fare_amount <= 500.0
        assert req.payment_type in ["1", "2"]
        assert req.RatecodeID == "1"


def test_airport_payload_generator_conforms_to_schema():
    """Verifies that 50 airport commuter payloads conform to schema with correct airport rates and zones."""
    for _ in range(50):
        payload = generate_airport_payload()
        req = TaxiPredictionRequest(**payload)
        assert req.trip_distance >= 12.0
        assert req.fare_amount >= 35.0
        assert req.DOLocationID in ["132", "138"]
        assert req.RatecodeID in ["1", "2"]


def test_boundary_payload_generator_conforms_to_schema():
    """Verifies that edge-case boundary payloads are valid and conform to schema."""
    for _ in range(50):
        payload = generate_boundary_payload()
        req = TaxiPredictionRequest(**payload)
        assert req.fare_amount >= 2.50
        assert req.trip_distance >= 0.1
        assert 1.0 <= req.passenger_count <= 6.0

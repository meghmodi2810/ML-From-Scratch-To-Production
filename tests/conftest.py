# tests/conftest.py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_valid_taxi_df():
    """Generates clean, schema-compliant TLC Yellow Taxi records."""
    return pd.DataFrame({
        "trip_distance": [1.5, 4.2, 0.8, 12.0],
        "fare_amount": [10.5, 24.0, 7.5, 65.0],
        "tolls_amount": [0.0, 6.55, np.nan, 12.50],
        "PULocationID": ["Zone_A", "Zone_B", "Zone_C", "Zone_A"],
        "payment_type": ["Credit_Card", "Cash", "Credit_Card", "Dispute"],
        "RatecodeID": ["Standard", "JFK", "Standard", "Negotiated"],
        "high_tip_indicator": [1, 0, 1, 0]
    })

@pytest.fixture
def sample_invalid_taxi_df():
    """Generates intentionally corrupted data violating multiple constraints."""
    return pd.DataFrame({
        "trip_distance": [-3.5, 4.2],                      # Negative distance violation
        "fare_amount": [10.5, -20.0],                      # Negative fare violation
        "tolls_amount": [0.0, 6.55],
        "PULocationID": ["Invalid_Zone", "Zone_B"],        # Unrecognized zone
        "payment_type": ["Crypto", "Cash"],                # Invalid category
        "RatecodeID": ["Standard", "Unknown"],             # Invalid ratecode
        "high_tip_indicator": [5, 0]                       # Non-binary target
    })

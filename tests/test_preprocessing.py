# tests/test_preprocessing.py
import pytest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from src.features.preprocess import build_preprocessor, clean_missing_tolls

# =====================================================================
# LAYER 2: UNIT TESTS FOR PREPROCESSING PIPELINE
# =====================================================================
def test_clean_missing_tolls_imputes_zero(sample_valid_taxi_df):
    """Unit Test: Verifies clean_missing_tolls replaces NaNs with 0.0."""
    cleaned = clean_missing_tolls(sample_valid_taxi_df)
    assert cleaned["tolls_amount"].isna().sum() == 0
    assert cleaned.loc[2, "tolls_amount"] == 0.0

def test_preprocessor_output_dimensions(sample_valid_taxi_df):
    """Unit Test: Verifies ColumnTransformer outputs a valid 2D array without NaNs."""
    preprocessor = build_preprocessor()
    X = sample_valid_taxi_df.drop("high_tip_indicator", axis=1)
    
    X_proc = preprocessor.fit_transform(X)
    
    assert isinstance(X_proc, np.ndarray)
    assert not np.isnan(X_proc).any()
    assert X_proc.shape[0] == len(sample_valid_taxi_df)

def test_preprocessor_idempotence(sample_valid_taxi_df):
    """Unit Test: Transforming the same input multiple times must yield identical results."""
    preprocessor = build_preprocessor()
    X = sample_valid_taxi_df.drop("high_tip_indicator", axis=1)
    
    preprocessor.fit(X)
    out1 = preprocessor.transform(X)
    out2 = preprocessor.transform(X)
    
    np.testing.assert_array_equal(out1, out2)

# =====================================================================
# LAYER 3: MODEL QUALITY & BEHAVIORAL TESTS
# =====================================================================
def test_model_overfits_batch_sanity_check(sample_valid_taxi_df):
    """Model Test: XGBoost must easily fit a small batch (Sanity Check)."""
    preprocessor = build_preprocessor()
    X = sample_valid_taxi_df.drop("high_tip_indicator", axis=1)
    y = sample_valid_taxi_df["high_tip_indicator"].values
    
    X_proc = preprocessor.fit_transform(X)
    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X_proc, y)
    
    preds = model.predict(X_proc)
    acc = accuracy_score(y, preds)
    assert acc >= 0.75, f"Sanity check failed: Training accuracy ({acc:.2f}) < 0.75"

def test_directional_prediction_logic():
    """Model Test (Directional): Longer distance/higher fare should yield higher tip probabilities."""
    preprocessor = build_preprocessor()
    
    train_df = pd.DataFrame({
        "trip_distance": [1.0, 2.0, 5.0, 10.0, 15.0] * 20,
        "fare_amount": [5.0, 10.0, 20.0, 40.0, 60.0] * 20,
        "tolls_amount": [0.0, 0.0, 6.55, 6.55, 12.5] * 20,
        "PULocationID": ["Zone_A"] * 100,
        "payment_type": ["Credit_Card"] * 100,
        "RatecodeID": ["Standard"] * 100,
        "high_tip_indicator": [0, 0, 1, 1, 1] * 20
    })
    
    X = train_df.drop("high_tip_indicator", axis=1)
    y = train_df["high_tip_indicator"].values
    
    X_proc = preprocessor.fit_transform(X)
    model = XGBClassifier(n_estimators=30, max_depth=3, random_state=42)
    model.fit(X_proc, y)
    
    # Compare short trip vs long trip probability
    short_trip = pd.DataFrame({
        "trip_distance": [0.5], "fare_amount": [4.0], "tolls_amount": [0.0],
        "PULocationID": ["Zone_A"], "payment_type": ["Credit_Card"], "RatecodeID": ["Standard"]
    })
    long_trip = pd.DataFrame({
        "trip_distance": [20.0], "fare_amount": [80.0], "tolls_amount": [12.5],
        "PULocationID": ["Zone_A"], "payment_type": ["Credit_Card"], "RatecodeID": ["Standard"]
    })
    
    p_short = model.predict_proba(preprocessor.transform(short_trip))[0, 1]
    p_long = model.predict_proba(preprocessor.transform(long_trip))[0, 1]
    
    assert p_long >= p_short, (
        f"Directional test failed: Short trip prob ({p_short:.3f}) > Long trip prob ({p_long:.3f})"
    )
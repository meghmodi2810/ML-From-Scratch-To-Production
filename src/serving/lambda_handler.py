# src/serving/lambda_handler.py
import json
import os
import time
from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import pandas as pd
from pydantic import ValidationError
from xgboost import XGBClassifier

from src.features.preprocess_serve import build_preprocessor
from src.serving.schemas import TaxiPredictionRequest

# =====================================================================
# GLOBAL WARMUP (Executed ONCE during Lambda Sandbox Cold Start)
# Initializing model and preprocessor outside the handler ensures
# subsequent warm invocations execute in ~10-15ms!
# =====================================================================
DATA_DIR = os.getenv("DATA_DIR", "/data" if os.path.exists("/data") else "data")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "TLC_Yellow_Taxi_Production_DAG")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}/{MODEL_STAGE}")

print(f"[Lambda Cold Start] Initializing Serverless Inference Engine in {os.getcwd()}...")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

PREPROCESSOR = build_preprocessor()
MODEL: Any = None
MODEL_VERSION = "serverless-baseline-v1"
DATA_SOURCE = "Baseline Reference Sample"

# Attempt to load model or fit lightweight baseline
try:
    MODEL = mlflow.pyfunc.load_model(MODEL_URI)
    MODEL_VERSION = MODEL_STAGE
    print(f"[Lambda Init] Successfully loaded model from MLflow: {MODEL_URI}")
except Exception as e:
    print(f"[Lambda Init INFO] MLflow load skipped ({e}). Initializing warm baseline model.")
    # Fit preprocessor & lightweight baseline model
    baseline_df = pd.DataFrame({
        "trip_distance": [1.0, 2.5, 5.0, 10.0, 15.0, 0.5, 3.2, 8.0] * 50,
        "fare_amount": [5.0, 12.0, 20.0, 45.0, 60.0, 4.0, 14.5, 35.0] * 50,
        "tolls_amount": [0.0, 0.0, 6.55, 6.55, 12.5, 0.0, 0.0, 6.55] * 50,
        "passenger_count": [1.0, 2.0, 1.0, 4.0, 1.0, 1.0, 2.0, 1.0] * 50,
        "PULocationID": ["161", "236", "132", "138", "161", "236", "132", "138"] * 50,
        "DOLocationID": ["236", "161", "138", "132", "236", "161", "138", "132"] * 50,
        "payment_type": ["1", "2", "1", "1", "2", "1", "2", "1"] * 50,
        "RatecodeID": ["1", "1", "2", "1", "1", "1", "2", "1"] * 50,
        "high_tip_indicator": [0, 1, 1, 1, 0, 0, 1, 1] * 50
    })
    X_baseline = baseline_df.drop("high_tip_indicator", axis=1)
    y_baseline = baseline_df["high_tip_indicator"].values

    PREPROCESSOR.fit(X_baseline)
    X_proc = PREPROCESSOR.transform(X_baseline)

    clf = XGBClassifier(n_estimators=10, max_depth=3, learning_rate=0.1, eval_metric="logloss", random_state=42)
    clf.fit(X_proc, y_baseline)
    MODEL = clf
    MODEL_VERSION = "serverless-warm-v1"
    print("[Lambda Init] Warm baseline inference engine ready!")


def _create_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Formats standardized API Gateway HTTP response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        },
        "body": json.dumps(body)
    }


def handler(event: Dict[str, Any], context: Any = None) -> Dict[str, Any]:
    """
    AWS Lambda entrypoint triggered by Amazon API Gateway.

    Supports:
      - GET /health  -> Healthcheck probe
      - POST /predict -> Real-time ML tip prediction
    """
    start_time = time.perf_counter()

    # Normalize HTTP method and path across API Gateway HTTP API v2 & REST API v1
    http_method = event.get("requestContext", {}).get("http", {}).get("method") or event.get("httpMethod", "POST")
    raw_path = event.get("rawPath") or event.get("path", "/predict")

    # 1. Healthcheck Route
    if "/health" in raw_path or http_method == "GET":
        return _create_response(200, {
            "status": "healthy",
            "model_loaded": MODEL is not None,
            "model_version": MODEL_VERSION,
            "data_source": DATA_SOURCE,
            "runtime": "AWS Lambda Serverless"
        })

    # 2. Prediction Route (POST)
    try:
        # Extract and parse raw JSON body
        raw_body = event.get("body", "{}")
        if isinstance(raw_body, str):
            payload_dict = json.loads(raw_body)
        else:
            payload_dict = raw_body

        # Validate against Pydantic schema
        validated_request = TaxiPredictionRequest(**payload_dict)

    except (json.JSONDecodeError, ValidationError) as e:
        return _create_response(422, {
            "error": "Validation Error",
            "detail": str(e)
        })

    try:
        # Feature Transformation & Model Scoring
        input_df = pd.DataFrame([validated_request.model_dump()])
        X_proc = PREPROCESSOR.transform(input_df)

        if hasattr(MODEL, "predict_proba"):
            probs = MODEL.predict_proba(X_proc)
            prob_high_tip = float(probs[0, 1])
            pred_class = int(prob_high_tip >= 0.5)
        else:
            raw_preds = MODEL.predict(X_proc)
            pred_class = int(raw_preds[0])
            prob_high_tip = float(pred_class)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return _create_response(200, {
            "high_tip_prediction": pred_class,
            "probability": round(prob_high_tip, 4),
            "model_version": MODEL_VERSION,
            "latency_ms": round(latency_ms, 2),
            "serverless_engine": "AWS Lambda"
        })

    except Exception as e:
        return _create_response(500, {
            "error": "Inference Execution Failed",
            "detail": str(e)
        })

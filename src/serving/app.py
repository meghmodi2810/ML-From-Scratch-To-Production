# src/serving/app.py
import glob
import os
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from xgboost import XGBClassifier

from src.features.preprocess_serve import build_preprocessor, prepare_real_taxi_dataset
from src.serving.metrics import (
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_TOTAL,
    TAXI_FEATURE_FARE_AMOUNT,
    TAXI_FEATURE_TRIP_DISTANCE,
    TAXI_MODEL_LOAD_STATUS,
    TAXI_PREDICTION_PROBABILITY,
    TAXI_PREDICTIONS_TOTAL,
)
from src.serving.schemas import (
    HealthCheckResponse,
    TaxiPredictionRequest,
    TaxiPredictionResponse,
)

# Configuration & Paths
DATA_DIR = os.getenv("DATA_DIR", "/data" if os.path.exists("/data") else "data")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "TLC_Yellow_Taxi_Production_DAG")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}/{MODEL_STAGE}")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads preprocessor and ML model once during server startup."""
    print(f"[Startup] Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"[Startup] Attempting to load model from: {MODEL_URI} ...")

    app.state.preprocessor = build_preprocessor()
    app.state.data_source = "None"

    # Locate real dataset in /data
    real_data_files = glob.glob(os.path.join(DATA_DIR, "*.parquet")) + glob.glob(os.path.join(DATA_DIR, "*.csv"))

    if real_data_files:
        train_file = real_data_files[0]
        app.state.data_source = train_file
        print(f"[Startup] Fitting preprocessor on real dataset: {train_file}")
        real_df = prepare_real_taxi_dataset(train_file, sample_size=20000)
        X_real = real_df.drop("high_tip_indicator", axis=1)
        y_real = real_df["high_tip_indicator"].values
        app.state.preprocessor.fit(X_real)
    else:
        print(f"[Startup WARNING] No dataset found in '{DATA_DIR}'. Fitting preprocessor on baseline reference sample.")
        baseline_df = pd.DataFrame({
            "trip_distance": [1.0, 2.5, 5.0, 10.0, 15.0, 0.5, 3.2, 8.0] * 100,
            "fare_amount": [5.0, 12.0, 20.0, 45.0, 60.0, 4.0, 14.5, 35.0] * 100,
            "tolls_amount": [0.0, 0.0, 6.55, 6.55, 12.5, 0.0, 0.0, 6.55] * 100,
            "passenger_count": [1.0, 2.0, 1.0, 4.0, 1.0, 1.0, 2.0, 1.0] * 100,
            "PULocationID": ["161", "236", "132", "138", "161", "236", "132", "138"] * 100,
            "DOLocationID": ["236", "161", "138", "132", "236", "161", "138", "132"] * 100,
            "payment_type": ["1", "2", "1", "1", "2", "1", "2", "1"] * 100,
            "RatecodeID": ["1", "1", "2", "1", "1", "1", "2", "1"] * 100,
            "high_tip_indicator": [0, 1, 1, 1, 0, 0, 1, 1] * 100
        })
        X_real = baseline_df.drop("high_tip_indicator", axis=1)
        y_real = baseline_df["high_tip_indicator"].values
        app.state.preprocessor.fit(X_real)
        app.state.data_source = "Baseline Reference Sample"

    # Load from MLflow or train on real data
    try:
        app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
        app.state.model_version = MODEL_STAGE
        print(f"[Startup] Loaded model from MLflow Registry ({MODEL_URI})")
    except Exception as e:
        print(f"[Startup INFO] MLflow registry load skipped ({e}). Training warm model on baseline data.")
        X_proc = app.state.preprocessor.transform(X_real)
        clf = XGBClassifier(n_estimators=10, max_depth=3, learning_rate=0.1, eval_metric="logloss", random_state=42)
        clf.fit(X_proc, y_real)
        app.state.model = clf
        app.state.model_version = "real-data-baseline-v1"

    # Update Prometheus Model Readiness Gauge
    TAXI_MODEL_LOAD_STATUS.labels(model_name=MODEL_NAME, model_version=app.state.model_version).set(1)

    yield

    print("[Shutdown] Cleaning up server resources.")
    TAXI_MODEL_LOAD_STATUS.labels(model_name=MODEL_NAME, model_version=getattr(app.state, "model_version", "unknown")).set(0)


app = FastAPI(
    title="NYC TLC Yellow Taxi Tip Prediction Service",
    description="Production-grade ML inference API with Prometheus observability & telemetry.",
    version="1.0.0",
    lifespan=lifespan
)


# =====================================================================
# PROMETHEUS TELEMETRY MIDDLEWARE
# =====================================================================
@app.middleware("http")
async def prometheus_telemetry_middleware(request: Request, call_next):
    """Measures request latency and increments request counts by status code and endpoint."""
    start_time = time.perf_counter()
    endpoint = request.url.path

    # Process request
    response = await call_next(request)

    # Compute execution duration in seconds
    duration = time.perf_counter() - start_time

    # Record Prometheus metrics (avoid explosion on non-standard dynamic paths)
    normalized_path = endpoint if endpoint in ["/predict", "/health", "/metrics", "/docs", "/openapi.json"] else "other"
    HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, endpoint=normalized_path).observe(duration)
    HTTP_REQUESTS_TOTAL.labels(method=request.method, endpoint=normalized_path, status_code=response.status_code).inc()

    return response


# =====================================================================
# API ROUTES
# =====================================================================
@app.get("/metrics", tags=["Observability"])
async def metrics():
    """Exposes Prometheus time-series metrics in standard exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", response_model=HealthCheckResponse, status_code=status.HTTP_200_OK, tags=["Monitoring"])
async def health_check():
    """Liveness/readiness probe verifying model readiness."""
    is_loaded = hasattr(app.state, "model") and app.state.model is not None
    return HealthCheckResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        data_source=str(getattr(app.state, "data_source", "None")),
        model_uri=MODEL_URI
    )


@app.post("/predict", response_model=TaxiPredictionResponse, status_code=status.HTTP_200_OK, tags=["Inference"])
async def predict(request: TaxiPredictionRequest):
    """Real-time scoring endpoint for taxi ride features with telemetry recording."""
    if not hasattr(app.state, "model") or app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready for inference."
        )

    start_time = time.perf_counter()
    input_df = pd.DataFrame([request.model_dump()])

    try:
        X_proc = app.state.preprocessor.transform(input_df)

        if hasattr(app.state.model, "predict_proba"):
            probs = app.state.model.predict_proba(X_proc)
            prob_high_tip = float(probs[0, 1])
            pred_class = int(prob_high_tip >= 0.5)
        else:
            raw_preds = app.state.model.predict(X_proc)
            pred_class = int(raw_preds[0])
            prob_high_tip = float(pred_class)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution failed: {str(e)}"
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # -----------------------------------------------------------------
    # RECORD DOMAIN & ML TELEMETRY METRICS
    # -----------------------------------------------------------------
    TAXI_PREDICTIONS_TOTAL.labels(prediction_class=str(pred_class), model_version=app.state.model_version).inc()
    TAXI_PREDICTION_PROBABILITY.labels(model_version=app.state.model_version).observe(prob_high_tip)
    TAXI_FEATURE_FARE_AMOUNT.observe(request.fare_amount)
    TAXI_FEATURE_TRIP_DISTANCE.observe(request.trip_distance)

    return TaxiPredictionResponse(
        high_tip_prediction=pred_class,
        probability=round(prob_high_tip, 4),
        model_version=app.state.model_version,
        latency_ms=round(latency_ms, 2)
    )

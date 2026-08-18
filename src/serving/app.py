# src/serving/app.py
import os
import glob
import time
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from xgboost import XGBClassifier

from src.features.preprocess_serve import build_preprocessor, prepare_real_taxi_dataset
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
        raise RuntimeError(f"No real dataset found in {DATA_DIR}! Please place Yellow Taxi files in /data.")

    # Load from MLflow or train on real data
    try:
        app.state.model = mlflow.pyfunc.load_model(MODEL_URI)
        app.state.model_version = MODEL_STAGE
        print(f"[Startup] Loaded model from MLflow Registry ({MODEL_URI})")
    except Exception as e:
        print(f"[Startup INFO] MLflow registry load skipped ({e}). Training warm model on real data.")
        X_proc = app.state.preprocessor.transform(X_real)
        clf = XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42)
        clf.fit(X_proc, y_real)
        app.state.model = clf
        app.state.model_version = "real-data-baseline-v1"

    yield

    print("[Shutdown] Cleaning up server resources.")

app = FastAPI(
    title="NYC TLC Yellow Taxi Tip Prediction Service",
    description="Real-time ML inference API running on real TLC Yellow Taxi data.",
    version="1.0.0",
    lifespan=lifespan
)

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
    """Real-time scoring endpoint for taxi ride features."""
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

    return TaxiPredictionResponse(
        high_tip_prediction=pred_class,
        probability=round(prob_high_tip, 4),
        model_version=app.state.model_version,
        latency_ms=round(latency_ms, 2)
    )
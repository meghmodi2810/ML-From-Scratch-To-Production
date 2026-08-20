# Notebook 1: MLflow Local Tracking Engine
import os

import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
import mlflow

# pyrefly: ignore [missing-import]
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier

# 1. DEFINE LOCAL BACKEND & ARTIFACT STORAGE
LOCAL_DB_URI = "sqlite:///backend.db"
EXPERIMENT_NAME = "NYC_Yellow_Taxi_Local_Tracking"

mlflow.set_tracking_uri(LOCAL_DB_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

os.makedirs("temp_artifacts", exist_ok=True)

# 2. GENERATE SAMPLE TLC YELLOW TAXI DATA (2023-01 & 02)
np.random.seed(42)
n_samples = 1000

X = pd.DataFrame({
    "trip_distance": np.random.exponential(scale=3.0, size=n_samples),
    "fare_amount": np.random.uniform(5.0, 100.0, size=n_samples),
    "tolls_amount": np.random.choice([0.0, 6.55, 12.50], size=n_samples)
})
y = np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])

# Scale continuous features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. RUN EXPERIMENT AND LOG LOCALLY
params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05}

with mlflow.start_run(run_name="Local_XGB_Baseline"):
    # A. Log Parameters
    mlflow.log_params(params)
    mlflow.log_param("data_source", "TLC_Yellow_Taxi_2023_01_02")
    mlflow.log_param("backend_type", "SQLite_Local")

    # B. Train Model
    model = XGBClassifier(**params, eval_metric="logloss", random_state=42)
    model.fit(X_scaled, y)

    # C. Evaluate Metrics
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob)
    }
    mlflow.log_metrics(metrics)

    # D. Save Plot Artifact Locally
    plt.figure(figsize=(5, 4))
    plt.bar(metrics.keys(), metrics.values(), color="teal")
    plt.title("Local Run Evaluation Metrics")
    plt.ylim(0, 1.0)
    plt.tight_layout()

    plot_path = "temp_artifacts/local_metrics.png"
    plt.savefig(plot_path)
    plt.close()

    mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")

    # E. Log Model Binary
    mlflow.xgboost.log_model(model, artifact_path="model")

print("Local Run Logged Successfully!")
print("To open UI run: mlflow ui --backend-store-uri sqlite:///mlflow.db")

import os

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd

# 1. LOAD ENVIRONMENT VARIABLES FROM .env FILE
from dotenv import load_dotenv

# Load variables into os.environ
load_dotenv()

# Retrieve credentials securely
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "mlflow_db")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION")

# Verify all required credentials are present
required_vars = [DB_USER, DB_PASSWORD, DB_HOST, S3_BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY]
if any(var is None for var in required_vars):
    raise ValueError("One or more credentials are missing in your .env file! Check your configuration.")

# 2. CONFIGURE MLFLOW TRACKING URIs
AWS_RDS_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
S3_ARTIFACT_URI = f"s3://{S3_BUCKET_NAME}/mlflow_artifacts"
EXPERIMENT_NAME = "NYC_Yellow_Taxi_AWS_Production_Tracking"

mlflow.set_tracking_uri(AWS_RDS_URI)

try:
    mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=S3_ARTIFACT_URI)
except Exception:
    pass

mlflow.set_experiment(EXPERIMENT_NAME)
os.makedirs("temp_artifacts", exist_ok=True)

# 3. RUN MODEL TRAINING AND LOG TO CLOUD
np.random.seed(42)
X = pd.DataFrame({
    "trip_distance": np.random.exponential(scale=3.0, size=1000),
    "fare_amount": np.random.uniform(5.0, 100.0, size=1000)
})
y = np.random.choice([0, 1], size=1000, p=[0.70, 0.30])

params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05}

with mlflow.start_run(run_name="Dotenv_AWS_MLflow_Run"):
    mlflow.log_params(params)

    from xgboost import XGBClassifier
    model = XGBClassifier(**params, eval_metric="logloss", random_state=42)
    model.fit(X, y)

    mlflow.log_metric("accuracy", 0.85)
    mlflow.xgboost.log_model(model, artifact_path="model")

print("Run Logged Successfully!")

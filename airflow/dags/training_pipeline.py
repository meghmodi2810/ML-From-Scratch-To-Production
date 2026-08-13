# Day 17: Apache Airflow Training Pipeline DAG
import os
import pendulum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from airflow.decorators import dag, task

# Default execution arguments applied across all DAG tasks
default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# Determine MLflow tracking URI pointing to root backend.db
if os.path.exists("/opt/airflow/backend.db"):
    MLFLOW_TRACKING_URI = "sqlite:////opt/airflow/backend.db"
else:
    _root_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend.db"))
    MLFLOW_TRACKING_URI = f"sqlite:///{_root_db.replace(os.sep, '/')}"

@dag(
    dag_id="tlc_yellow_taxi_training_pipeline",
    default_args=default_args,
    description="End-to-end ML training pipeline for TLC Yellow Taxi Tripdata",
    schedule_interval="@daily",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "xgboost", "tlc_taxi", "day17"],
)
def taxi_training_pipeline():

    # -----------------------------------------------------------------
    # TASK 1: DATA INGESTION
    # -----------------------------------------------------------------
    @task()
    def ingest_data() -> str:
        """Simulates ingestion of TLC Yellow Taxi Parquet data."""
        np.random.seed(42)
        n_samples = 1500
        
        data = {
            "trip_distance": np.random.exponential(scale=3.0, size=n_samples),
            "fare_amount": np.random.uniform(5.0, 100.0, size=n_samples),
            "tolls_amount": np.random.choice([0.0, 6.55, 12.50], size=n_samples, p=[0.8, 0.15, 0.05]),
            "PULocationID": np.random.choice(["Zone_A", "Zone_B", "Zone_C"], size=n_samples),
            "payment_type": np.random.choice(["Credit_Card", "Cash", "Dispute"], size=n_samples),
            "high_tip_indicator": np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
        }
        
        df = pd.DataFrame(data)
        os.makedirs("/tmp/airflow_ml", exist_ok=True)
        raw_path = "/tmp/airflow_ml/raw_taxi_data.parquet"
        df.to_parquet(raw_path, index=False)
        
        print(f"[Ingest] Ingested {len(df)} records -> {raw_path}")
        return raw_path

    # -----------------------------------------------------------------
    # TASK 2: DATA PREPROCESSING
    # -----------------------------------------------------------------
    @task()
    def preprocess_data(raw_data_path: str) -> str:
        """Cleans data and handles missing values idempotently."""
        df = pd.read_parquet(raw_data_path)
        
        # Handle missing values
        df["tolls_amount"] = df["tolls_amount"].fillna(0.0)
        
        proc_path = "/tmp/airflow_ml/processed_taxi_data.parquet"
        df.to_parquet(proc_path, index=False)
        
        print(f"[Preprocess] Processed {len(df)} records -> {proc_path}")
        return proc_path

    # -----------------------------------------------------------------
    # TASK 3: MODEL TRAINING
    # -----------------------------------------------------------------
    @task()
    def train_model(processed_data_path: str) -> dict:
        """Trains XGBoost model on preprocessed data and logs to MLflow."""
        import mlflow
        import mlflow.xgboost
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        
        df = pd.read_parquet(processed_data_path)
        
        X = df[["trip_distance", "fare_amount", "tolls_amount"]]
        y = df["high_tip_indicator"].values
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05, "eval_metric": "logloss"}
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Airflow_TLC_Taxi_Pipeline")
        
        with mlflow.start_run(run_name="Airflow_DAG_Run") as run:
            mlflow.log_params(params)
            
            model = XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            model_artifact_path = f"/tmp/airflow_ml/model_{run.info.run_id}.bin"
            model.save_model(model_artifact_path)
            
            mlflow.xgboost.log_model(model, artifact_path="model")
            
            return {
                "run_id": run.info.run_id,
                "model_path": model_artifact_path,
                "processed_data_path": processed_data_path
            }

    # -----------------------------------------------------------------
    # TASK 4: MODEL EVALUATION
    # -----------------------------------------------------------------
    @task()
    def evaluate_model(train_output: dict) -> dict:
        """Evaluates trained model metrics and updates MLflow run."""
        import mlflow
        from xgboost import XGBClassifier
        from sklearn.metrics import f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        
        df = pd.read_parquet(train_output["processed_data_path"])
        X = df[["trip_distance", "fare_amount", "tolls_amount"]]
        y = df["high_tip_indicator"].values
        
        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBClassifier()
        model.load_model(train_output["model_path"])
        
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        f1 = float(f1_score(y_val, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_val, y_prob))
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_id=train_output["run_id"]):
            mlflow.log_metrics({"eval_f1_score": f1, "eval_roc_auc": auc})
            
        print(f"[Evaluate] Run ID: {train_output['run_id']} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        return {
            "run_id": train_output["run_id"],
            "f1_score": f1,
            "roc_auc": auc
        }

    # -----------------------------------------------------------------
    # TASK 5: MODEL REGISTRATION
    # -----------------------------------------------------------------
    @task()
    def register_model(eval_output: dict):
        """Registers candidate model in MLflow Registry if it passes Quality Gate."""
        import mlflow
        
        min_f1_threshold = 0.50
        
        if eval_output["f1_score"] >= min_f1_threshold:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"runs:/{eval_output['run_id']}/model"
            mlflow.register_model(model_uri, "TLC_Yellow_Taxi_Production_DAG")
            print(f"[Register] Model {eval_output['run_id']} registered successfully! (F1: {eval_output['f1_score']:.4f})")
        else:
            print(f"[Register] Model F1 ({eval_output['f1_score']:.4f}) below threshold ({min_f1_threshold}). Skipped.")

    # -----------------------------------------------------------------
    # PIPELINE EXECUTION & TASK DEPENDENCY CHAINING
    # -----------------------------------------------------------------
    raw_path = ingest_data()
    processed_path = preprocess_data(raw_path)
    train_res = train_model(processed_path)
    eval_res = evaluate_model(train_res)
    register_model(eval_res)

# Instantiate the DAG
dag_instance = taxi_training_pipeline()
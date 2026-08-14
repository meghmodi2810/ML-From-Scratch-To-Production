# Day 18: Complete Robust Airflow Training Pipeline
# Location: dags/training_pipeline_with_CT.py

import os
import pendulum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.sensors.filesystem import FileSensor
from airflow.exceptions import AirflowFailException

# =====================================================================
# 1. FAILURE ALERT CALLBACK
# =====================================================================
def on_task_failure_callback(context):
    """Callback function triggered when a task exhausts all retries."""
    task_id = context.get("task_instance").task_id
    dag_id = context.get("task_instance").dag_id
    execution_date = context.get("execution_date")
    exception = context.get("exception")
    
    print("\n" + "!" * 75)
    print(f"ALERT: TASK PERMANENTLY FAILED IN PIPELINE!")
    print(f"DAG              : {dag_id}")
    print(f"Task             : {task_id}")
    print(f"Execution Date   : {execution_date}")
    print(f"Exception        : {exception}")
    print("!" * 75 + "\n")

# =====================================================================
# 2. DEFAULT ARGUMENTS & RETRY POLICY
# =====================================================================
default_args = {
    "owner": "mlops_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,                               # Retry failed tasks up to 3 times
    "retry_delay": timedelta(seconds=15),       # Initial retry delay
    "retry_exponential_backoff": True,          # Delay doubles on each retry (15s -> 30s -> 60s)
    "max_retry_delay": timedelta(minutes=5),    # Cap maximum wait time
    "on_failure_callback": on_task_failure_callback,
}

# Determine MLflow tracking URI pointing to root backend.db
if os.path.exists("/opt/airflow/backend.db"):
    MLFLOW_TRACKING_URI = "sqlite:////opt/airflow/backend.db?timeout=60"
else:
    _root_db = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend.db"))
    MLFLOW_TRACKING_URI = f"sqlite:///{_root_db.replace(os.sep, '/')}?timeout=60"

# =====================================================================
# 3. DAG DECLARATION
# =====================================================================
@dag(
    dag_id="robust_tlc_yellow_taxi_ct_pipeline",
    default_args=default_args,
    description="Self-healing CT pipeline with FileSensors, Retries, and Model Promotion Gates",
    schedule_interval="@weekly",                # Scheduled Continuous Training (CT)
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["mlops", "robustness", "continuous_training", "day18"],
)
def robust_taxi_ct_pipeline():

    # -----------------------------------------------------------------
    # TASK 1: DATA INGESTION (Simulates Batch Generation)
    # -----------------------------------------------------------------
    @task()
    def ingest_weekly_batch() -> str:
        """Simulates weekly ingestion of NYC TLC Yellow Taxi Parquet data."""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        n_samples = 1500
        
        data = {
            "trip_distance": np.random.exponential(scale=3.2, size=n_samples),
            "fare_amount": np.random.uniform(5.0, 110.0, size=n_samples),
            "tolls_amount": np.random.choice([0.0, 6.55, 12.50], size=n_samples, p=[0.75, 0.20, 0.05]),
            "PULocationID": np.random.choice(["Zone_A", "Zone_B", "Zone_C"], size=n_samples),
            "payment_type": np.random.choice(["Credit_Card", "Cash", "Dispute"], size=n_samples),
            "high_tip_indicator": np.random.choice([0, 1], size=n_samples, p=[0.68, 0.32])
        }
        
        df = pd.DataFrame(data)
        os.makedirs("/tmp/airflow_ml/landing", exist_ok=True)
        raw_path = "/tmp/airflow_ml/landing/weekly_taxi_data.parquet"
        df.to_parquet(raw_path, index=False)
        
        print(f"[Ingest] Created weekly batch: {len(df)} records -> {raw_path}")
        return raw_path

    # -----------------------------------------------------------------
    # TASK 2: FILE SENSOR (Waits for Landing File)
    # -----------------------------------------------------------------
    wait_for_raw_file = FileSensor(
        task_id="wait_for_raw_parquet_file",
        filepath="/tmp/airflow_ml/landing/weekly_taxi_data.parquet",
        poke_interval=10,        # Poll every 10 seconds
        timeout=120,             # Fail after 2 minutes if missing
        mode="reschedule"        # Release Airflow worker slot between checks
    )

    # -----------------------------------------------------------------
    # TASK 3: PREPROCESSING & SCHEMA VALIDATION
    # -----------------------------------------------------------------
    @task()
    def preprocess_and_validate(raw_data_path: str) -> str:
        """Cleans data and asserts schema integrity before training."""
        if not os.path.exists(raw_data_path):
            raise AirflowFailException(f"Raw data file missing at path: {raw_data_path}")
            
        df = pd.read_parquet(raw_data_path)
        
        # Schema Assertion Check
        required_cols = ["trip_distance", "fare_amount", "tolls_amount", "high_tip_indicator"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise AirflowFailException(f"Data Schema Validation Failed! Missing columns: {missing_cols}")

        df["tolls_amount"] = df["tolls_amount"].fillna(0.0)
        
        proc_path = "/tmp/airflow_ml/processed_weekly_taxi.parquet"
        df.to_parquet(proc_path, index=False)
        
        print(f"[Preprocess] Successfully processed {len(df)} records -> {proc_path}")
        return proc_path

    # -----------------------------------------------------------------
    # TASK 4: RETRAIN CANDIDATE MODEL
    # -----------------------------------------------------------------
    @task()
    def train_candidate_model(processed_data_path: str) -> dict:
        """Trains new candidate XGBoost model on newly arrived batch."""
        import mlflow
        import mlflow.xgboost
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        
        df = pd.read_parquet(processed_data_path)
        
        X = df[["trip_distance", "fare_amount", "tolls_amount"]]
        y = df["high_tip_indicator"].values
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        params = {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.08, "eval_metric": "logloss"}
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment("Robust_CT_Pipeline")
        
        with mlflow.start_run(run_name="CT_Candidate_Run") as run:
            mlflow.log_params(params)
            
            model = XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            
            model_artifact_path = f"/tmp/airflow_ml/candidate_model_{run.info.run_id}.bin"
            model.save_model(model_artifact_path)
            
            mlflow.xgboost.log_model(model, artifact_path="model")
            
            return {
                "candidate_run_id": run.info.run_id,
                "model_path": model_artifact_path,
                "processed_data_path": processed_data_path
            }

    # -----------------------------------------------------------------
    # TASK 5: EVALUATE CANDIDATE METRICS
    # -----------------------------------------------------------------
    @task()
    def evaluate_candidate(train_output: dict) -> dict:
        """Evaluates candidate model performance and logs evaluation metrics."""
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
        with mlflow.start_run(run_id=train_output["candidate_run_id"]):
            mlflow.log_metrics({"candidate_f1": f1, "candidate_auc": auc})
            
        print(f"[Evaluate] Candidate Run: {train_output['candidate_run_id']} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        return {
            "candidate_run_id": train_output["candidate_run_id"],
            "candidate_f1": f1,
            "candidate_auc": auc
        }

    # -----------------------------------------------------------------
    # TASK 6: CONTINUOUS TRAINING (CT) PROMOTION GATE
    # -----------------------------------------------------------------
    @task()
    def ct_promotion_gate(eval_output: dict):
        """Automated Quality Gate comparing Candidate vs Production Model."""
        import mlflow
        from mlflow.tracking import MlflowClient
        
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        model_name = "TLC_Yellow_Taxi_Production_DAG"
        min_absolute_f1_threshold = 0.55
        candidate_f1 = eval_output["candidate_f1"]
        candidate_run_id = eval_output["candidate_run_id"]
        
        # Check current production model baseline
        prod_f1_baseline = 0.50
        try:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            if prod_versions:
                prod_run = client.get_run(prod_versions[0].run_id)
                prod_f1_baseline = prod_run.data.metrics.get("candidate_f1", 0.50)
                print(f"[Gate] Existing Production Model -> Version {prod_versions[0].version} (F1: {prod_f1_baseline:.4f})")
        except Exception:
            print("[Gate] No existing Production model found. Using initial baseline threshold.")

        print(f"[Gate] Comparing Candidate F1 ({candidate_f1:.4f}) vs Baseline F1 ({prod_f1_baseline:.4f})")
        
        # PROMOTION DECISION
        if candidate_f1 >= max(min_absolute_f1_threshold, prod_f1_baseline):
            model_uri = f"runs:/{candidate_run_id}/model"
            reg_model = mlflow.register_model(model_uri, model_name)
            
            client.transition_model_version_stage(
                name=model_name,
                version=reg_model.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"[GATE PASSED] Candidate Model {candidate_run_id} promoted to PRODUCTION (Version {reg_model.version})! 🚀")
        else:
            print(f"[GATE REJECTED] Candidate Model F1 ({candidate_f1:.4f}) did not beat Production baseline ({prod_f1_baseline:.4f}). Candidate archived.")

    # -----------------------------------------------------------------
    # PIPELINE DEPENDENCY EXECUTION CHAIN
    # -----------------------------------------------------------------
    raw_file = ingest_weekly_batch()
    
    # Preprocessing executes after raw_file ingestion and FileSensor validation
    proc_file = preprocess_and_validate(raw_file)
    
    # Task flow dependencies with FileSensor
    raw_file >> wait_for_raw_file >> proc_file
    
    # Remaining ML pipeline steps execute sequentially
    train_res = train_candidate_model(proc_file)
    eval_res = evaluate_candidate(train_res)
    ct_promotion_gate(eval_res)

# Instantiate the DAG
dag_instance = robust_taxi_ct_pipeline()
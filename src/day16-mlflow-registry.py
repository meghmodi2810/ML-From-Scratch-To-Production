# Day 16: Optuna Hyperparameter Tuning & MLflow Model Registry
import logging
import os
import warnings

# Disable MLflow artifact progress bars and suppress warnings
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)

import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

# Silence Optuna logging output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 1. CONFIGURE MLFLOW TRACKING & REGISTRY
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "backend.db")
MLFLOW_DB_PATH = "sqlite:///backend.db"
EXPERIMENT_NAME = "NYC_Yellow_Taxi_Optuna_Tuning"
REGISTERED_MODEL_NAME = "TLC_Taxi_Tip_Classifier"
PROMOTION_THRESHOLD_F1 = 0.30  # Quality gate threshold for Production promotion
N_TRIALS = 5                   # Number of Optuna trials for efficient execution

mlflow.set_tracking_uri(MLFLOW_DB_PATH)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

print("=" * 75)
print("DAY 16: OPTUNA TUNING & MODEL REGISTRY")
print(f"Tracking URI : {MLFLOW_DB_PATH}")
print(f"Experiment   : {EXPERIMENT_NAME}")
print("=" * 75 + "\n")

# =====================================================================
# 2. GENERATE TLC YELLOW TAXI DATASET (2023-01 & 2023-02)
# =====================================================================
np.random.seed(42)
n_samples = 1500

data = {
    "trip_distance": np.random.exponential(scale=3.0, size=n_samples),
    "fare_amount": np.random.uniform(5.0, 100.0, size=n_samples),
    "tolls_amount": np.random.choice([0.0, 6.55, 12.50], size=n_samples, p=[0.8, 0.15, 0.05]),
    "PULocationID": np.random.choice(["Zone_A", "Zone_B", "Zone_C"], size=n_samples, p=[0.5, 0.3, 0.2]),
    "payment_type": np.random.choice(["Credit_Card", "Cash", "Dispute"], size=n_samples, p=[0.7, 0.25, 0.05]),
    "RatecodeID": np.random.choice(["Standard", "JFK", "Negotiated"], size=n_samples),
    "high_tip_indicator": np.random.choice([0, 1], size=n_samples, p=[0.70, 0.30])
}

df = pd.DataFrame(data)
mask = np.random.rand(len(df)) < 0.05
df.loc[mask, "tolls_amount"] = np.nan

X = df.drop("high_tip_indicator", axis=1)
y = df["high_tip_indicator"].values

num_cols = ["trip_distance", "fare_amount", "tolls_amount"]
ohe_cols = ["PULocationID", "payment_type"]
ordinal_cols = ["RatecodeID"]
ratecode_order = [["Standard", "JFK", "Negotiated"]]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("ohe", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", sparse_output=False))]), ohe_cols),
        ("ord", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder(categories=ratecode_order))]), ordinal_cols),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =====================================================================
# 3. DEFINE OPTUNA OBJECTIVE FUNCTION
# =====================================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 30, 200, step=10),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "eval_metric": "logloss",
        "random_state": 42
    }

    with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number+1:02d}"):
        mlflow.log_params(params)
        mlflow.log_param("trial_number", trial.number + 1)

        f1s, aucs = [], []

        for train_idx, val_idx in cv.split(X, y):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_val, y_val = X.iloc[val_idx], y[val_idx]

            X_tr_proc = preprocessor.fit_transform(X_train)
            X_val_proc = preprocessor.transform(X_val)

            smote = SMOTE(random_state=42)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_proc, y_train)

            model = XGBClassifier(**params)
            model.fit(X_tr_res, y_tr_res)

            y_pred = model.predict(X_val_proc)
            y_prob = model.predict_proba(X_val_proc)[:, 1]

            f1s.append(f1_score(y_val, y_pred, zero_division=0))
            aucs.append(roc_auc_score(y_val, y_prob))

        mean_f1 = float(np.mean(f1s))
        mean_auc = float(np.mean(aucs))

        mlflow.log_metric("cv_f1_score", mean_f1)
        mlflow.log_metric("cv_roc_auc", mean_auc)

        # Log trained model for trial silently
        mlflow.xgboost.log_model(model, artifact_path="model", input_example=X_tr_res[:2])

        print(f"  -> Trial {trial.number + 1:02d}/{N_TRIALS} completed | CV F1: {mean_f1:.4f} | ROC-AUC: {mean_auc:.4f}")

    return mean_f1

# =====================================================================
# 4. EXECUTE OPTUNA HYPERPARAMETER STUDY
# =====================================================================
print(f"Starting Optuna {N_TRIALS}-trial optimization study...")
with mlflow.start_run(run_name="Optuna_Parent_Study") as parent_run:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    best_trial = study.best_trial
    print("\n" + "=" * 75)
    print("OPTUNA STUDY COMPLETED!")
    print(f"Best Trial Number : Trial {best_trial.number + 1}")
    print(f"Best CV F1-Score   : {best_trial.value:.4f}")
    print(f"Best Parameters    : {best_trial.params}")
    print("=" * 75 + "\n")

    mlflow.log_params(best_trial.params)
    mlflow.log_metric("best_cv_f1_score", best_trial.value)

# =====================================================================
# 5. FIND TOP RUN, REGISTER MODEL & TRANSITION TO PRODUCTION
# =====================================================================
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.cv_f1_score DESC"],
    max_results=1
)

top_run = runs[0]
top_run_id = top_run.info.run_id
top_f1_score = top_run.data.metrics["cv_f1_score"]

print(f"Top Run ID : {top_run_id}")
print(f"Top CV F1  : {top_f1_score:.4f}")

# A. Register the Top Model
model_uri = f"runs:/{top_run_id}/model"
registered_model = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
model_version = registered_model.version

print(f"Successfully registered model '{REGISTERED_MODEL_NAME}' -> Version {model_version}")

# B. Automated Quality Gate Promotion
if top_f1_score >= PROMOTION_THRESHOLD_F1:
    target_stage = "Production"
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"QUALITY GATE PASSED (F1 {top_f1_score:.4f} >= {PROMOTION_THRESHOLD_F1})")
    print(f"Model '{REGISTERED_MODEL_NAME}' Version {model_version} promoted to 'Production' stage!")
else:
    target_stage = "Staging"
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_version,
        stage="Staging"
    )
    print(f"Quality gate warning (F1 {top_f1_score:.4f} < {PROMOTION_THRESHOLD_F1}). Promoted to 'Staging'.")

# =====================================================================
# 6. VERIFY UNIFORM SERVING INTERFACE (mlflow.pyfunc)
# =====================================================================
print("\n--- Verifying Uniform Serving Interface (mlflow.pyfunc) ---")
registered_model_uri = f"models:/{REGISTERED_MODEL_NAME}/{target_stage}"
pyfunc_model = mlflow.pyfunc.load_model(registered_model_uri)

# Generate sample features for inference test (use transform, not fit_transform)
X_dummy_proc = preprocessor.transform(X.head(5))
sample_preds = pyfunc_model.predict(X_dummy_proc)

print(f"Loaded model from URI : {registered_model_uri}")
print(f"Sample Predictions    : {sample_preds}")
print("=" * 75)

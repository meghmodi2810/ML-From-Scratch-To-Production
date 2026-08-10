import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

# Create directory for manual run plot artifacts
os.makedirs("../artifacts/day14", exist_ok=True)

# =====================================================================
# 1. LOAD TELCO CHURN DATASET & RUN STATISTICAL EDA
# =====================================================================
np.random.seed(42)
n_samples = 1500

data = {
    "tenure": np.random.randint(1, 72, size=n_samples),
    "MonthlyCharges": np.random.uniform(18.25, 118.75, size=n_samples),
    "TotalCharges": np.random.uniform(18.25, 8500.0, size=n_samples),
    "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.25, 0.20]),
    "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.4, 0.45, 0.15]),
    "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], size=n_samples),
    "PaperlessBilling": np.random.choice(["Yes", "No"], size=n_samples),
    "Churn": np.random.choice([0, 1], size=n_samples, p=[0.73, 0.27])
}

df = pd.DataFrame(data)

# Introduce 5% missing values in TotalCharges to simulate messy real-world data
mask = np.random.rand(len(df)) < 0.05
df.loc[mask, "TotalCharges"] = np.nan

print("=" * 75)
print("DAY 14: STATISTICAL EDA ON TELCO CHURN DATASET")
print("=" * 75)


def make_classifier(params):
    if XGBClassifier is not None:
        return XGBClassifier(**params, eval_metric="logloss", random_state=42)

    print("XGBoost could not be loaded; using HistGradientBoostingClassifier fallback.")
    return HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        max_iter=params["n_estimators"],
        random_state=42,
    )

# A. Point-Biserial Correlation for Continuous Features
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
print("\n--- Point-Biserial Correlation (Continuous Features vs Churn) ---")
for col in num_cols:
    clean_series = df[[col, "Churn"]].dropna()
    corr, p_val = stats.pointbiserialr(clean_series["Churn"], clean_series[col])
    print(f"Feature: {col:<15} | Correlation r_pb: {corr:+.4f} | p-value: {p_val:.4e} | Significant: {p_val < 0.05}")

# B. Chi-Square Test of Independence for Categorical Features
cat_cols = ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]
print("\n--- Chi-Square Test of Independence (Categorical Features vs Churn) ---")
for col in cat_cols:
    contingency_table = pd.crosstab(df[col], df["Churn"])
    chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)
    print(f"Feature: {col:<15} | Chi2 Score: {chi2:8.4f} | p-value: {p_val:.4e} | Significant: {p_val < 0.05}")

print("=" * 75 + "\n")

# =====================================================================
# 2. DEFINE LEAKAGE-FREE PREPROCESSING PIPELINE
# =====================================================================
ohe_cols = ["InternetService", "PaymentMethod", "PaperlessBilling"]
ordinal_cols = ["Contract"]
contract_order = [["Month-to-month", "One year", "Two year"]]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("ohe", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", sparse_output=False))]), ohe_cols),
        ("ord", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder(categories=contract_order))]), ordinal_cols),
    ]
)

X = df.drop("Churn", axis=1)
y = df["Churn"].values

# =====================================================================
# 3. MANUAL EXPERIMENT TRACKING: LOGGING 15 RUNS TO results.csv
# =====================================================================
results_csv_path = "results.csv"

# Hyperparameter search grid (15 trials)
hyperparam_grid = [
    {"n_estimators": 30,  "max_depth": 3, "learning_rate": 0.01, "subsample": 0.8},
    {"n_estimators": 50,  "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8},
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.10, "subsample": 0.8},
    {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.7},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.7},
    {"n_estimators": 50,  "max_depth": 5, "learning_rate": 0.20, "subsample": 1.0},
    {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.9},
    {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.10, "subsample": 0.8},
    {"n_estimators": 80,  "max_depth": 4, "learning_rate": 0.15, "subsample": 0.85},
    {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.95},
    {"n_estimators": 60,  "max_depth": 7, "learning_rate": 0.25, "subsample": 0.6},
    {"n_estimators": 180, "max_depth": 5, "learning_rate": 0.08, "subsample": 0.75},
    {"n_estimators": 90,  "max_depth": 4, "learning_rate": 0.12, "subsample": 0.8},
    {"n_estimators": 110, "max_depth": 6, "learning_rate": 0.04, "subsample": 0.9},
    {"n_estimators": 160, "max_depth": 3, "learning_rate": 0.07, "subsample": 0.85},
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
manual_logs = []

print("Starting 15 Manual Experiment Runs (Logging to results.csv)...")

for run_id, params in enumerate(hyperparam_grid, start=1):
    accs, precs, recs, f1s, aucs = [], [], [], [], []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        # Preprocessing & SMOTE strictly on training fold
        X_tr_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        
        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_proc, y_train)
        
        # Train gradient boosting model
        model = make_classifier(params)
        model.fit(X_tr_res, y_tr_res)
        
        y_pred = model.predict(X_val_proc)
        y_prob = model.predict_proba(X_val_proc)[:, 1]
        
        accs.append(accuracy_score(y_val, y_pred))
        precs.append(precision_score(y_val, y_pred, zero_division=0))
        recs.append(recall_score(y_val, y_pred, zero_division=0))
        f1s.append(f1_score(y_val, y_pred, zero_division=0))
        aucs.append(roc_auc_score(y_val, y_prob))

    # Record trial metadata manually
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    artifact_path = f"../artifacts/day14/xgb_run_{run_id}.png"

    # Save artifact plot manually
    plt.figure(figsize=(5, 4))
    plt.bar(["Acc", "F1", "AUC"], [np.mean(accs), np.mean(f1s), np.mean(aucs)], color="teal")
    plt.title(f"Run {run_id} Metrics")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(artifact_path)
    plt.close()

    manual_logs.append({
        "run_id": f"RUN_{run_id:02d}",
        "timestamp": timestamp,
        "n_estimators": params["n_estimators"],
        "max_depth": params["max_depth"],
        "learning_rate": params["learning_rate"],
        "subsample": params["subsample"],
        "cv_accuracy": round(float(np.mean(accs)), 4),
        "cv_precision": round(float(np.mean(precs)), 4),
        "cv_recall": round(float(np.mean(recs)), 4),
        "cv_f1_score": round(float(np.mean(f1s)), 4),
        "cv_roc_auc": round(float(np.mean(aucs)), 4),
        "artifact_plot_path": artifact_path
    })

# Write to results.csv
results_df = pd.DataFrame(manual_logs)
results_df.to_csv(results_csv_path, index=False)

print("\n=" * 75)
print("MANUAL TRACKING RESULTS LOGGED TO results.csv")
print("=" * 75)
print(results_df[["run_id", "n_estimators", "max_depth", "learning_rate", "cv_f1_score", "cv_roc_auc"]].head(10).to_string(index=False))
print("=" * 75 + "\n")

# =====================================================================
# 4. VISUALIZE MANUAL TRACKING DEBT / FRICTION & EDA
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Day 14: Manual Experiment Tracking & Statistical EDA", fontsize=14, fontweight="bold")

# Plot 1: Manual Run Comparison (F1 Score vs ROC-AUC)
axes[0].plot(results_df["run_id"], results_df["cv_f1_score"], marker="o", color="crimson", label="CV F1-Score")
axes[0].plot(results_df["run_id"], results_df["cv_roc_auc"], marker="s", color="navy", label="CV ROC-AUC")
axes[0].set_title("1. Manual Experiment Log Comparison (results.csv)", fontsize=11)
axes[0].set_xlabel("Run ID")
axes[0].set_ylabel("Score")
axes[0].tick_params(axis="x", rotation=45)
axes[0].legend()
axes[0].grid(True, linestyle=":", alpha=0.6)

# Plot 2: Chi-Square Test Feature Signal Visualization
chi2_scores = []
for col in cat_cols:
    contingency = pd.crosstab(df[col], df["Churn"])
    chi2, _, _, _ = stats.chi2_contingency(contingency)
    chi2_scores.append(chi2)

axes[1].barh(cat_cols, chi2_scores, color="teal", alpha=0.8)
axes[1].set_title("2. Categorical Feature Signal (Chi-Square Independence Test)", fontsize=11)
axes[1].set_xlabel("Chi-Square Statistic (χ²)")
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
summary_artifact_path = "../artifacts/day14/day14_summary.png"
plt.savefig(summary_artifact_path, bbox_inches="tight")
plt.close(fig)

print(f"Summary plot saved to {summary_artifact_path}")
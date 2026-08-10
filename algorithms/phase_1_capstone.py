# Day 13: Phase 1 Capstone — Telco Customer Churn (Custom Workspace Algorithms)
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in path to import from 'algorithms' and 'src'
sys.path.append(os.path.abspath(".."))

# 1. IMPORT YOUR CUSTOM FROM-SCRATCH ALGORITHMS FROM `algorithms/`
from algorithms.logistic_regression import LogisticRegressionScratch
from algorithms.decision_tree import DecisionTreeScratch
from algorithms.svm import LinearSVMScratch
from algorithms.random_forest import RandomForestClassifierScratch
from algorithms.adaBoost import AdaBoostClassifierScratch
from algorithms.XGBoost import XGBoostClassifierScratch

# 2. IMPORT SKLEARN PREPROCESSING & EVALUATION TOOLS
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE


# Suppress a known scikit-learn deprecation warning emitted by the current
# imbalanced-learn / scikit-learn version combination during fit_resample.
warnings.filterwarnings(
    "ignore",
    module=r"sklearn\.base",
    category=FutureWarning,
)


# =====================================================================
# 1. GENERATE SYNTHETIC TELCO CHURN DATASET
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

# Introduce 5% missing values in TotalCharges to simulate real-world data issues
mask = np.random.rand(len(df)) < 0.05
df.loc[mask, "TotalCharges"] = np.nan

X = df.drop("Churn", axis=1)
y = df["Churn"].values

print("=" * 65)
print("MODULE 1 CAPSTONE: TELCO CUSTOMER CHURN DATASET")
print("=" * 65)
print(f"Dataset Shape       : {df.shape}")
print(f"Churn Class Ratio   : {np.bincount(y) / len(y)}")
print(f"Missing Values Count: {df.isnull().sum().to_dict()}")
print("=" * 65 + "\n")


# =====================================================================
# 2. DEFINE LEAKAGE-FREE PREPROCESSING PIPELINE
# =====================================================================
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
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


# =====================================================================
# 3. BENCHMARK CUSTOM WORKSPACE ALGORITHMS ON STRATIFIED CV
# =====================================================================
scratch_models = {
    "Logistic Regression (Scratch)": LogisticRegressionScratch(learning_rate=0.05, epochs=500),
    "Decision Tree (Scratch)"      : DecisionTreeScratch(max_depth=6),
    "Support Vector Machine (Scratch)": LinearSVMScratch(C=1.0, lr=0.001, epochs=300),
    "Random Forest (Scratch)"      : RandomForestClassifierScratch(n_estimators=30, max_depth=6),
    "AdaBoost (Scratch)"           : AdaBoostClassifierScratch(n_estimators=30),
    "XGBoost (Scratch)"            : XGBoostClassifierScratch(n_estimators=20, learning_rate=0.1, max_depth=3)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
leaderboard_results = []

for name, model in scratch_models.items():
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    
    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        # 1. Transform Features (Strictly Prevents Data Leakage)
        X_tr_proc = preprocessor.fit_transform(X_train)
        X_val_proc = preprocessor.transform(X_val)
        
        # 2. Oversample Training Fold ONLY using SMOTE
        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_proc, y_train)
        
        # 3. Train From-Scratch Workspace Model & Evaluate
        model.fit(X_tr_res, y_tr_res)
        y_pred = model.predict(X_val_proc)
        
        acc_list.append(accuracy_score(y_val, y_pred))
        prec_list.append(precision_score(y_val, y_pred, zero_division=0))
        rec_list.append(recall_score(y_val, y_pred, zero_division=0))
        f1_list.append(f1_score(y_val, y_pred, zero_division=0))

    leaderboard_results.append({
        "Algorithm": name,
        "Accuracy": np.mean(acc_list),
        "Precision": np.mean(prec_list),
        "Recall": np.mean(rec_list),
        "F1-Score": np.mean(f1_list)
    })

leaderboard_df = pd.DataFrame(leaderboard_results).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

print("=" * 75)
print("MODULE 1 CAPSTONE: CUSTOM WORKSPACE ALGORITHMS BENCHMARK LEADERBOARD")
print("=" * 75)
print(leaderboard_df.to_string(index=False))
print("=" * 75 + "\n")


# =====================================================================
# 4. VISUALIZE LEADERBOARD METRICS
# =====================================================================
plt.figure(figsize=(12, 6))
df_melted = leaderboard_df.melt(id_vars="Algorithm", value_vars=["Accuracy", "Recall", "F1-Score"], var_name="Metric", value_name="Score")
sns.barplot(data=df_melted, x="Algorithm", y="Score", hue="Metric", palette="mako")

plt.title("Module 1 Capstone: Telco Customer Churn (Workspace Algorithms)", fontsize=14, fontweight="bold")
plt.ylabel("Cross-Validation Score (5-Fold)")
plt.ylim(0.4, 1.0)
plt.xticks(rotation=20)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
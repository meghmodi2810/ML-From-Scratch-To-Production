# Day 12: Cross-Validation, Diagnostic Curves, & Bayesian Optimization (Optuna)
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from xgboost import XGBClassifier

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 1. DATASET SETUP
# =====================================================================
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=12, n_redundant=4,
    n_classes=2, weights=[0.75, 0.25], random_state=42
)

# =====================================================================
# 2. 30-TRIAL OPTUNA BAYESIAN OPTIMIZATION STUDY
# =====================================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 30, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "eval_metric": "logloss",
        "random_state": 42
    }

    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    return float(np.mean(scores))

print("Starting 30-trial Optuna Bayesian Optimization Study...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=30)

print("=" * 65)
print("OPTUNA BAYESIAN OPTIMIZATION RESULTS")
print("=" * 65)
print(f"Best Validation Accuracy : {study.best_value:.4f}")
print("Best Hyperparameters     :")
for k, v in study.best_params.items():
    print(f"  {k:<18}: {v}")
print("=" * 65 + "\n")

# =====================================================================
# 3. VISUALIZING OPTUNA HISTORY & DIAGNOSTIC LEARNING CURVE
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot 1: Optimization History
trial_values = [t.value for t in study.trials if t.value is not None]
best_values = np.maximum.accumulate(trial_values)

axes[0].plot(range(1, len(trial_values) + 1), trial_values, "o", color="teal", alpha=0.6, label="Trial Score")
axes[0].plot(range(1, len(best_values) + 1), best_values, "-", color="crimson", linewidth=2.5, label="Best Score So Far")
axes[0].set_title("1. Optuna Trial Optimization History (30 Trials)", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Trial Number")
axes[0].set_ylabel("5-Fold Stratified CV Accuracy")
axes[0].legend()
axes[0].grid(True, linestyle=":", alpha=0.6)

# Plot 2: Diagnostic Learning Curve for Best XGBoost Model
best_model = XGBClassifier(**study.best_params, eval_metric="logloss", random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy"
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

axes[1].plot(train_sizes, train_mean, "o-", color="crimson", linewidth=2, label="Train Accuracy")
axes[1].plot(train_sizes, test_mean, "s-", color="navy", linewidth=2, label="Validation Accuracy")
axes[1].set_title("2. Best XGBoost Diagnostic Learning Curve", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Training Set Size (Samples)")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()

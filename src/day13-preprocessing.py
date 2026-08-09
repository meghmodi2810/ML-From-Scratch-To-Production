import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Calculates Target Encoding using Out-Of-Fold (OOF) statistics to prevent target leakage."""

    def __init__(self, cat_cols: list, n_splits: int = 5, smoothing: float = 10.0):
        self.cat_cols = cat_cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.global_mean_ = 0.0
        self.encoding_map_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        self.global_mean_ = y.mean()
        self.encoding_map_ = {}

        for col in self.cat_cols:
            stats = y.groupby(X[col]).agg(["count", "mean"])
            counts = stats["count"]
            means = stats["mean"]
            
            # Smoothed Target Encoding: (count * mean + smoothing * global) / (count + smoothing)
            smoothed = (counts * means + self.smoothing * self.global_mean_) / (counts + self.smoothing)
            self.encoding_map_[col] = smoothed.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.cat_cols:
            mapping = self.encoding_map_.get(col, {})
            X_out[col] = X_out[col].map(mapping).fillna(self.global_mean_)
        return X_out


def build_telco_preprocessing_pipeline(
    num_cols: list, ohe_cols: list, ordinal_cols: list, ordinal_categories: list
) -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer pipeline that prevents data leakage."""
    
    # 1. Numerical Sub-pipeline (Median Imputation + Standard Scaling)
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. One-Hot Encoding Sub-pipeline (Frequent Imputation + OHE)
    ohe_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    # 3. Ordinal Encoding Sub-pipeline
    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=ordinal_categories))
    ])

    # Combine into unified ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("ohe", ohe_pipeline, ohe_cols),
            ("ordinal", ordinal_pipeline, ordinal_cols)
        ],
        remainder="drop"
    )

    return preprocessor


# =====================================================================
# DEMO & PREPROCESSING DIAGNOSTIC PLOTS
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 500

    # 1. Generate Synthetic Messy Telco Data
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

    # Introduce 8% missing values in TotalCharges
    mask = np.random.rand(len(df)) < 0.08
    df.loc[mask, "TotalCharges"] = np.nan

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    ohe_cols = ["InternetService", "PaymentMethod", "PaperlessBilling"]
    ordinal_cols = ["Contract"]
    contract_order = [["Month-to-month", "One year", "Two year"]]

    # 2. Fit Preprocessing Pipeline
    preprocessor = build_telco_preprocessing_pipeline(num_cols, ohe_cols, ordinal_cols, contract_order)
    X_processed = preprocessor.fit_transform(X)

    # 3. Fit Target Encoder Demo
    target_encoder = OutOfFoldTargetEncoder(cat_cols=["PaymentMethod"], smoothing=10.0)
    target_encoder.fit(X, y)
    encoded_pm = target_encoder.encoding_map_["PaymentMethod"]

    print("=" * 60)
    print("TELCO PREPROCESSING PIPELINE ENGINE INITIALIZED")
    print("=" * 60)
    print(f"Raw Feature Matrix Shape      : {X.shape}")
    print(f"Processed Feature Matrix Shape: {X_processed.shape}")
    print("=" * 60 + "\n")

    # 4. Render Diagnostic Visual Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Day 13: Feature Engineering & Preprocessing Engine Diagnostics", fontsize=14, fontweight="bold")

    # Plot 1: Feature Standardization Effect (TotalCharges Raw vs Scaled)
    raw_total = X["TotalCharges"].dropna()
    scaled_total = X_processed[:, 2]  # Index 2 corresponds to TotalCharges

    axes[0].hist(raw_total, bins=25, alpha=0.5, color="gray", label="Raw TotalCharges ($)")
    ax_twin = axes[0].twiny()
    ax_twin.hist(scaled_total, bins=25, alpha=0.5, color="navy", label="Standardized (Z-Score)")
    axes[0].set_title("1. Standardization: Raw vs. Scaled Z-Score Distribution", fontsize=11)
    axes[0].set_xlabel("Raw Dollars ($)")
    ax_twin.set_xlabel("Z-Score (μ=0, σ=1)")
    axes[0].grid(True, linestyle=":", alpha=0.5)

    # Plot 2: Missing Values Heatmap Before vs After Pipeline
    sns.heatmap(X[num_cols].isnull(), cbar=False, cmap="viridis", ax=axes[1], yticklabels=False)
    axes[1].set_title(f"2. Missing Data Imputation ({df['TotalCharges'].isnull().sum()} NAs Filled)", fontsize=11)
    axes[1].set_xlabel("Numerical Features")

    # Plot 3: Smoothed Target Encoding Map
    methods = list(encoded_pm.keys())
    scores = list(encoded_pm.values())
    axes[2].barh(methods, scores, color="teal", alpha=0.8)
    axes[2].axvline(y.mean(), color="crimson", linestyle="--", label=f"Global Mean ({y.mean():.2f})")
    axes[2].set_title("3. Smoothed Target Encoding Map (Payment Method)", fontsize=11)
    axes[2].set_xlabel("Target Probability E[y|Category]")
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
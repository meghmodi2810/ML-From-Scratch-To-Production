# src/features/preprocess.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUM_COLS = ["trip_distance", "fare_amount", "tolls_amount", "passenger_count"]
CAT_COLS = ["PULocationID", "DOLocationID", "payment_type", "RatecodeID"]

def build_preprocessor() -> ColumnTransformer:
    """Builds a leakage-free ColumnTransformer for real NYC Taxi features."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                NUM_COLS
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
                ]),
                CAT_COLS
            ),
        ]
    )

def prepare_real_taxi_dataset(file_path: str, sample_size: int = 50000) -> pd.DataFrame:
    """Loads and preprocesses real TLC Yellow Taxi Parquet/CSV data."""
    if file_path.endswith(".parquet"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    # Filter out anomalous trips
    df = df[
        (df["fare_amount"] > 2.5) & (df["fare_amount"] <= 500.0) &
        (df["trip_distance"] > 0.0) & (df["trip_distance"] <= 100.0) &
        (df["payment_type"].isin([1, 2]))  # 1 = Credit Card, 2 = Cash
    ].copy()

    # Subsample for fast, stable training/serving warmups if requested
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Cast IDs to strings for categorical handling
    df["PULocationID"] = df["PULocationID"].astype(str)
    df["DOLocationID"] = df["DOLocationID"].astype(str)
    df["payment_type"] = df["payment_type"].astype(str)
    df["RatecodeID"] = df["RatecodeID"].fillna(1).astype(int).astype(str)
    df["passenger_count"] = df["passenger_count"].fillna(1).astype(float)
    df["tolls_amount"] = df["tolls_amount"].fillna(0.0).astype(float)

    # Target: High tip indicator (Tip > 20% of base fare)
    df["high_tip_indicator"] = ((df["tip_amount"] / df["fare_amount"]) >= 0.20).astype(int)

    return df
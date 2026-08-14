# src/features/preprocess.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

NUM_COLS = ["trip_distance", "fare_amount", "tolls_amount"]
OHE_COLS = ["PULocationID", "payment_type"]
ORDINAL_COLS = ["RatecodeID"]
RATECODE_ORDER = [["Standard", "JFK", "Negotiated"]]

def build_preprocessor() -> ColumnTransformer:
    """Builds an isolated, deterministic ColumnTransformer pipeline."""
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
                "ohe",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
                ]),
                OHE_COLS
            ),
            (
                "ord",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ord", OrdinalEncoder(categories=RATECODE_ORDER, handle_unknown="use_encoded_value", unknown_value=-1))
                ]),
                ORDINAL_COLS
            ),
        ]
    )

def clean_missing_tolls(df: pd.DataFrame) -> pd.DataFrame:
    """Explicitly imputes missing values in tolls_amount with 0.0."""
    df_clean = df.copy()
    df_clean["tolls_amount"] = df_clean["tolls_amount"].fillna(0.0)
    return df_clean
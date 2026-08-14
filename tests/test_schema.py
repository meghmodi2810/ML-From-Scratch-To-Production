# tests/test_schema.py
import pytest
import pandera as pa
from src.validation.schema import validate_raw_data, RawTaxiSchema

def test_valid_data_passes_schema(sample_valid_taxi_df):
    """Test that clean, valid data passes Pandera schema validation."""
    validated_df = validate_raw_data(sample_valid_taxi_df)
    assert len(validated_df) == 4
    assert set(RawTaxiSchema.columns.keys()).issubset(set(validated_df.columns))

def test_invalid_negative_values_raise_error(sample_invalid_taxi_df):
    """Test that negative numeric values or invalid enums trigger SchemaError."""
    with pytest.raises(pa.errors.SchemaError):
        validate_raw_data(sample_invalid_taxi_df)

def test_null_target_raises_error(sample_valid_taxi_df):
    """Test that null values in the target column are rejected."""
    df_corrupted = sample_valid_taxi_df.copy()
    df_corrupted.loc[0, "high_tip_indicator"] = None
    with pytest.raises(pa.errors.SchemaError):
        validate_raw_data(df_corrupted)
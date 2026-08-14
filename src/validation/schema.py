# src/validation/schema.py
import pandera as pa
from pandera import Column, Check, DataFrameSchema

# =====================================================================
# RAW TAXI DATA SCHEMA CONTRACT (PANDERA)
# =====================================================================
RawTaxiSchema = DataFrameSchema(
    columns={
        "trip_distance": Column(
            pa.Float,
            Check.greater_than_or_equal_to(0.0),
            nullable=False,
            description="Trip distance in miles; must be non-negative."
        ),
        "fare_amount": Column(
            pa.Float,
            Check.greater_than_or_equal_to(0.0),
            nullable=False,
            description="Base fare amount; must be non-negative."
        ),
        "tolls_amount": Column(
            pa.Float,
            Check.greater_than_or_equal_to(0.0),
            nullable=True,
            description="Toll fees; can contain nulls prior to preprocessing."
        ),
        "PULocationID": Column(
            pa.String,
            Check.isin(["Zone_A", "Zone_B", "Zone_C"]),
            nullable=False,
            description="Pickup Location Zone identifier."
        ),
        "payment_type": Column(
            pa.String,
            Check.isin(["Credit_Card", "Cash", "Dispute"]),
            nullable=False,
            description="Transaction payment method."
        ),
        "RatecodeID": Column(
            pa.String,
            Check.isin(["Standard", "JFK", "Negotiated"]),
            nullable=False,
            description="Rate code ID assigned to the trip."
        ),
        "high_tip_indicator": Column(
            pa.Int,
            Check.isin([0, 1]),
            nullable=False,
            description="Binary target label (1 = high tip, 0 = standard/low tip)."
        ),
    },
    coerce=True,
    strict=False  # Allow extra metadata columns if passed
)

def validate_raw_data(df):
    """
    Validates an incoming DataFrame against the RawTaxiSchema contract.
    Raises pandera.errors.SchemaError if validation fails.
    """
    return RawTaxiSchema.validate(df)
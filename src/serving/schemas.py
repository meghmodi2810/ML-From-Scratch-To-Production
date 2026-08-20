# src/serving/schemas.py
from pydantic import BaseModel, ConfigDict, Field


class TaxiPredictionRequest(BaseModel):
    """Input payload matching real NYC TLC Yellow Taxi features."""
    trip_distance: float = Field(..., ge=0.0, le=100.0, description="Trip distance in miles", examples=[3.4])
    fare_amount: float = Field(..., ge=2.5, le=500.0, description="Base fare in USD", examples=[18.50])
    tolls_amount: float = Field(default=0.0, ge=0.0, description="Tolls paid in USD", examples=[0.0])
    passenger_count: float = Field(default=1.0, ge=1.0, le=9.0, description="Number of passengers", examples=[1.0])
    PULocationID: str = Field(default="161", description="Pickup TLC Zone ID (e.g. 161 = Midtown Center)", examples=["161"])
    DOLocationID: str = Field(default="236", description="Dropoff TLC Zone ID (e.g. 236 = Upper East Side North)", examples=["236"])
    payment_type: str = Field(default="1", description="Payment Type (1 = Credit card, 2 = Cash)", examples=["1"])
    RatecodeID: str = Field(default="1", description="Ratecode (1 = Standard, 2 = JFK)", examples=["1"])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trip_distance": 3.4,
                "fare_amount": 18.50,
                "tolls_amount": 0.0,
                "passenger_count": 1.0,
                "PULocationID": "161",
                "DOLocationID": "236",
                "payment_type": "1",
                "RatecodeID": "1"
            }
        }
    )

class TaxiPredictionResponse(BaseModel):
    """Prediction output schema."""
    high_tip_prediction: int = Field(..., description="0 = Standard/Low Tip, 1 = High Tip (>20%)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of receiving high tip")
    model_version: str = Field(..., description="Model registry version used for inference")
    latency_ms: float = Field(..., description="Inference execution time in milliseconds")

class HealthCheckResponse(BaseModel):
    """Health check probe schema."""
    status: str = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., examples=[True])
    data_source: str = Field(..., examples=["/data/yellow_tripdata_2023-01.parquet"])
    model_uri: str = Field(..., examples=["models:/TLC_Yellow_Taxi_Production_DAG/Production"])

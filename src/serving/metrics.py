# src/serving/metrics.py
"""
Prometheus Metrics Registry for NYC TLC Yellow Taxi Model Serving.
Tracks operational telemetry (RPS, latency percentiles, error rates) and
domain-specific ML telemetry (prediction distributions, probability confidence, data drift).
"""
from prometheus_client import Counter, Gauge, Histogram

# =====================================================================
# 1. OPERATIONAL & SYSTEM METRICS
# =====================================================================
# Request Counter by method, endpoint path, and HTTP response status code
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total count of incoming HTTP requests across endpoints.",
    ["method", "endpoint", "status_code"]
)

# Request Latency Histogram with fine-grained sub-second ML buckets (in seconds)
# Buckets: 2ms, 5ms, 10ms, 25ms, 50ms, 75ms, 100ms, 250ms, 500ms, 1s, 2.5s
ML_LATENCY_BUCKETS = (
    0.002, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0, 2.5
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "End-to-end HTTP request processing latency in seconds.",
    ["method", "endpoint"],
    buckets=ML_LATENCY_BUCKETS
)

# =====================================================================
# 2. MACHINE LEARNING & PREDICTION DRIFT METRICS
# =====================================================================
# Prediction Count by Class (0 = Standard/Low Tip, 1 = High Tip) and Model Version
TAXI_PREDICTIONS_TOTAL = Counter(
    "taxi_predictions_total",
    "Total count of model inference predictions partitioned by predicted class.",
    ["prediction_class", "model_version"]
)

# Confidence Distribution: Predicted probability of high tip (0.0 to 1.0)
PROBABILITY_BUCKETS = (
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
)

TAXI_PREDICTION_PROBABILITY = Histogram(
    "taxi_prediction_probability",
    "Histogram of predicted high tip probability scores.",
    ["model_version"],
    buckets=PROBABILITY_BUCKETS
)

# Model Health Status (1 = Loaded in Memory & Healthy, 0 = Unloaded / Failed)
TAXI_MODEL_LOAD_STATUS = Gauge(
    "taxi_model_load_status",
    "Indicates model memory readiness (1 = Healthy/Loaded, 0 = Unhealthy).",
    ["model_name", "model_version"]
)

# =====================================================================
# 3. REAL-TIME DATA DRIFT TELEMETRY
# =====================================================================
# Feature Value Distributions for Drift Detection
FARE_AMOUNT_BUCKETS = (
    2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 200.0
)

TAXI_FEATURE_FARE_AMOUNT = Histogram(
    "taxi_feature_fare_amount_dollars",
    "Distribution of incoming fare_amount feature values for data drift monitoring.",
    buckets=FARE_AMOUNT_BUCKETS
)

TRIP_DISTANCE_BUCKETS = (
    0.5, 1.0, 2.0, 3.5, 5.0, 7.5, 10.0, 15.0, 25.0, 50.0
)

TAXI_FEATURE_TRIP_DISTANCE = Histogram(
    "taxi_feature_trip_distance_miles",
    "Distribution of incoming trip_distance feature values for data drift monitoring.",
    buckets=TRIP_DISTANCE_BUCKETS
)

# tests/locustfile.py
# =====================================================================
# NYC TLC YELLOW TAXI MLOPS - PRODUCTION LOAD TESTING SUITE (LOCUST)
# =====================================================================
"""
Multi-persona realistic load testing script for FastAPI ML inference service.

User Personas:
1. StandardTaxiPassenger (weight=6): Standard NYC trips across Midtown/Downtown.
2. AirportCommuter (weight=2): High-fare JFK/LGA trips with tolls and high tip probability.
3. EdgeCaseAndChaosUser (weight=1): Boundary conditions and occasional 422 payload tests.
4. SystemHealthProbe (weight=1): Infrastructure health/metrics monitoring probes.

Usage:
  - Web UI: locust -f tests/locustfile.py --host http://localhost:8000
  - Headless CLI: locust -f tests/locustfile.py --headless -u 20 -r 5 --run-time 1m --host http://localhost:8000
"""

import logging
import random
import time
from typing import Any, Dict

from locust import HttpUser, between, constant_pacing, events, tag, task

logger = logging.getLogger("locust.mlops")

# ---------------------------------------------------------------------
# REALISTIC DATA DISTRIBUTIONS & NYC TLC ZONE CATALOGS
# ---------------------------------------------------------------------
POPULAR_MANHATTAN_ZONES = [
    "161",  # Midtown Center
    "162",  # Midtown East
    "236",  # Upper East Side North
    "237",  # Upper East Side South
    "142",  # Lincoln Square East
    "141",  # Lenox Hill West
    "48",   # Clinton East
    "68",   # East Chelsea
    "79",   # East Village
    "170",  # Murray Hill
    "230",  # Times Sq/Theatre District
    "186",  # Penn Station/Madison Sq West
]

AIRPORT_ZONES = [
    "132",  # JFK Airport
    "138",  # LaGuardia Airport
    "1"     # Newark Airport
]


def generate_standard_payload() -> Dict[str, Any]:
    """Generates a realistic standard NYC inner-city taxi ride."""
    distance = round(random.uniform(0.8, 6.5), 2)
    base_fare = round(random.uniform(7.0, 26.0) + (distance * 2.5), 2)
    return {
        "trip_distance": distance,
        "fare_amount": max(2.50, min(500.0, base_fare)),
        "tolls_amount": 0.0,
        "passenger_count": float(random.choices([1, 2, 3, 4], weights=[0.70, 0.18, 0.08, 0.04])[0]),
        "PULocationID": random.choice(POPULAR_MANHATTAN_ZONES),
        "DOLocationID": random.choice(POPULAR_MANHATTAN_ZONES),
        "payment_type": random.choices(["1", "2"], weights=[0.82, 0.18])[0],  # 82% credit card
        "RatecodeID": "1",  # Standard rate
    }


def generate_airport_payload() -> Dict[str, Any]:
    """Generates a long-distance, high-fare airport commute trip."""
    is_jfk = random.choice([True, False])
    distance = round(random.uniform(12.0, 24.0), 2)
    fare = round(random.uniform(52.0, 75.0), 2) if is_jfk else round(random.uniform(35.0, 55.0), 2)
    tolls = random.choice([6.55, 13.10, 0.0])
    pu_zone = random.choice(POPULAR_MANHATTAN_ZONES)
    do_zone = "132" if is_jfk else "138"

    return {
        "trip_distance": distance,
        "fare_amount": fare,
        "tolls_amount": tolls,
        "passenger_count": float(random.choices([1, 2, 3, 5], weights=[0.60, 0.25, 0.10, 0.05])[0]),
        "PULocationID": pu_zone,
        "DOLocationID": do_zone,
        "payment_type": "1",
        "RatecodeID": "2" if is_jfk else "1",
    }


def generate_boundary_payload() -> Dict[str, Any]:
    """Generates valid edge-case boundary payloads (min fares, max distances)."""
    scenario = random.choice(["min_fare", "short_trip", "max_group", "long_suburb"])
    if scenario == "min_fare":
        return {
            "trip_distance": 0.1,
            "fare_amount": 2.50,
            "tolls_amount": 0.0,
            "passenger_count": 1.0,
            "PULocationID": "161",
            "DOLocationID": "161",
            "payment_type": "2",
            "RatecodeID": "1",
        }
    elif scenario == "short_trip":
        return {
            "trip_distance": 0.4,
            "fare_amount": 4.50,
            "tolls_amount": 0.0,
            "passenger_count": 1.0,
            "PULocationID": "236",
            "DOLocationID": "237",
            "payment_type": "1",
            "RatecodeID": "1",
        }
    elif scenario == "max_group":
        return {
            "trip_distance": 4.5,
            "fare_amount": 22.00,
            "tolls_amount": 0.0,
            "passenger_count": 6.0,
            "PULocationID": "142",
            "DOLocationID": "79",
            "payment_type": "1",
            "RatecodeID": "1",
        }
    else:
        return {
            "trip_distance": 32.0,
            "fare_amount": 125.00,
            "tolls_amount": 18.50,
            "passenger_count": 2.0,
            "PULocationID": "161",
            "DOLocationID": "1",
            "payment_type": "1",
            "RatecodeID": "3",
        }


# =====================================================================
# LOCUST USER CLASSES (PERSONAS)
# =====================================================================

class StandardTaxiPassenger(HttpUser):
    """
    Persona 1: Standard NYC Ride Passenger.
    Highest weight persona representing regular yellow cab trips in Manhattan.
    """
    weight = 6
    wait_time = between(0.1, 0.8)

    @tag("predict", "standard", "smoke")
    @task(10)
    def predict_standard_trip(self):
        payload = generate_standard_payload()
        with self.client.post(
            "/predict",
            json=payload,
            name="POST /predict [Standard]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "high_tip_prediction" in data and "probability" in data:
                    response.success()
                else:
                    response.failure(f"Malformed response payload: {response.text}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @tag("health", "smoke")
    @task(1)
    def check_health(self):
        self.client.get("/health", name="GET /health [Passenger Probe]")


class AirportCommuter(HttpUser):
    """
    Persona 2: Airport Commuter.
    Long distance, high fare trips to JFK & LGA with tolls.
    """
    weight = 2
    wait_time = between(0.2, 1.2)

    @tag("predict", "airport")
    @task(5)
    def predict_airport_trip(self):
        payload = generate_airport_payload()
        with self.client.post(
            "/predict",
            json=payload,
            name="POST /predict [Airport]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Airport prediction failed: {response.status_code}")


class EdgeCaseAndChaosUser(HttpUser):
    """
    Persona 3: Edge Case and Boundary Testing.
    Sends boundary values and intentional invalid requests to test resilience and 422 error rate tracking.
    """
    weight = 1
    wait_time = between(0.5, 2.0)

    @tag("predict", "boundary")
    @task(3)
    def predict_boundary_case(self):
        payload = generate_boundary_payload()
        with self.client.post(
            "/predict",
            json=payload,
            name="POST /predict [Boundary Edge-Case]",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Boundary request failed: {response.status_code}")

    @tag("predict", "chaos", "validation")
    @task(1)
    def predict_invalid_fare_boundary(self):
        """Intentionally invalid payload (fare < 2.50) to verify 422 rejection handling under load."""
        invalid_payload = {
            "trip_distance": 3.0,
            "fare_amount": 1.00,  # Below legal minimum threshold ($2.50)
            "tolls_amount": 0.0,
            "passenger_count": 1.0,
            "PULocationID": "161",
            "DOLocationID": "236",
            "payment_type": "1",
            "RatecodeID": "1",
        }
        with self.client.post(
            "/predict",
            json=invalid_payload,
            name="POST /predict [Chaos 422 Validation]",
            catch_response=True,
        ) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"Expected 422 Validation Error, got: {response.status_code}")


class SystemHealthProbe(HttpUser):
    """
    Persona 4: Synthetic Monitoring / ALB Health Probe.
    Periodically checks /health and /metrics endpoints.
    """
    weight = 1
    wait_time = constant_pacing(2.0)

    @tag("health", "observability")
    @task(2)
    def probe_health(self):
        with self.client.get("/health", name="GET /health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy" and data.get("model_loaded") is True:
                    response.success()
                else:
                    response.failure(f"Unhealthy response: {response.text}")
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @tag("metrics", "observability")
    @task(1)
    def scrape_metrics(self):
        with self.client.get("/metrics", name="GET /metrics", catch_response=True) as response:
            if response.status_code == 200 and "text/plain" in response.headers.get("content-type", ""):
                response.success()
            else:
                response.failure(f"Metrics scrape failed: {response.status_code}")


# =====================================================================
# LOCUST EVENT LISTENERS & TELEMETRY HOOKS
# =====================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Fired when a new load test execution begins."""
    target_host = environment.host or "http://localhost:8000"
    print("\n" + "=" * 70)
    print("[LOCUST LOAD TEST INITIALIZING]")
    print(f"Target Host: {target_host}")
    print(f"Prometheus Metrics Endpoint: {target_host}/metrics")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Fired when load test execution concludes."""
    stats = environment.runner.stats.total
    print("\n" + "=" * 70)
    print("[LOCUST LOAD TEST COMPLETED]")
    print(f"Total Requests Processed: {stats.num_requests}")
    print(f"Total Failures:           {stats.num_failures}")
    print(f"Average RPS:              {round(stats.total_rps, 2)}")
    print(f"Median Latency (p50):     {round(stats.median_response_time, 2)} ms")
    print(f"95th Percentile (p95):    {round(stats.get_response_time_percentile(0.95), 2)} ms")
    print(f"99th Percentile (p99):    {round(stats.get_response_time_percentile(0.99), 2)} ms")
    print("=" * 70 + "\n")

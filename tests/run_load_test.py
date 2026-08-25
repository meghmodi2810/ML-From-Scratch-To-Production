# tests/run_load_test.py
# =====================================================================
# NYC TAXI MLOPS - AUTOMATED LOAD TEST RUNNER & SLA BENCHMARK SUITE
# =====================================================================
"""
Automated benchmark runner for Locust load tests.
Executes configured test profiles, exports HTML/CSV reports, and validates performance SLAs.

Usage:
  python tests/run_load_test.py --profile smoke --host http://localhost:8000
  python tests/run_load_test.py --profile nominal --host http://localhost:8000
  python tests/run_load_test.py --profile stress --host http://localhost:8000
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from typing import Any, Dict

# Predefined Test Profiles
PROFILES = {
    "smoke": {
        "description": "Fast sanity verification under minimal concurrency",
        "users": 5,
        "spawn_rate": 2,
        "run_time": "15s",
        "sla": {
            "max_error_rate_pct": 1.0,
            "max_p95_latency_ms": 100.0,
            "max_p99_latency_ms": 200.0,
        },
    },
    "nominal": {
        "description": "Standard baseline production load",
        "users": 20,
        "spawn_rate": 5,
        "run_time": "45s",
        "sla": {
            "max_error_rate_pct": 0.5,
            "max_p95_latency_ms": 120.0,
            "max_p99_latency_ms": 200.0,
        },
    },
    "stress": {
        "description": "High-concurrency stress test to identify saturation bottlenecks",
        "users": 50,
        "spawn_rate": 10,
        "run_time": "60s",
        "sla": {
            "max_error_rate_pct": 2.0,
            "max_p95_latency_ms": 150.0,
            "max_p99_latency_ms": 300.0,
        },
    },
    "spike": {
        "description": "Sudden burst traffic spike simulation",
        "users": 100,
        "spawn_rate": 50,
        "run_time": "30s",
        "sla": {
            "max_error_rate_pct": 5.0,
            "max_p95_latency_ms": 250.0,
            "max_p99_latency_ms": 500.0,
        },
    },
}


def parse_locust_stats_csv(stats_csv_path: str) -> Dict[str, Any]:
    """Parses the Locust aggregated stats CSV output."""
    if not os.path.exists(stats_csv_path):
        return {}

    with open(stats_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Name") == "Aggregated" or row.get("Type") is None or row.get("Name") == "Total":
                try:
                    num_requests = int(row.get("Request Count", 0))
                    num_failures = int(row.get("Failure Count", 0))
                    median_ms = float(row.get("Median Response Time", 0))
                    avg_ms = float(row.get("Average Response Time", 0))
                    p95_ms = float(row.get("95%", 0))
                    p99_ms = float(row.get("99%", 0))
                    rps = float(row.get("Requests/s", 0))
                    error_rate_pct = (num_failures / num_requests * 100.0) if num_requests > 0 else 0.0

                    return {
                        "num_requests": num_requests,
                        "num_failures": num_failures,
                        "error_rate_pct": error_rate_pct,
                        "median_ms": median_ms,
                        "avg_ms": avg_ms,
                        "p95_ms": p95_ms,
                        "p99_ms": p99_ms,
                        "rps": rps,
                    }
                except (ValueError, KeyError):
                    continue
    return {}


def run_benchmark(
    profile_name: str,
    host: str,
    users: int = None,
    spawn_rate: int = None,
    run_time: str = None,
    tags: str = None,
    output_dir: str = "reports/load_tests",
) -> bool:
    """Executes the Locust test and evaluates SLA performance criteria."""
    profile = PROFILES.get(profile_name, PROFILES["smoke"])
    target_users = users or profile["users"]
    target_spawn = spawn_rate or profile["spawn_rate"]
    target_duration = run_time or profile["run_time"]
    sla = profile["sla"]

    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"loadtest_{profile_name}_{timestamp}"
    csv_prefix = os.path.join(output_dir, prefix)
    html_report = os.path.join(output_dir, f"{prefix}_report.html")

    print("\n" + "=" * 75)
    print(f">> RUNNING LOAD TEST BENCHMARK: [{profile_name.upper()}]")
    print(f"Description: {profile['description']}")
    print(f"Target Host: {host}")
    print(f"Concurrency: {target_users} Users | Spawn Rate: {target_spawn}/s | Duration: {target_duration}")
    print(f"HTML Report: {html_report}")
    print("=" * 75 + "\n")

    # Locate Python executable inside current environment
    python_bin = sys.executable
    locust_script = os.path.join("tests", "locustfile.py")

    cmd = [
        python_bin,
        "-m",
        "locust",
        "-f",
        locust_script,
        "--headless",
        "-u",
        str(target_users),
        "-r",
        str(target_spawn),
        "--run-time",
        str(target_duration),
        "--host",
        host,
        "--html",
        html_report,
        "--csv",
        csv_prefix,
    ]

    if tags:
        cmd.extend(["--tags", tags])

    print(f"[Command] {' '.join(cmd)}\n")
    proc = subprocess.run(cmd, check=False)

    stats_csv_file = f"{csv_prefix}_stats.csv"
    stats = parse_locust_stats_csv(stats_csv_file)

    print("\n" + "=" * 75)
    print("BENCHMARK PERFORMANCE & SLA VALIDATION SUMMARY")
    print("=" * 75)

    if not stats:
        print("[Warning] Could not parse stats CSV file. Check Locust run logs above.")
        return proc.returncode == 0

    print(f"Total Requests:       {stats['num_requests']:,}")
    print(f"Total Failures:       {stats['num_failures']:,} ({stats['error_rate_pct']:.2f}%)")
    print(f"Throughput:           {stats['rps']:.2f} req/sec")
    print(f"Median Latency (p50): {stats['median_ms']:.2f} ms")
    print(f"Average Latency:      {stats['avg_ms']:.2f} ms")
    print(f"p95 Latency:          {stats['p95_ms']:.2f} ms (SLA Target: < {sla['max_p95_latency_ms']} ms)")
    print(f"p99 Latency:          {stats['p99_ms']:.2f} ms (SLA Target: < {sla['max_p99_latency_ms']} ms)")
    print("-" * 75)

    # SLA Checks
    sla_passed = True
    if stats["error_rate_pct"] > sla["max_error_rate_pct"]:
        print(f"[FAIL] SLA FAILED: Error rate {stats['error_rate_pct']:.2f}% exceeds threshold {sla['max_error_rate_pct']}%")
        sla_passed = False
    else:
        print(f"[PASS] SLA PASSED: Error rate {stats['error_rate_pct']:.2f}% <= {sla['max_error_rate_pct']}%")

    if stats["p95_ms"] > sla["max_p95_latency_ms"]:
        print(f"[FAIL] SLA FAILED: p95 latency {stats['p95_ms']:.2f} ms exceeds threshold {sla['max_p95_latency_ms']} ms")
        sla_passed = False
    else:
        print(f"[PASS] SLA PASSED: p95 latency {stats['p95_ms']:.2f} ms <= {sla['max_p95_latency_ms']} ms")

    if stats["p99_ms"] > sla["max_p99_latency_ms"]:
        print(f"[FAIL] SLA FAILED: p99 latency {stats['p99_ms']:.2f} ms exceeds threshold {sla['max_p99_latency_ms']} ms")
        sla_passed = False
    else:
        print(f"[PASS] SLA PASSED: p99 latency {stats['p99_ms']:.2f} ms <= {sla['max_p99_latency_ms']} ms")

    print("=" * 75)
    return sla_passed


def main():
    parser = argparse.ArgumentParser(description="NYC Taxi MLOps Automated Load Test Suite")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default="smoke", help="Predefined load test profile")
    parser.add_argument("--host", default="http://127.0.0.1:8000", help="Target API host")
    parser.add_argument("--users", type=int, help="Override number of concurrent users")
    parser.add_argument("--spawn-rate", type=int, help="Override user spawn rate per second")
    parser.add_argument("--run-time", help="Override test duration (e.g. 30s, 2m)")
    parser.add_argument("--tags", help="Filter tasks by Locust tags (e.g. 'predict', 'health')")
    parser.add_argument("--output-dir", default="reports/load_tests", help="Directory for HTML/CSV reports")

    args = parser.parse_args()
    success = run_benchmark(
        profile_name=args.profile,
        host=args.host,
        users=args.users,
        spawn_rate=args.spawn_rate,
        run_time=args.run_time,
        tags=args.tags,
        output_dir=args.output_dir,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

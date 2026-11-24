"""Headless Locust flood tests; appends a summary row per run to locust/results/summary.csv.

    # API running on :8000 with N containers / workers
    python scripts/run_locust.py --host http://127.0.0.1:8000 --containers 1 --users 10,50,100 --time 60s
    python scripts/run_locust.py --table      # print the markdown table of all runs
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "locust" / "results"
SUMMARY = OUT / "summary.csv"
FIELDS = ["ran_at", "setup", "containers", "users", "duration", "requests", "failures",
          "median_ms", "p95_ms", "p99_ms", "avg_ms", "max_ms", "rps"]


def run_one(users: int, time_s: str, host: str, containers: str, setup: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = OUT / f"{setup}_c{containers}_u{users}"
    cmd = [
        sys.executable, "-m", "locust", "-f", str(ROOT / "locust" / "locustfile.py"),
        "--headless", "-u", str(users), "-r", str(max(1, users // 5)), "-t", time_s,
        "--host", host, "--csv", str(prefix), "--html", str(prefix) + ".html", "--only-summary",
    ]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False, cwd=ROOT)
    with open(f"{prefix}_stats.csv", newline="", encoding="utf-8") as fh:
        row = next(r for r in csv.DictReader(fh) if r["Name"] == "/predict")
    result = {
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "setup": setup,
        "containers": containers,
        "users": users,
        "duration": time_s,
        "requests": int(row["Request Count"]),
        "failures": int(row["Failure Count"]),
        "median_ms": float(row["Median Response Time"]),
        "p95_ms": float(row["95%"]),
        "p99_ms": float(row["99%"]),
        "avg_ms": round(float(row["Average Response Time"]), 1),
        "max_ms": round(float(row["Max Response Time"]), 1),
        "rps": round(float(row["Requests/s"]), 2),
    }
    new = not SUMMARY.exists()
    with SUMMARY.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(result)
    print(result, flush=True)
    return result


def table() -> str:
    with SUMMARY.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    head = "| Setup | Containers | Users | Requests | Failures | Median (ms) | p95 (ms) | p99 (ms) | Avg (ms) | RPS |"
    lines = [head, "|" + "---|" * 10]
    for r in rows:
        lines.append(
            f"| {r['setup']} | {r['containers']} | {r['users']} | {r['requests']} | {r['failures']} | "
            f"{float(r['median_ms']):.0f} | {float(r['p95_ms']):.0f} | {float(r['p99_ms']):.0f} | "
            f"{float(r['avg_ms']):.0f} | {r['rps']} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:8000")
    parser.add_argument("--containers", default="1", help="number of API containers/workers behind --host")
    parser.add_argument("--setup", default="docker", help="label, e.g. docker or local-workers")
    parser.add_argument("--users", default="10,50,100")
    parser.add_argument("--time", default="60s")
    parser.add_argument("--table", action="store_true")
    args = parser.parse_args()
    if args.table:
        print(table())
        return
    for users in [int(x) for x in args.users.split(",")]:
        run_one(users, args.time, args.host, args.containers, args.setup)
    print(table())


if __name__ == "__main__":
    main()

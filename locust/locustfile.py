"""Flood the prediction API with handwriting images.

    locust -f locust/locustfile.py --host http://127.0.0.1:8000
"""

import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = sorted(ROOT.glob("data/test/*/*.png"))[:40]
if not SAMPLES:
    raise FileNotFoundError("No test PNGs found. Run: python scripts/prepare_data.py")
PAYLOADS = [(p.name, p.read_bytes()) for p in SAMPLES]


class HandwritingUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, data = random.choice(PAYLOADS)
        self.client.post("/predict", files={"file": (name, data, "image/png")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")

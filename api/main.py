"""FastAPI service for the dyslexia handwriting screening model: health, predict, upload, retrain, evaluate."""

from __future__ import annotations

import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import json

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    AUTO_RETRAIN,
    CLASS_NAMES,
    FEATURE_STATS_PATH,
    MODEL_META_PATH,
    MODEL_PATH,
    PRODUCTION_EVAL_PATH,
    RETRAIN_EPOCHS,
    RETRAIN_MIN_NEW_SAMPLES,
    TEST_DIR,
    TRAIN_DIR,
)
from src.model import evaluate_model, load_metrics  # noqa: E402
from src.prediction import get_model, predict_bytes, reload_model  # noqa: E402
from src.preprocessing import FEATURE_COLUMNS, build_feature_table, dataset_counts  # noqa: E402
from src.retrain import (  # noqa: E402
    count_uploads,
    get_status,
    ingest_zip,
    maybe_auto_retrain,
    save_labelled_files,
    should_retrain,
    trigger_retrain_async,
)

STARTED_AT = datetime.now(timezone.utc)
LATENCIES = deque(maxlen=500)

app = FastAPI(
    title="Dyslexia Handwriting Screening API",
    description=(
        "Early identification of children at risk of dyslexia from images of single handwritten "
        "characters (Mendeley Data 10.17632/b37s8b94nv.1). Educational prototype, not a diagnosis."
    ),
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _uptime_seconds() -> float:
    return (datetime.now(timezone.utc) - STARTED_AT).total_seconds()


@app.get("/")
def root():
    return {
        "service": "Dyslexia Handwriting Screening API",
        "docs": "/docs",
        "endpoints": [
            "/health", "/model/info", "/model/history", "/predict", "/upload-data",
            "/retrain", "/retrain/status", "/evaluate", "/visualizations/stats",
            "/visualizations/features", "/production/latency",
        ],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(_uptime_seconds(), 2),
        "started_at": STARTED_AT.isoformat(),
        "model_loaded": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/info")
def model_info():
    metrics = load_metrics()
    return {
        "classes": CLASS_NAMES,
        "train_counts": dataset_counts(TRAIN_DIR),
        "test_counts": dataset_counts(TEST_DIR),
        "metrics": {
            k: metrics[k]
            for k in (
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "sensitivity_at_risk",
                "specificity",
                "roc_auc",
                "n_test",
                "evaluated_at",
            )
            if k in metrics
        },
        "confusion_matrix": metrics.get("confusion_matrix"),
        "uploads_pending": count_uploads(),
        "retrain_threshold": RETRAIN_MIN_NEW_SAMPLES,
        "retrain_recommended": should_retrain(),
        "auto_retrain_enabled": AUTO_RETRAIN,
    }


@app.get("/model/history")
def model_history():
    if not MODEL_META_PATH.exists():
        raise HTTPException(404, "No trained model metadata yet.")
    return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))


def _feature_table():
    import pandas as pd

    if not FEATURE_STATS_PATH.exists():
        df = build_feature_table()
        if df.empty:
            raise HTTPException(404, "No images found to build the feature table.")
        return df
    return pd.read_csv(FEATURE_STATS_PATH)


@app.get("/visualizations/features")
def visualization_features():
    """Per-image features (about 1,600 rows) for distribution plots in the UI."""
    df = _feature_table()
    cols = ["label", "label_name", "split", *FEATURE_COLUMNS]
    return {"rows": df[cols].round(5).to_dict(orient="records")}


@app.get("/visualizations/stats")
def visualization_stats():
    df = _feature_table()
    summary = (
        df.groupby("label_name")[FEATURE_COLUMNS]
        .mean()
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "n_rows": int(len(df)),
        "class_counts": df["label_name"].value_counts().to_dict(),
        "feature_means_by_class": summary,
        "overall": df[FEATURE_COLUMNS].mean().to_dict(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    started = time.perf_counter()
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    try:
        result = predict_bytes(data, filename=file.filename or "upload.png")
    except UnidentifiedImageError as exc:
        raise HTTPException(400, "File is not a readable image.") from exc
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Prediction failed: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000
    LATENCIES.append(latency_ms)
    result["latency_ms"] = round(latency_ms, 2)
    return result


@app.post("/upload-data")
async def upload_data(
    files: list[UploadFile] = File(default=[]),
    class_id: int | None = Form(default=None),
    archive: UploadFile | None = File(default=None),
):
    """Bulk labelled images for retraining.

    Send either a zip whose folders name the class (`dyslexic` / `non_dyslexic`,
    `yes` / `no` or `1` / `0`), or many images plus `class_id` (0 = non-dyslexic, 1 = dyslexic).
    """
    saved = 0
    skipped = 0
    extra = {}
    if archive is not None and archive.filename:
        payload = await archive.read()
        extra = ingest_zip(payload)
        saved += extra.get("saved", 0)
        skipped += extra.get("skipped", 0)
    if files and class_id is None:
        raise HTTPException(400, "Provide class_id (0 = non-dyslexic, 1 = dyslexic) with loose image uploads.")
    if files:
        if class_id not in CLASS_NAMES:
            raise HTTPException(400, "class_id must be 0 (non-dyslexic) or 1 (dyslexic)")
        blobs = [(f.filename or "img.png", await f.read()) for f in files]
        loose = save_labelled_files(blobs, class_id)
        saved += loose["saved"]
        skipped += loose["skipped"]
    if saved == 0:
        raise HTTPException(400, "No valid labelled images found in the upload.")
    auto = should_retrain()
    auto_job = maybe_auto_retrain()
    if auto_job:
        message = "Upload threshold reached — retraining started automatically."
    elif auto:
        message = "Upload threshold reached — press Trigger retraining."
    else:
        message = f"{count_uploads()}/{RETRAIN_MIN_NEW_SAMPLES} new images before retraining is recommended."
    return {
        "saved": saved,
        "skipped": skipped,
        "uploads_total": count_uploads(),
        "retrain_recommended": auto,
        "auto_retrain_started": bool(auto_job),
        "message": message,
    }


@app.post("/retrain")
def retrain(epochs: int = RETRAIN_EPOCHS):
    if not 1 <= epochs <= 50:
        raise HTTPException(400, "epochs must be between 1 and 50")
    return JSONResponse(trigger_retrain_async(epochs=epochs, reason="manual (button)"), status_code=202)


@app.post("/evaluate")
def evaluate_in_production():
    """Score the live model on the held-out test folder, inside the running service."""
    started = time.perf_counter()
    try:
        metrics = evaluate_model(get_model(), TEST_DIR, out_path=PRODUCTION_EVAL_PATH)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    metrics["eval_seconds"] = round(time.perf_counter() - started, 2)
    return metrics


@app.get("/retrain/status")
def retrain_status():
    return get_status()


@app.get("/production/latency")
def production_latency():
    values = list(LATENCIES)
    if not values:
        return {"n": 0, "mean_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    return {
        "n": len(values),
        "mean_ms": round(sum(values) / len(values), 2),
        "p95_ms": round(p95, 2),
        "max_ms": round(max(values), 2),
        "uptime_seconds": round(_uptime_seconds(), 2),
    }


@app.post("/admin/reload-model")
def admin_reload():
    reload_model()
    return {"ok": True}

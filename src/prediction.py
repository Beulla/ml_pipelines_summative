"""Single-image prediction for the dyslexia handwriting screening model."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.config import CLASS_NAMES, LOGS_DIR, MODEL_PATH, PREDICTION_LOG_PATH
from src.model import load_trained_model, predict_proba
from src.preprocessing import extract_image_features, load_bytes, load_image

_model = None
_model_mtime = None


def _current_mtime():
    return MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None


def get_model():
    """Cached model; reloads automatically when models/dyslexia_cnn.pt changes on disk,
    so every API container picks up a retrained model from the shared volume."""
    global _model, _model_mtime
    mtime = _current_mtime()
    if _model is None or mtime != _model_mtime:
        _model = load_trained_model()
        _model_mtime = mtime
    return _model


def reload_model():
    global _model, _model_mtime
    _model = load_trained_model()
    _model_mtime = _current_mtime()
    return _model


def predict_image(image: Image.Image, model=None) -> dict:
    """``image`` must already be preprocessed (preprocessing.preprocess_image)."""
    model = model or get_model()
    probs = predict_proba(model, image)[0]
    class_id = int(probs.argmax())
    return {
        "class_id": class_id,
        "label": CLASS_NAMES[class_id],
        "confidence": float(probs[class_id]),
        "at_risk_probability": float(probs[1]),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
        "features": extract_image_features(image),
    }


def predict_path(path: Path, model=None) -> dict:
    result = predict_image(load_image(Path(path)), model=model)
    result["source"] = str(path)
    _append_prediction_log(result)
    return result


def predict_bytes(data: bytes, filename: str = "upload.png", model=None) -> dict:
    result = predict_image(load_bytes(data), model=model)
    result["source"] = filename
    _append_prediction_log(result)
    return result


def _append_prediction_log(result: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not PREDICTION_LOG_PATH.exists()
    with PREDICTION_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp", "source", "label", "class_id", "confidence"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": result.get("source", ""),
                "label": result["label"],
                "class_id": result["class_id"],
                "confidence": f"{result['confidence']:.4f}",
            }
        )

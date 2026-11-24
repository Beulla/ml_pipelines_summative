"""Shared paths and constants for the dyslexia handwriting screening pipeline."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"
USED_UPLOAD_DIR = DATA_DIR / "uploads_used"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

IMG_SIZE = 128           # handwriting images are normalised to 128x128 grayscale
BATCH_SIZE = 32
EPOCHS = 30
RETRAIN_EPOCHS = 6
TEST_FRACTION = 0.2
RANDOM_STATE = 42
NUM_CLASSES = 2

CLASS_NAMES = {
    0: "Non-dyslexic",
    1: "Dyslexic (at risk)",
}

# Folder names accepted for labelled uploads (zip folders or class pickers).
CLASS_FOLDER_ALIASES = {
    "0": 0,
    "no": 0,
    "non_dyslexic": 0,
    "non-dyslexic": 0,
    "nondyslexic": 0,
    "normal": 0,
    "typical": 0,
    "1": 1,
    "yes": 1,
    "dyslexic": 1,
    "at_risk": 1,
    "at-risk": 1,
}

MODEL_NAME = "dyslexia_cnn"
MODEL_PATH = MODELS_DIR / f"{MODEL_NAME}.pt"
PKL_PATH = MODELS_DIR / f"{MODEL_NAME}.pkl"
METRICS_PATH = MODELS_DIR / "eval_metrics.json"
MODEL_META_PATH = MODELS_DIR / "model_meta.json"
FEATURE_STATS_PATH = DATA_DIR / "feature_stats.csv"
PREDICTION_LOG_PATH = LOGS_DIR / "predictions.csv"
RETRAIN_STATUS_PATH = LOGS_DIR / "retrain_status.json"
PRODUCTION_EVAL_PATH = LOGS_DIR / "production_eval.json"

RETRAIN_MIN_NEW_SAMPLES = int(os.environ.get("RETRAIN_MIN_NEW_SAMPLES", "20"))
# When true, an upload that pushes pending samples past the threshold starts
# retraining automatically (the "trigger when the need arises").
AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "false").lower() in {"1", "true", "yes"}

# Data source: Multilingual AI-Based Dyslexia Detection (Mendeley Data, CC BY 4.0)
# DOI 10.17632/b37s8b94nv.1 — 852 character images from 100 children aged 6-10.
DATASET_DOI = "10.17632/b37s8b94nv.1"
DATASET_ZIP_URL = (
    "https://data.mendeley.com/public-files/datasets/b37s8b94nv/files/"
    "65d6ff56-9090-44d4-8b45-6289e94fc4a7/file_downloaded"
)
DATASET_ZIP_SHA256 = "dfe07ad79fc5dd7b25f95703cf9839ae472f022bf122855a1357404bbf7c9ad4"
DATASET_FOLDER_LABELS = {"No": 0, "Yes": 1}

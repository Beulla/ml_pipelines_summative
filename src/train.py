"""Train or retrain the model.

    python -m src.train                       # first training
    python -m src.train --epochs 8 --retrain  # fine-tune the current model

A retrained model only replaces the current one if its macro F1 on the test
set does not drop by more than PROMOTE_TOLERANCE.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from src.model import LEARNING_RATE, evaluate_model, load_metrics, train_model, write_model_meta
from src.preprocessing import build_feature_table, dataset_counts

CANDIDATE_DIR = MODELS_DIR / "candidate"
PROMOTE_TOLERANCE = 0.02
RETRAIN_LR = 1e-4


def _public(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def run_training(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    retrain: bool = False,
    force_promote: bool = False,
) -> dict:
    if not any(TRAIN_DIR.glob("*/*")):
        raise FileNotFoundError(
            f"No training images under {TRAIN_DIR}. Run: python scripts/prepare_data.py"
        )
    train_counts = dataset_counts(TRAIN_DIR)
    test_counts = dataset_counts(TEST_DIR)

    warm = MODEL_PATH if (retrain and MODEL_PATH.exists()) else None
    previous = {}
    if warm:
        old = load_metrics()
        for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "sensitivity_at_risk", "specificity", "roc_auc"):
            previous[k] = old.get(k)

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = CANDIDATE_DIR / MODEL_PATH.name
    model, history = train_model(
        TRAIN_DIR,
        epochs=epochs,
        batch_size=batch_size,
        model_path=candidate_path,
        warm_start_from=warm,
        lr=RETRAIN_LR if warm else LEARNING_RATE,
    )
    metrics = evaluate_model(model, TEST_DIR, out_path=CANDIDATE_DIR / METRICS_PATH.name)
    metrics = _public(metrics)

    old_f1 = previous.get("f1_macro")
    new_f1 = metrics.get("f1_macro") or 0.0
    promoted = (
        force_promote
        or old_f1 is None
        or new_f1 >= old_f1 - PROMOTE_TOLERANCE
    )

    if promoted:
        shutil.copy2(candidate_path, MODEL_PATH)
        shutil.copy2(candidate_path.with_suffix(".pkl"), MODEL_PATH.with_suffix(".pkl"))
        METRICS_PATH.write_text(json.dumps(_public(metrics), indent=2), encoding="utf-8")
        hist = {k: [float(x) for x in v] for k, v in history.items()}
        write_model_meta(
            {
                "train_counts": train_counts,
                "test_counts": test_counts,
                "epochs_ran": len(hist.get("loss", [])),
                "history": hist,
                "test_accuracy": metrics.get("accuracy"),
                "test_f1_macro": metrics.get("f1_macro"),
                "trained_mode": "retrain (warm start)" if warm else "initial",
                "previous_metrics": previous or None,
            }
        )
        build_feature_table()

    return {
        "metrics": metrics,
        "previous_metrics": previous or None,
        "promoted": promoted,
        "mode": "retrain" if warm else "initial",
        "model_path": str(MODEL_PATH if promoted else candidate_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Train the dyslexia handwriting classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--retrain", action="store_true", help="fine-tune the current production model")
    parser.add_argument("--force", action="store_true", help="promote even if macro F1 drops")
    args = parser.parse_args()
    result = run_training(
        epochs=args.epochs, batch_size=args.batch_size, retrain=args.retrain, force_promote=args.force
    )
    print("Mode:", result["mode"], "| promoted:", result["promoted"])
    print("Model:", result["model_path"])
    print("Test accuracy:", round(result["metrics"]["accuracy"], 4))
    print("Macro F1:", round(result["metrics"]["f1_macro"], 4))
    if result["previous_metrics"]:
        print("Previous macro F1:", result["previous_metrics"].get("f1_macro"))
    print("Finished", datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()

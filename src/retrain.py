"""Uploading new data and retraining the model.

1. Uploaded images are checked, preprocessed and saved in the SQLite database
   (status 'pending') and in data/train/.
2. Retraining is started by the button (POST /retrain) or automatically when
   enough new images are pending and AUTO_RETRAIN is on.
3. Retraining reads the pending images from the database, fine-tunes the
   current model, and marks the images 'used' if the new model is kept.
"""

import json
import threading
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.config import (
    AUTO_RETRAIN,
    CLASS_NAMES,
    LOGS_DIR,
    RETRAIN_EPOCHS,
    RETRAIN_MIN_NEW_SAMPLES,
    RETRAIN_STATUS_PATH,
    TRAIN_DIR,
)
from src.database import connect, now
from src.prediction import reload_model
from src.preprocessing import CLASS_FOLDERS, IMAGE_EXTS, ensure_class_folders, load_bytes, resolve_class_id
from src.train import run_training

_lock = threading.Lock()


def _status(**kwargs) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    current = get_status() if RETRAIN_STATUS_PATH.exists() else {}
    current.update(kwargs)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    RETRAIN_STATUS_PATH.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def get_status() -> dict:
    if RETRAIN_STATUS_PATH.exists():
        return json.loads(RETRAIN_STATUS_PATH.read_text(encoding="utf-8"))
    return {"state": "idle", "message": "No retraining has been run yet."}


def _is_valid_image(data: bytes) -> bool:
    try:
        with Image.open(BytesIO(data)) as img:
            img.verify()
        return True
    except Exception:
        return False


def _write_train_image(class_id: int, name: str, png: bytes) -> None:
    path = TRAIN_DIR / CLASS_FOLDERS[class_id] / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)


def _store(data: bytes, class_id: int, filename: str, source: str) -> None:
    """Preprocess an uploaded image and save it to the database and data/train/."""
    image = load_bytes(data)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stored_name = f"upload_{stamp}_{Path(filename).stem[:40]}.png"
    buf = BytesIO()
    image.save(buf, format="PNG")
    png = buf.getvalue()

    with connect() as conn:
        conn.execute(
            """INSERT INTO uploads (original_name, stored_name, class_id, class_label, source,
                                    width, height, image_png, uploaded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (Path(filename).name, stored_name, class_id, CLASS_NAMES[class_id], source,
             image.size[0], image.size[1], png, now()),
        )
    _write_train_image(class_id, stored_name, png)


def save_labelled_files(files: list[tuple[str, bytes]], class_id: int) -> dict:
    ensure_class_folders(TRAIN_DIR)
    saved = skipped = 0
    for name, data in files:
        if Path(name).suffix.lower() not in IMAGE_EXTS or not _is_valid_image(data):
            skipped += 1
            continue
        _store(data, class_id, name, source="images")
        saved += 1
    return {"saved": saved, "skipped": skipped}


def ingest_zip(data: bytes) -> dict:
    """Save the images in a zip. The folder names give the class
    (e.g. Yes/No or dyslexic/non_dyslexic)."""
    ensure_class_folders(TRAIN_DIR)
    saved = skipped = 0
    try:
        zf = zipfile.ZipFile(BytesIO(data))
    except zipfile.BadZipFile:
        return {"saved": 0, "skipped": 0, "error": "Not a valid zip file."}
    with zf:
        for info in zf.infolist():
            path = Path(info.filename)
            if info.is_dir() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            class_id = resolve_class_id(path.parent.name) if len(path.parts) > 1 else None
            blob = zf.read(info) if info.file_size < 25_000_000 else b""
            if class_id is None or "__MACOSX" in path.parts or not _is_valid_image(blob):
                skipped += 1
                continue
            _store(blob, class_id, path.name, source="zip")
            saved += 1
    return {"saved": saved, "skipped": skipped}


def count_uploads() -> int:
    """Number of uploaded images that have not been used for retraining yet."""
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM uploads WHERE status = 'pending'").fetchone()[0]


def should_retrain() -> bool:
    return count_uploads() >= RETRAIN_MIN_NEW_SAMPLES


def retrain_job(epochs: int = RETRAIN_EPOCHS, reason: str = "manual") -> dict:
    if not _lock.acquire(blocking=False):
        return _status(state="busy", message="Retraining already in progress.")
    run_id = None
    try:
        with connect() as conn:
            rows = conn.execute(
                "SELECT id, stored_name, class_id, image_png FROM uploads WHERE status = 'pending'"
            ).fetchall()
            run_id = conn.execute(
                "INSERT INTO retrain_runs (reason, epochs, new_images, started_at) VALUES (?, ?, ?, ?)",
                (reason, epochs, len(rows), now()),
            ).lastrowid
        for row in rows:
            _write_train_image(row["class_id"], row["stored_name"], row["image_png"])

        _status(
            state="running",
            message=f"Retraining started ({reason}) with {len(rows)} new uploads.",
            epochs=epochs,
            reason=reason,
            started_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
        )

        result = run_training(epochs=epochs, retrain=True)
        promoted = result["promoted"]
        new = result["metrics"]
        old = result["previous_metrics"] or {}
        if promoted:
            reload_model()
            message = "New model promoted and reloaded."
        else:
            message = "Candidate model was worse on the test set; kept the current model."

        with connect() as conn:
            if promoted:
                conn.executemany(
                    "UPDATE uploads SET status = 'used', retrain_run_id = ? WHERE id = ?",
                    [(run_id, row["id"]) for row in rows],
                )
            conn.execute(
                """UPDATE retrain_runs SET finished_at = ?, state = 'complete', message = ?, promoted = ?,
                          accuracy = ?, f1_macro = ?, previous_accuracy = ?, previous_f1_macro = ?
                   WHERE id = ?""",
                (now(), message, int(promoted), new.get("accuracy"), new.get("f1_macro"),
                 old.get("accuracy"), old.get("f1_macro"), run_id),
            )

        return _status(
            state="complete",
            message=message,
            promoted=promoted,
            uploads_used=len(rows),
            accuracy=new.get("accuracy"),
            f1_macro=new.get("f1_macro"),
            previous_accuracy=old.get("accuracy"),
            previous_f1_macro=old.get("f1_macro"),
            model_path=result["model_path"],
        )
    except Exception as exc:
        if run_id is not None:
            with connect() as conn:
                conn.execute(
                    "UPDATE retrain_runs SET finished_at = ?, state = 'error', message = ? WHERE id = ?",
                    (now(), str(exc), run_id),
                )
        return _status(state="error", message=str(exc))
    finally:
        _lock.release()


def trigger_retrain_async(epochs: int = RETRAIN_EPOCHS, reason: str = "manual") -> dict:
    """Start retraining in a background thread so the API keeps answering."""
    status = get_status()
    if status.get("state") == "running":
        # If a container crashed during retraining the status stays "running",
        # so only trust it for one hour.
        started = datetime.fromisoformat(status["updated_at"])
        if (datetime.now(timezone.utc) - started).total_seconds() < 3600:
            return status

    _status(state="running", message=f"Retrain job queued ({reason}).", reason=reason)
    thread = threading.Thread(target=retrain_job, kwargs={"epochs": epochs, "reason": reason}, daemon=True)
    thread.start()
    return get_status()


def maybe_auto_retrain() -> dict | None:
    """Automatic trigger: start retraining when AUTO_RETRAIN is on and enough new images are pending."""
    if AUTO_RETRAIN and should_retrain():
        return trigger_retrain_async(reason=f"auto: >= {RETRAIN_MIN_NEW_SAMPLES} new uploads")
    return None

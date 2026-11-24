"""Upload ingestion and retraining trigger (manual button or sample threshold)."""

from __future__ import annotations

import json
import shutil
import threading
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.config import (
    AUTO_RETRAIN,
    LOGS_DIR,
    RETRAIN_EPOCHS,
    RETRAIN_MIN_NEW_SAMPLES,
    RETRAIN_STATUS_PATH,
    TRAIN_DIR,
    UPLOAD_DIR,
    USED_UPLOAD_DIR,
)
from src.prediction import reload_model
from src.preprocessing import CLASS_FOLDERS, IMAGE_EXTS, ensure_class_folders, load_bytes, resolve_class_id
from src.train import run_training

_lock = threading.Lock()


def _status(**kwargs) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    current = {}
    if RETRAIN_STATUS_PATH.exists():
        current = json.loads(RETRAIN_STATUS_PATH.read_text(encoding="utf-8"))
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
    except Exception:  # noqa: BLE001
        return False


def _store(data: bytes, class_id: int, dest_root: Path, stem: str) -> None:
    """Preprocess once and store a 128x128 PNG in data/train and in the upload queue."""
    image = load_bytes(data)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"upload_{stamp}_{Path(stem).stem[:40]}.png"
    for root in (dest_root, UPLOAD_DIR):
        folder = root / CLASS_FOLDERS[class_id]
        folder.mkdir(parents=True, exist_ok=True)
        image.save(folder / name)


def save_labelled_files(files: list[tuple[str, bytes]], class_id: int, dest_root: Path = TRAIN_DIR) -> dict:
    ensure_class_folders(dest_root)
    saved = skipped = 0
    for name, data in files:
        if Path(name).suffix.lower() not in IMAGE_EXTS or not _is_valid_image(data):
            skipped += 1
            continue
        _store(data, class_id, dest_root, name)
        saved += 1
    return {"saved": saved, "skipped": skipped}


def ingest_zip(data: bytes, dest_root: Path = TRAIN_DIR) -> dict:
    """Accept a zip whose folders name the class (yes/no, dyslexic/non_dyslexic, 1/0).
    Images are read straight from the archive; nothing is extracted to disk."""
    ensure_class_folders(dest_root)
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
            _store(blob, class_id, dest_root, path.name)
            saved += 1
    return {"saved": saved, "skipped": skipped}


def count_uploads() -> int:
    if not UPLOAD_DIR.exists():
        return 0
    return sum(1 for p in UPLOAD_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def should_retrain() -> bool:
    return count_uploads() >= RETRAIN_MIN_NEW_SAMPLES


def _archive_uploads() -> int:
    """Move consumed uploads out of the pending queue so the trigger resets.
    The images stay in data/train, so later retrains still learn from them."""
    n = count_uploads()
    if n and UPLOAD_DIR.exists():
        dest = USED_UPLOAD_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(UPLOAD_DIR), str(dest))
    return n


def retrain_job(epochs: int = RETRAIN_EPOCHS, reason: str = "manual") -> dict:
    if not _lock.acquire(blocking=False):
        return _status(state="busy", message="Retraining already in progress.")
    try:
        pending = count_uploads()
        _status(
            state="running",
            message=f"Retraining started ({reason}) with {pending} new uploads.",
            epochs=epochs,
            reason=reason,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        result = run_training(epochs=epochs, retrain=True)
        if result["promoted"]:
            reload_model()
            _archive_uploads()
        new, old = result["metrics"], result["previous_metrics"] or {}
        return _status(
            state="complete",
            message=(
                "New model promoted and reloaded."
                if result["promoted"]
                else "Candidate model was worse on the test set; kept the current model."
            ),
            promoted=result["promoted"],
            uploads_used=pending,
            accuracy=new.get("accuracy"),
            f1_macro=new.get("f1_macro"),
            previous_accuracy=old.get("accuracy"),
            previous_f1_macro=old.get("f1_macro"),
            model_path=result["model_path"],
        )
    except Exception as exc:  # noqa: BLE001 — surface in API status
        return _status(state="error", message=str(exc))
    finally:
        _lock.release()


def _is_running() -> bool:
    status = get_status()
    if status.get("state") != "running":
        return False
    # A container that died mid-job leaves "running" behind; ignore after 1 hour.
    try:
        updated = datetime.fromisoformat(status["updated_at"])
        return (datetime.now(timezone.utc) - updated).total_seconds() < 3600
    except (KeyError, ValueError):
        return True


def trigger_retrain_async(epochs: int = RETRAIN_EPOCHS, reason: str = "manual") -> dict:
    if _is_running():
        return get_status()
    _status(state="running", message=f"Retrain job queued ({reason}).", reason=reason)
    thread = threading.Thread(target=retrain_job, kwargs={"epochs": epochs, "reason": reason}, daemon=True)
    thread.start()
    return get_status()


def maybe_auto_retrain() -> dict | None:
    """Automatic trigger: fires when AUTO_RETRAIN is on and enough new data arrived."""
    if AUTO_RETRAIN and should_retrain():
        return trigger_retrain_async(reason=f"auto: >= {RETRAIN_MIN_NEW_SAMPLES} new uploads")
    return None

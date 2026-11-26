"""SQLite database for uploaded training images and retraining runs.

uploads       - one row per uploaded image (the preprocessed PNG is stored in
                image_png). status is 'pending' until a retrained model uses it,
                then 'used'.
retrain_runs  - one row per retraining job with the metrics before and after.

sqlite3 comes with Python, so nothing extra needs to be installed.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS uploads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    original_name   TEXT    NOT NULL,
    stored_name     TEXT    NOT NULL UNIQUE,
    class_id        INTEGER NOT NULL,
    class_label     TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    image_png       BLOB    NOT NULL,
    uploaded_at     TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'pending',
    retrain_run_id  INTEGER
);

CREATE TABLE IF NOT EXISTS retrain_runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    reason            TEXT    NOT NULL,
    epochs            INTEGER NOT NULL,
    new_images        INTEGER NOT NULL,
    started_at        TEXT    NOT NULL,
    finished_at       TEXT,
    state             TEXT    NOT NULL DEFAULT 'running',
    promoted          INTEGER,
    accuracy          REAL,
    f1_macro          REAL,
    previous_accuracy REAL,
    previous_f1_macro REAL,
    message           TEXT
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    """Open the database, create the tables if needed, and commit at the end."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # timeout: wait if another API container is writing at the same time
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()

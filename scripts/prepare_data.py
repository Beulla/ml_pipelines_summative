"""Data acquisition: download the children's handwriting dataset and build train/test folders.

Source: "Multilingual AI-Based Dyslexia Detection", Mendeley Data, CC BY 4.0,
DOI 10.17632/b37s8b94nv.1 — 852 character-level images (English letters, Urdu
letters, digits) from 100 children aged 6-10; 426 dyslexic / 426 non-dyslexic.

Output (preprocessed 128x128 grayscale PNGs):
    data/train/non_dyslexic/*.png   data/train/dyslexic/*.png
    data/test/non_dyslexic/*.png    data/test/dyslexic/*.png
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import (  # noqa: E402
    CLASS_NAMES,
    DATASET_FOLDER_LABELS,
    DATASET_ZIP_SHA256,
    DATASET_ZIP_URL,
    RANDOM_STATE,
    RAW_DIR,
    TEST_DIR,
    TEST_FRACTION,
    TRAIN_DIR,
)
from src.preprocessing import CLASS_FOLDERS, IMAGE_EXTS, ensure_class_folders, preprocess_image  # noqa: E402

ZIP_PATH = RAW_DIR / "dyslexia_handwriting.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and sha256(ZIP_PATH) == DATASET_ZIP_SHA256:
        print("Using cached", ZIP_PATH)
        return ZIP_PATH
    print("Downloading dataset (56 MB) from Mendeley Data ...")
    urllib.request.urlretrieve(DATASET_ZIP_URL, ZIP_PATH)
    digest = sha256(ZIP_PATH)
    if digest != DATASET_ZIP_SHA256:
        raise RuntimeError(f"Checksum mismatch for {ZIP_PATH}: {digest}")
    return ZIP_PATH


def list_members(zf: zipfile.ZipFile) -> list[tuple[str, int]]:
    """(member name, label) for every image under Dataset/Yes and Dataset/No."""
    items = []
    for name in zf.namelist():
        parts = Path(name).parts
        if len(parts) < 2 or Path(name).suffix.lower() not in IMAGE_EXTS:
            continue
        label = DATASET_FOLDER_LABELS.get(parts[-2])
        if label is not None:
            items.append((name, label))
    return sorted(items)


def main():
    zip_path = download()
    with zipfile.ZipFile(zip_path) as zf:
        items = list_members(zf)
        names = [n for n, _ in items]
        labels = [y for _, y in items]
        train_names, test_names = train_test_split(
            names, test_size=TEST_FRACTION, stratify=labels, random_state=RANDOM_STATE
        )
        label_of = dict(items)
        for root in (TRAIN_DIR, TEST_DIR):
            ensure_class_folders(root)
            for folder in CLASS_FOLDERS.values():
                for old in (root / folder).glob("*.png"):
                    old.unlink()
        for split_root, split_names in ((TRAIN_DIR, train_names), (TEST_DIR, test_names)):
            for i, name in enumerate(sorted(split_names)):
                label = label_of[name]
                with Image.open(BytesIO(zf.read(name))) as img:
                    processed = preprocess_image(img)
                out = split_root / CLASS_FOLDERS[label] / f"{CLASS_FOLDERS[label]}_{i:04d}.png"
                processed.save(out, optimize=True)
            print(f"Wrote {len(split_names)} images -> {split_root}")
    print("Classes:", CLASS_NAMES)


if __name__ == "__main__":
    main()

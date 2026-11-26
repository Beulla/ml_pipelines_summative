"""Download the handwriting dataset and create the train/test folders.

Dataset: "Multilingual AI-Based Dyslexia Detection", Mendeley Data (CC BY 4.0),
DOI 10.17632/b37s8b94nv.1. 852 images, 426 dyslexic and 426 non-dyslexic.
Images are preprocessed to 128x128 and saved in data/train/ and data/test/.
"""

import hashlib
import sys
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (
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
from src.preprocessing import CLASS_FOLDERS, IMAGE_EXTS, ensure_class_folders, preprocess_image

ZIP_PATH = RAW_DIR / "dyslexia_handwriting.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. Download the zip (skipped if it is already downloaded and the checksum matches)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and sha256(ZIP_PATH) == DATASET_ZIP_SHA256:
        print("Using cached", ZIP_PATH)
    else:
        print("Downloading dataset (56 MB) from Mendeley Data ...")
        urllib.request.urlretrieve(DATASET_ZIP_URL, ZIP_PATH)
        digest = sha256(ZIP_PATH)
        if digest != DATASET_ZIP_SHA256:
            raise RuntimeError(f"Checksum mismatch for {ZIP_PATH}: {digest}")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        # 2. Find every image in the Yes / No folders
        items = []
        for name in zf.namelist():
            parts = Path(name).parts
            if len(parts) < 2 or Path(name).suffix.lower() not in IMAGE_EXTS:
                continue
            label = DATASET_FOLDER_LABELS.get(parts[-2])
            if label is not None:
                items.append((name, label))
        items.sort()

        # 3. Stratified 80/20 split, then preprocess and save each image
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

"""Image preprocessing and handwriting features.

Every image (train, test, upload, prediction) goes through preprocess_image:
fix phone rotation -> grayscale -> auto-contrast -> pad to square -> 128x128.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.config import (
    CLASS_FOLDER_ALIASES,
    CLASS_NAMES,
    FEATURE_STATS_PATH,
    IMG_SIZE,
    NUM_CLASSES,
    TEST_DIR,
    TRAIN_DIR,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CLASS_FOLDERS = {0: "non_dyslexic", 1: "dyslexic"}


def ensure_class_folders(root: Path) -> None:
    for idx in range(NUM_CLASSES):
        (root / CLASS_FOLDERS[idx]).mkdir(parents=True, exist_ok=True)


def resolve_class_id(name: str) -> int | None:
    key = name.strip().lower().replace(" ", "_")
    return CLASS_FOLDER_ALIASES.get(key)


def iter_labelled_images(root: Path):
    if not root.exists():
        return
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = resolve_class_id(class_dir.name)
        if class_id is None:
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                yield path, class_id


def dataset_counts(root: Path) -> dict[str, int]:
    counts = {CLASS_NAMES[i]: 0 for i in range(NUM_CLASSES)}
    for _, class_id in iter_labelled_images(root):
        counts[CLASS_NAMES[class_id]] += 1
    return counts


def preprocess_image(image: Image.Image, size: int = IMG_SIZE) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("L")
    # Auto-contrast so differences in paper colour and lighting don't matter.
    image = ImageOps.autocontrast(image, cutoff=1)
    w, h = image.size
    if w != h:
        # Pad with the paper colour so the letter keeps its shape.
        paper = int(np.percentile(np.asarray(image), 90))
        side = max(w, h)
        canvas = Image.new("L", (side, side), paper)
        canvas.paste(image, ((side - w) // 2, (side - h) // 2))
        image = canvas
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def load_image(path: Path, size: int = IMG_SIZE) -> Image.Image:
    with Image.open(path) as img:
        return preprocess_image(img, size)


def load_bytes(data: bytes, size: int = IMG_SIZE) -> Image.Image:
    with Image.open(BytesIO(data)) as img:
        return preprocess_image(img, size)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels darker than halfway between the paper (median) and the ink (darkest 1%)."""
    paper = float(np.median(gray))
    ink = float(np.percentile(gray, 1))
    return gray < (paper + ink) / 2


def extract_image_features(image: Image.Image | np.ndarray) -> dict[str, float]:
    """The 3 handwriting features used in the visualisations:
    ink_coverage    - share of the image covered by pencil
    stroke_darkness - ink vs paper contrast (roughly pen pressure)
    stroke_thinness - edge pixels per ink pixel (low = thick or retraced strokes)
    """
    gray = np.asarray(image, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    ink = ink_mask(gray)
    ink_px = int(ink.sum())
    paper_vals = gray[~ink]
    ink_vals = gray[ink]

    # Edge pixels = ink pixels that touch at least one non-ink pixel.
    padded = np.pad(ink, 1, constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary_px = int((ink & ~interior).sum())

    return {
        "ink_coverage": ink_px / gray.size,
        "stroke_darkness": float((paper_vals.mean() - ink_vals.mean()) / 255.0) if ink_px and paper_vals.size else 0.0,
        "stroke_thinness": boundary_px / ink_px if ink_px else 0.0,
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
    }


FEATURE_COLUMNS = ["ink_coverage", "stroke_darkness", "stroke_thinness", "brightness", "contrast"]


def build_feature_table(roots: list[Path] | None = None) -> pd.DataFrame:
    roots = roots or [TRAIN_DIR, TEST_DIR]
    rows = []
    for root in roots:
        for path, class_id in iter_labelled_images(root):
            feats = extract_image_features(load_image(path))
            feats.update(
                {
                    "path": str(path.relative_to(root.parent)),
                    "split": root.name,
                    "label": int(class_id),
                    "label_name": CLASS_NAMES[class_id],
                }
            )
            rows.append(feats)
    df = pd.DataFrame(rows)
    if not df.empty:
        FEATURE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(FEATURE_STATS_PATH, index=False)
    return df

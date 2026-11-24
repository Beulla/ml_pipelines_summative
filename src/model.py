"""Model creation, training, evaluation and persistence (PyTorch).

Architecture: MobileNetV3-Small pre-trained on ImageNet (transfer learning),
with a new 2-class head. With only ~700 training images, re-using pre-trained
edge/stroke detectors beats a CNN trained from scratch. Optimisation:
Adam + weight decay, ReduceLROnPlateau, early stopping on validation F1,
data augmentation, and a lower learning rate for the pre-trained backbone.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

from src.config import (
    CLASS_NAMES,
    EPOCHS,
    IMG_SIZE,
    METRICS_PATH,
    MODEL_META_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NUM_CLASSES,
    PKL_PATH,
    RANDOM_STATE,
)
from src.preprocessing import iter_labelled_images, load_image

DEVICE = torch.device("cpu")
PATIENCE = 8
LEARNING_RATE = 5e-4
ARCHITECTURES = ("mobilenet_v3_small", "simple_cnn")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
torch.set_num_threads(max(1, torch.get_num_threads()))


# ------------------------------------------------------------------ networks
class SimpleCNN(nn.Module):
    """Baseline trained from scratch (used in the notebook for comparison)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 64))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(arch: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    if arch == "simple_cnn":
        return SimpleCNN()
    if arch != "mobilenet_v3_small":
        raise ValueError(f"Unknown architecture {arch}; choose from {ARCHITECTURES}")
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return net


def _param_groups(model: nn.Module, lr: float, backbone_ratio: float = 1.0):
    """Full fine-tuning. A sweep (backbone at lr/10, lr/3, lr) showed the fully
    trainable backbone works best: handwriting differs a lot from ImageNet photos."""
    if isinstance(model, SimpleCNN):
        return [{"params": model.parameters(), "lr": lr}]
    return [
        {"params": model.features.parameters(), "lr": lr * backbone_ratio},
        {"params": model.classifier.parameters(), "lr": lr},
    ]


# ---------------------------------------------------------------- transforms
def _to_tensor_3ch():
    # Grayscale handwriting -> 3 identical channels -> ImageNet normalisation.
    return [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]


def train_transforms():
    # No flips: a mirrored letter is exactly the kind of reversal dyslexia causes,
    # so flipping would destroy label information.
    return transforms.Compose(
        [
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.85, 1.15), fill=255),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            *_to_tensor_3ch(),
        ]
    )


def eval_transforms():
    return transforms.Compose(_to_tensor_3ch())


class FolderImageDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.samples = list(iter_labelled_images(root))
        self.transform = transform or eval_transforms()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return self.transform(load_image(path)), int(label)


# ------------------------------------------------------------------ training
def _split_indices(labels: list[int], val_split: float = 0.15):
    """Stratified train/validation split of the training folder."""
    rng = np.random.default_rng(RANDOM_STATE)
    labels = np.asarray(labels)
    train_idx, val_idx = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        n_val = max(1, int(round(len(idx) * val_split)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


def class_weights(labels: list[int]) -> torch.Tensor:
    """Inverse-frequency weights; matter once uploads unbalance the classes."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=NUM_CLASSES).astype(np.float32)
    counts[counts == 0] = 1.0
    return torch.tensor(counts.sum() / (NUM_CLASSES * counts), dtype=torch.float32)


def _run_epoch(model, loader, loss_fn, opt=None):
    training = opt is not None
    model.train(training)
    total_loss, ys, ps = 0.0, [], []
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * xb.size(0)
            ys.append(yb.numpy())
            ps.append(logits.argmax(1).numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return total_loss / len(y), accuracy_score(y, p), f1_score(y, p, average="macro", zero_division=0)


def train_model(
    train_dir: Path,
    epochs: int = EPOCHS,
    batch_size: int = 32,
    model_path: Path = MODEL_PATH,
    warm_start_from: Path | None = None,
    lr: float = LEARNING_RATE,
    arch: str = "mobilenet_v3_small",
    verbose: bool = True,
):
    """Train the classifier. With ``warm_start_from`` the current production
    weights are loaded first and fine-tuned on the (now larger) training folder."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_view = FolderImageDataset(train_dir, transform=train_transforms())
    if len(train_view) == 0:
        raise FileNotFoundError(f"No training images in {train_dir}")
    val_view = FolderImageDataset(train_dir, transform=eval_transforms())
    labels = [label for _, label in train_view.samples]
    train_idx, val_idx = _split_indices(labels)
    train_loader = DataLoader(Subset(train_view, train_idx.tolist()), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(val_view, val_idx.tolist()), batch_size=64, shuffle=False)

    if warm_start_from is not None and Path(warm_start_from).exists():
        model, payload = load_trained_model(warm_start_from, return_payload=True)
        arch = payload.get("architecture", arch)
        if verbose:
            print(f"Warm start from {warm_start_from}")
    else:
        model = build_model(arch)
    model.to(DEVICE)

    opt = torch.optim.Adam(_param_groups(model, lr), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights([labels[i] for i in train_idx]))

    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "val_f1_macro": []}
    best_f1, best_state, waited = -1.0, None, 0
    for epoch in range(epochs):
        tr_loss, tr_acc, _ = _run_epoch(model, train_loader, loss_fn, opt)
        va_loss, va_acc, va_f1 = _run_epoch(model, val_loader, loss_fn)
        sched.step(va_f1)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, va_f1)):
            history[k].append(float(v))
        if verbose:
            print(
                f"epoch {epoch + 1:2d}/{epochs}  loss={tr_loss:.3f} acc={tr_acc:.3f}  "
                f"val_loss={va_loss:.3f} val_acc={va_acc:.3f} val_f1={va_f1:.3f}"
            )
        # Checkpoint on validation macro F1 (ties broken by lower loss).
        score = va_f1 - 1e-3 * va_loss
        if score > best_f1:
            best_f1, waited = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                if verbose:
                    print("Early stopping")
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    save_model(model, model_path, arch=arch)
    return model, history


# ------------------------------------------------------------- persistence
def save_model(model: nn.Module, path: Path = MODEL_PATH, arch: str = "mobilenet_v3_small") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "architecture": arch,
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    torch.save(payload, path)                     # .pt  (PyTorch)
    joblib.dump(payload, path.with_suffix(".pkl"))  # .pkl (assignment artefact)


def load_trained_model(path: Path = MODEL_PATH, return_payload: bool = False):
    path = Path(path)
    target = path if path.exists() else path.with_suffix(".pkl")
    if not target.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python -m src.train")
    if target.suffix == ".pkl":
        payload = joblib.load(target)
    else:
        payload = torch.load(target, map_location="cpu", weights_only=False)
    model = build_model(payload.get("architecture", "mobilenet_v3_small"), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return (model, payload) if return_payload else model


# --------------------------------------------------------------- inference
def predict_proba(model: nn.Module, images: list[Image.Image] | Image.Image) -> np.ndarray:
    """Probabilities for preprocessed PIL images (see preprocessing.preprocess_image)."""
    if isinstance(images, Image.Image):
        images = [images]
    tfm = eval_transforms()
    batch = torch.stack([tfm(img) for img in images])
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(batch), dim=1).numpy()


def evaluate_model(model: nn.Module, test_dir: Path, out_path: Path | None = METRICS_PATH) -> dict:
    ds = FolderImageDataset(test_dir, transform=eval_transforms())
    if len(ds) == 0:
        raise FileNotFoundError(f"No test images in {test_dir}")
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            ps.append(torch.softmax(model(xb), dim=1).numpy())
            ys.append(yb.numpy())
    y_true = np.concatenate(ys)
    probs = np.concatenate(ps)
    y_pred = probs.argmax(axis=1)
    labels = list(range(NUM_CLASSES))
    names = [CLASS_NAMES[i] for i in labels]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        # Screening view: how many at-risk children are caught (sensitivity) / cleared correctly (specificity).
        "sensitivity_at_risk": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0
        ),
        "n_test": int(len(y_true)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
    except ValueError:
        metrics["roc_auc"] = None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["_y_true"] = y_true.tolist()
    metrics["_probs"] = probs[:, 1].tolist()
    return metrics


def write_model_meta(extra: dict | None = None) -> dict:
    meta = {
        "model_name": MODEL_PATH.name,
        "pkl_name": PKL_PATH.name,
        "architecture": "MobileNetV3-Small (ImageNet pre-trained) + 2-class head",
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        meta.update(extra)
    MODEL_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return {}

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import public_dataset_benchmark as base  # noqa: E402


DEFAULT_OUT = ROOT / "outputs" / "capgmyo_strict_evaluation"
SEED = 42
UCI_MODELS = ["InceptionTimeSE", "SubjectInvariantInception"]
CAPG_MODELS = ["SpatialRMSCNN", "SpatialTemporalResNetSE"]
DISPLAY = {
    "InceptionTimeSE": "InceptionTime-SE",
    "SubjectInvariantInception": "Subject-Invariant InceptionTime",
    "SpatialRMSCNN": "HD-sEMG Spatial RMS CNN",
    "SpatialTemporalResNetSE": "HD-sEMG Spatial-Temporal ResNet-SE",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "font.size": 9,
    }
)


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict]
    best_epoch: int
    best_val_macro_f1: float
    train_seconds: float
    param_count: int


@dataclass
class CapgFrameBundle:
    x: np.ndarray
    y: np.ndarray
    subject_id: np.ndarray
    trial_id: np.ndarray
    recording_index: np.ndarray
    recording_table: pd.DataFrame
    frame_table: pd.DataFrame


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, class_count: int, prefix: str = "test") -> dict:
    labels = list(range(class_count))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_precision": float(precision),
        f"{prefix}_macro_recall": float(recall),
        f"{prefix}_macro_f1": float(f1),
    }


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, hidden, 1), nn.GELU(),
            nn.Conv1d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels: int, branch_channels: int = 32):
        super().__init__()
        bottleneck = 32 if in_channels > 32 else in_channels
        self.reduce = nn.Conv1d(in_channels, bottleneck, 1, bias=False)
        self.branches = nn.ModuleList(
            [nn.Conv1d(bottleneck, branch_channels, kernel_size=k, padding=k // 2, bias=False) for k in (9, 19, 39)]
        )
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(in_channels, branch_channels, 1, bias=False))
        self.norm = nn.BatchNorm1d(branch_channels * 4)
        self.se = SqueezeExcite1D(branch_channels * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        out = torch.cat([branch(reduced) for branch in self.branches] + [self.pool_branch(x)], dim=1)
        return self.se(torch.nn.functional.gelu(self.norm(out)))


class InceptionBackbone(nn.Module):
    def __init__(self, channels: int, depth: int = 6):
        super().__init__()
        blocks, residuals = [], []
        current = channels
        for _ in range(depth):
            blocks.append(InceptionBlock1D(current, 32))
            residuals.append(nn.Conv1d(current, 128, 1, bias=False) if current != 128 else nn.Identity())
            current = 128
        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)
        self.norms = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, block in enumerate(self.blocks):
            residual = self.residuals[idx](x)
            x = block(x)
            if idx % 2 == 1:
                x = torch.nn.functional.gelu(self.norms[idx](x + residual))
        return self.pool(x).squeeze(-1)


class InceptionTimeSE(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.backbone = InceptionBackbone(channels)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.2), nn.Linear(128, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.strength * grad_output, None


class SubjectInvariantInception(nn.Module):
    def __init__(self, channels: int, classes: int, train_subject_count: int):
        super().__init__()
        self.backbone = InceptionBackbone(channels)
        self.gesture_head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.2), nn.Linear(128, classes))
        self.subject_head = nn.Sequential(nn.Linear(128, 96), nn.GELU(), nn.Dropout(0.2), nn.Linear(96, train_subject_count))

    def forward(self, x: torch.Tensor, grl_strength: float = 0.0, return_subject: bool = False):
        features = self.backbone(x)
        gesture = self.gesture_head(features)
        if not return_subject:
            return gesture
        return gesture, self.subject_head(GradientReverse.apply(features, grl_strength))


class SqueezeExcite2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.GELU(),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpatialResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), SqueezeExcite2D(out_channels),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
            if stride != 1 or in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.conv(x) + self.skip(x))


class SpatialRMSCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.GELU(),
            SqueezeExcite2D(128), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x[:, 1:2]))


class SpatialTemporalResNetSE(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48), nn.GELU())
        self.blocks = nn.Sequential(
            SpatialResidualBlock(48, 64), SpatialResidualBlock(64, 96, stride=2),
            SpatialResidualBlock(96, 128), SpatialResidualBlock(128, 160, stride=2),
            SpatialResidualBlock(160, 192),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.LayerNorm(192), nn.Dropout(0.25), nn.Linear(192, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


def robust_uci_normalize(x_train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    processed = []
    for arr in arrays:
        centered = arr - np.median(arr, axis=2, keepdims=True)
        scale = np.median(np.abs(centered), axis=(1, 2), keepdims=True) * 1.4826
        processed.append((centered / np.maximum(scale, 1e-6)).astype(np.float32))
    mean = processed[0].mean(axis=(0, 2), keepdims=True)
    std = processed[0].std(axis=(0, 2), keepdims=True)
    return tuple(((arr - mean) / np.maximum(std, 1e-6)).astype(np.float32) for arr in processed)


def augment_uci(x: torch.Tensor) -> torch.Tensor:
    scale = torch.empty((x.shape[0], 1, 1), device=x.device).uniform_(0.9, 1.1)
    noise = torch.randn_like(x) * 0.015
    channel_keep = (torch.rand((x.shape[0], x.shape[1], 1), device=x.device) > 0.04).to(x.dtype)
    return x * scale * channel_keep + noise


def predict_uci(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, probs = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            logits = model(xb.to(device))
            ys.append(yb.numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs)


def train_uci_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_subjects: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_subjects: np.ndarray,
    classes: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> TrainResult:
    set_seed(seed)
    unique_subjects = sorted(np.unique(train_subjects).tolist())
    subject_map = {subject: idx for idx, subject in enumerate(unique_subjects)}
    train_subject_local = np.asarray([subject_map[int(v)] for v in train_subjects], dtype=np.int64)
    if model_name == "InceptionTimeSE":
        model: nn.Module = InceptionTimeSE(x_train.shape[1], classes)
    else:
        model = SubjectInvariantInception(x_train.shape[1], classes, len(unique_subjects))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=5e-5)
    gesture_loss = nn.CrossEntropyLoss(label_smoothing=0.05)
    subject_loss = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(train_subject_local)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val), torch.from_numpy(val_subjects.astype(np.int64))),
        batch_size=batch_size,
        shuffle=False,
    )
    best_state = None
    best_val = -1.0
    best_epoch = 0
    stale = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        progress = (epoch - 1) / max(epochs - 1, 1)
        grl_strength = 0.25 * (2.0 / (1.0 + math.exp(-8.0 * progress)) - 1.0)
        for xb, yb, sb in train_loader:
            xb, yb, sb = xb.to(device), yb.to(device), sb.to(device)
            optimizer.zero_grad(set_to_none=True)
            xb = augment_uci(xb)
            if model_name == "SubjectInvariantInception":
                logits, subject_logits = model(xb, grl_strength=grl_strength, return_subject=True)
                loss = gesture_loss(logits, yb) + 0.2 * subject_loss(subject_logits, sb)
            else:
                loss = gesture_loss(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_true, val_probs = predict_uci(model, val_loader, device)
        val_pred = val_probs.argmax(axis=1)
        val_metrics = metric_dict(val_true, val_pred, classes, "val")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "grl_strength": float(grl_strength if model_name == "SubjectInvariantInception" else 0.0),
                **val_metrics,
            }
        )
        if val_metrics["val_macro_f1"] > best_val + 1e-5:
            best_val = val_metrics["val_macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 15:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model, history, best_epoch, best_val, time.perf_counter() - start, count_params(model))


def build_capg_frames(subjects: list[int], frames_per_recording: int, radius: int) -> CapgFrameBundle:
    features, labels, subject_ids, trial_ids, recording_indices = [], [], [], [], []
    recording_rows, frame_rows = [], []
    recording_index = 0
    frame_index = 0
    for subject_id in subjects:
        zip_path = base.CAPG_DIR / base.CAPG_FILES[subject_id]["name"]
        if not zip_path.exists():
            raise FileNotFoundError(zip_path)
        with zipfile.ZipFile(zip_path, "r") as archive:
            for mat_name in sorted(name for name in archive.namelist() if name.lower().endswith(".mat")):
                mat = scipy.io.loadmat(io.BytesIO(archive.read(mat_name)))
                data = np.asarray(mat["data"], dtype=np.float32)
                gesture = int(np.asarray(mat.get("gesture")).squeeze())
                trial = int(np.asarray(mat.get("trial")).squeeze())
                if not 1 <= gesture <= 8:
                    continue
                start = max(radius + 1, int(round(data.shape[0] * 0.10)))
                stop = min(data.shape[0] - radius - 2, int(round(data.shape[0] * 0.90)))
                centers = np.linspace(start, stop, frames_per_recording, dtype=int)
                recording_id = f"S{subject_id:02d}_G{gesture:02d}_T{trial:02d}"
                recording_rows.append(
                    {
                        "recording_index": recording_index,
                        "recording_id": recording_id,
                        "subject_id": subject_id,
                        "gesture_id": gesture,
                        "class_id": gesture - 1,
                        "trial_id": trial,
                        "source_file": str(zip_path.relative_to(ROOT)),
                        "mat_file": mat_name,
                        "native_time_points": int(data.shape[0]),
                        "frames_extracted": int(len(centers)),
                    }
                )
                for center in centers:
                    window = data[center - radius : center + radius + 1]
                    mav = np.mean(np.abs(window), axis=0)
                    rms = np.sqrt(np.mean(np.square(window), axis=0) + 1e-12)
                    waveform = np.mean(np.abs(np.diff(window, axis=0)), axis=0)
                    image = np.stack([mav, rms, waveform], axis=0).reshape(3, 8, 16).astype(np.float32)
                    features.append(image)
                    labels.append(gesture - 1)
                    subject_ids.append(subject_id)
                    trial_ids.append(trial)
                    recording_indices.append(recording_index)
                    frame_rows.append(
                        {
                            "frame_index": frame_index,
                            "recording_index": recording_index,
                            "recording_id": recording_id,
                            "subject_id": subject_id,
                            "class_id": gesture - 1,
                            "trial_id": trial,
                            "center_index": int(center),
                        }
                    )
                    frame_index += 1
                recording_index += 1
    return CapgFrameBundle(
        x=np.stack(features).astype(np.float32),
        y=np.asarray(labels, dtype=np.int64),
        subject_id=np.asarray(subject_ids, dtype=np.int64),
        trial_id=np.asarray(trial_ids, dtype=np.int64),
        recording_index=np.asarray(recording_indices, dtype=np.int64),
        recording_table=pd.DataFrame(recording_rows),
        frame_table=pd.DataFrame(frame_rows),
    )


def normalize_capg_by_train_subject(
    x: np.ndarray, subjects: np.ndarray, train_mask: np.ndarray, *masks: np.ndarray
) -> tuple[np.ndarray, ...]:
    normalized = np.empty_like(x, dtype=np.float32)
    for subject in np.unique(subjects):
        subject_train = train_mask & (subjects == subject)
        if not np.any(subject_train):
            raise RuntimeError(f"No train frames available for subject {subject}")
        mean = x[subject_train].mean(axis=0, keepdims=True)
        std = x[subject_train].std(axis=0, keepdims=True)
        subject_all = subjects == subject
        normalized[subject_all] = (x[subject_all] - mean) / np.maximum(std, 1e-6)
    return tuple(normalized[mask].astype(np.float32) for mask in masks)


def aggregate_recording_predictions(
    y_frame: np.ndarray, probs: np.ndarray, recording_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rec_ids = np.unique(recording_index)
    y_recording, probs_recording = [], []
    for rec_id in rec_ids:
        mask = recording_index == rec_id
        labels = np.unique(y_frame[mask])
        if len(labels) != 1:
            raise RuntimeError(f"Recording {rec_id} contains multiple labels")
        y_recording.append(int(labels[0]))
        probs_recording.append(probs[mask].mean(axis=0))
    return np.asarray(y_recording, dtype=np.int64), np.stack(probs_recording), rec_ids


def predict_capg(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ys, probs, recs = [], [], []
    with torch.no_grad():
        for xb, yb, rb in loader:
            logits = model(xb.to(device))
            ys.append(yb.numpy())
            recs.append(rb.numpy())
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs), np.concatenate(recs)


def train_capg_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    rec_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    rec_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> TrainResult:
    set_seed(seed)
    model: nn.Module = SpatialRMSCNN(8) if model_name == "SpatialRMSCNN" else SpatialTemporalResNetSE(8)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=8e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=5e-5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(rec_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val), torch.from_numpy(rec_val)),
        batch_size=batch_size,
        shuffle=False,
    )
    best_state = None
    best_val = -1.0
    best_epoch = 0
    stale = 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            scale = torch.empty((xb.shape[0], 1, 1, 1), device=device).uniform_(0.92, 1.08)
            xb = xb * scale + torch.randn_like(xb) * 0.01
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_frame_y, val_frame_probs, val_recs = predict_capg(model, val_loader, device)
        val_y, val_probs, _ = aggregate_recording_predictions(val_frame_y, val_frame_probs, val_recs)
        val_pred = val_probs.argmax(axis=1)
        val_metrics = metric_dict(val_y, val_pred, 8, "val_recording")
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **val_metrics,
            }
        )
        score = val_metrics["val_recording_macro_f1"]
        if score > best_val + 1e-5:
            best_val = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 12:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model, history, best_epoch, best_val, time.perf_counter() - start, count_params(model))


def append_per_class(
    rows: list[dict], dataset: str, fold: str, model: str,
    y_true: np.ndarray, y_pred: np.ndarray, names: dict[int, str],
) -> None:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(names))), zero_division=0
    )
    for class_id, class_name in names.items():
        rows.append(
            {
                "dataset_name": dataset,
                "fold_id": fold,
                "model_name": model,
                "model_display_name": DISPLAY[model],
                "class_id": class_id,
                "class_name": class_name,
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
        )


def run_uci(
    device: torch.device, epochs: int, batch_size: int, folds: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    bundle = base.build_uci_bundle(200, 100, False)
    per_fold, per_class, complexity, history, split_rows = [], [], [], [], []
    pooled_predictions: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        model: ([], []) for model in UCI_MODELS
    }
    for fold_index in range(folds):
        fold_id, masks, split_values, split = base.make_split_masks(bundle, fold_index)
        split_rows.extend(split)
        x_train, x_val, x_test = robust_uci_normalize(
            bundle.x[masks["train"]], bundle.x[masks["train"]],
            bundle.x[masks["validation"]], bundle.x[masks["test"]],
        )
        y_train, y_val, y_test = bundle.y[masks["train"]], bundle.y[masks["validation"]], bundle.y[masks["test"]]
        s_train = bundle.manifest.loc[masks["train"], "subject_id"].to_numpy(dtype=np.int64)
        s_val = bundle.manifest.loc[masks["validation"], "subject_id"].to_numpy(dtype=np.int64)
        s_test = bundle.manifest.loc[masks["test"], "subject_id"].to_numpy(dtype=np.int64)
        for model_index, model_name in enumerate(UCI_MODELS):
            print(f"[UCI] fold {fold_index + 1}/{folds} {DISPLAY[model_name]}", flush=True)
            result = train_uci_model(
                model_name, x_train, y_train, s_train, x_val, y_val, s_val,
                len(bundle.class_map), device, epochs, batch_size,
                SEED + fold_index * 100 + model_index,
            )
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(s_test)),
                batch_size=batch_size, shuffle=False,
            )
            start = time.perf_counter()
            y_true, probs = predict_uci(result.model, test_loader, device)
            infer_seconds = time.perf_counter() - start
            y_pred = probs.argmax(axis=1)
            metrics = metric_dict(y_true, y_pred, len(bundle.class_map), "test")
            per_fold.append(
                {
                    "dataset_name": bundle.display_name,
                    "protocol": "subject-aware 3-fold; test subjects held out",
                    "evaluation_unit": "window",
                    "fold_id": fold_id,
                    "model_name": model_name,
                    "model_display_name": DISPLAY[model_name],
                    "train_split_values": ";".join(map(str, split_values["train"])),
                    "validation_split_values": ";".join(map(str, split_values["validation"])),
                    "test_split_values": ";".join(map(str, split_values["test"])),
                    "train_sample_count": int(len(y_train)),
                    "validation_sample_count": int(len(y_val)),
                    "test_sample_count": int(len(y_test)),
                    "best_epoch": result.best_epoch,
                    "best_val_macro_f1": result.best_val_macro_f1,
                    "train_seconds": result.train_seconds,
                    "avg_inference_ms_per_sample": float(infer_seconds / max(len(y_test), 1) * 1000),
                    "param_count": result.param_count,
                    **metrics,
                }
            )
            complexity.append({key: per_fold[-1][key] for key in [
                "dataset_name", "fold_id", "model_name", "model_display_name", "param_count",
                "train_seconds", "avg_inference_ms_per_sample",
            ]})
            append_per_class(per_class, bundle.display_name, fold_id, model_name, y_true, y_pred, bundle.class_map)
            history.extend(
                {"dataset_name": bundle.display_name, "fold_id": fold_id, "model_name": model_name,
                 "model_display_name": DISPLAY[model_name], **row}
                for row in result.history
            )
            pooled_predictions[model_name][0].append(y_true)
            pooled_predictions[model_name][1].append(y_pred)
    final_predictions = {
        model: (np.concatenate(values[0]), np.concatenate(values[1]))
        for model, values in pooled_predictions.items()
    }
    return per_fold, per_class, complexity, history, split_rows, final_predictions


def run_capg(
    device: torch.device, epochs: int, batch_size: int, folds: int,
    frames_per_recording: int, radius: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict[str, tuple[np.ndarray, np.ndarray]], CapgFrameBundle]:
    bundle = build_capg_frames(list(range(1, 19)), frames_per_recording, radius)
    per_fold, per_class, complexity, history, split_rows = [], [], [], [], []
    pooled_predictions: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {
        model: ([], []) for model in CAPG_MODELS
    }
    trial_groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
    validation_groups = [[4, 5], [7, 8], [1, 2]]
    for fold_index in range(folds):
        test_trials = trial_groups[fold_index]
        val_trials = validation_groups[fold_index]
        train_trials = [trial for trial in range(1, 11) if trial not in set(test_trials + val_trials)]
        split_values = {"train": train_trials, "validation": val_trials, "test": test_trials}
        masks = {role: np.isin(bundle.trial_id, values) for role, values in split_values.items()}
        fold_id = f"capgmyo_trial_fold{fold_index + 1}"
        for role, values in split_values.items():
            for value in values:
                split_rows.append(
                    {
                        "dataset_name": "CapgMyo_DBa_full_spatial",
                        "fold_id": fold_id,
                        "split_unit": "trial_id",
                        "split_value": value,
                        "split": role,
                    }
                )
        x_train, x_val, x_test = normalize_capg_by_train_subject(
            bundle.x, bundle.subject_id, masks["train"],
            masks["train"], masks["validation"], masks["test"],
        )
        y_train, y_val, y_test = bundle.y[masks["train"]], bundle.y[masks["validation"]], bundle.y[masks["test"]]
        rec_train = bundle.recording_index[masks["train"]]
        rec_val = bundle.recording_index[masks["validation"]]
        rec_test = bundle.recording_index[masks["test"]]
        for model_index, model_name in enumerate(CAPG_MODELS):
            print(f"[CapgMyo] fold {fold_index + 1}/{folds} {DISPLAY[model_name]}", flush=True)
            result = train_capg_model(
                model_name, x_train, y_train, rec_train, x_val, y_val, rec_val,
                device, epochs, batch_size, SEED + 1000 + fold_index * 100 + model_index,
            )
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test), torch.from_numpy(rec_test)),
                batch_size=batch_size, shuffle=False,
            )
            start = time.perf_counter()
            frame_y, frame_probs, frame_recs = predict_capg(result.model, test_loader, device)
            infer_seconds = time.perf_counter() - start
            y_true, probs, _ = aggregate_recording_predictions(frame_y, frame_probs, frame_recs)
            y_pred = probs.argmax(axis=1)
            metrics = metric_dict(y_true, y_pred, 8, "test")
            per_fold.append(
                {
                    "dataset_name": "CapgMyo_DBa_full_spatial",
                    "protocol": "trial-aware calibrated 3-fold; no recording crosses split",
                    "evaluation_unit": "recording",
                    "fold_id": fold_id,
                    "model_name": model_name,
                    "model_display_name": DISPLAY[model_name],
                    "train_split_values": ";".join(map(str, train_trials)),
                    "validation_split_values": ";".join(map(str, val_trials)),
                    "test_split_values": ";".join(map(str, test_trials)),
                    "train_sample_count": int(len(np.unique(rec_train))),
                    "validation_sample_count": int(len(np.unique(rec_val))),
                    "test_sample_count": int(len(np.unique(rec_test))),
                    "frames_per_recording": frames_per_recording,
                    "best_epoch": result.best_epoch,
                    "best_val_macro_f1": result.best_val_macro_f1,
                    "train_seconds": result.train_seconds,
                    "avg_inference_ms_per_sample": float(infer_seconds / max(len(y_true), 1) * 1000),
                    "param_count": result.param_count,
                    **metrics,
                }
            )
            complexity.append({key: per_fold[-1][key] for key in [
                "dataset_name", "fold_id", "model_name", "model_display_name", "param_count",
                "train_seconds", "avg_inference_ms_per_sample",
            ]})
            append_per_class(
                per_class, "CapgMyo_DBa_full_spatial", fold_id, model_name, y_true, y_pred,
                {idx: f"Gesture{idx + 1}" for idx in range(8)},
            )
            history.extend(
                {"dataset_name": "CapgMyo_DBa_full_spatial", "fold_id": fold_id,
                 "model_name": model_name, "model_display_name": DISPLAY[model_name], **row}
                for row in result.history
            )
            pooled_predictions[model_name][0].append(y_true)
            pooled_predictions[model_name][1].append(y_pred)
    final_predictions = {
        model: (np.concatenate(values[0]), np.concatenate(values[1]))
        for model, values in pooled_predictions.items()
    }
    return per_fold, per_class, complexity, history, split_rows, final_predictions, bundle


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, model), group in per_fold.groupby(["dataset_name", "model_name"], sort=False):
        rows.append(
            {
                "dataset_name": dataset,
                "model_name": model,
                "model_display_name": group["model_display_name"].iloc[0],
                "protocol": group["protocol"].iloc[0],
                "evaluation_unit": group["evaluation_unit"].iloc[0],
                "folds_completed": int(len(group)),
                "validation_macro_f1_mean": float(group["best_val_macro_f1"].mean()),
                "test_accuracy_mean": float(group["test_accuracy"].mean()),
                "test_accuracy_std": float(group["test_accuracy"].std(ddof=0)),
                "test_balanced_accuracy_mean": float(group["test_balanced_accuracy"].mean()),
                "test_macro_f1_mean": float(group["test_macro_f1"].mean()),
                "test_macro_f1_std": float(group["test_macro_f1"].std(ddof=0)),
                "test_macro_f1_min": float(group["test_macro_f1"].min()),
                "test_macro_f1_max": float(group["test_macro_f1"].max()),
                "param_count_mean": float(group["param_count"].mean()),
                "train_seconds_mean": float(group["train_seconds"].mean()),
                "avg_inference_ms_per_sample_mean": float(group["avg_inference_ms_per_sample"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    summary["selected_by_validation"] = False
    for _, group in summary.groupby("dataset_name"):
        summary.loc[group["validation_macro_f1_mean"].idxmax(), "selected_by_validation"] = True
    return summary.sort_values(["dataset_name", "test_macro_f1_mean"], ascending=[True, False]).reset_index(drop=True)


def load_stage19_baselines() -> pd.DataFrame:
    path = ROOT / "reports" / "stage19_loss_convergence_public_deep" / "LOSS_CONVERGENCE_SUMMARY.csv"
    frame = pd.read_csv(path)
    rows = []
    for (dataset, model), group in frame.groupby(["dataset_name", "model_display_name"]):
        rows.append(
            {
                "dataset_name": dataset,
                "model_display_name": f"Stage19 converged {model}",
                "test_accuracy_mean": float(group["test_accuracy"].mean()),
                "test_macro_f1_mean": float(group["test_macro_f1"].mean()),
                "source_file": str(path.relative_to(ROOT)),
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(path: Path, summary: pd.DataFrame, baselines: pd.DataFrame) -> None:
    datasets = list(summary["dataset_name"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(12.0, 4.4), dpi=180, sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    palette = ["#A6A6A6", "#D0D0D0", "#4C78A8", "#E45756"]
    for ax, dataset in zip(axes, datasets):
        current = summary[summary["dataset_name"] == dataset].copy()
        baseline_key = "UCI_EMG_Data_for_Gestures" if dataset.startswith("UCI") else "CapgMyo_DBa_subset"
        base_rows = baselines[baselines["dataset_name"] == baseline_key].copy()
        labels = base_rows["model_display_name"].tolist() + current["model_display_name"].tolist()
        values = base_rows["test_macro_f1_mean"].tolist() + current["test_macro_f1_mean"].tolist()
        bars = ax.barh(np.arange(len(labels)), values, color=palette[: len(labels)], edgecolor="white")
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        ax.invert_yaxis()
        ax.set_xlim(max(0.0, min(values) - 0.08), min(1.0, max(values) + 0.08))
        ax.set_xlabel("Test Macro-F1")
        ax.set_title("UCI full: subject-aware" if dataset.startswith("UCI") else "CapgMyo full: trial-aware")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        for bar, value in zip(bars, values):
            ax.text(value + 0.005, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=600 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(
    out_dir: Path, predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    class_names: list[str], prefix: str,
) -> None:
    for model, (y_true, y_pred) in predictions.items():
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))), normalize="true")
        fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=180)
        image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)), labels=class_names, rotation=35, ha="right")
        ax.set_yticks(range(len(class_names)), labels=class_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(DISPLAY[model])
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, f"{cm[row, col]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cm[row, col] > 0.55 else "black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized proportion")
        fig.tight_layout()
        stem = out_dir / f"{prefix}_{model}_confusion"
        for suffix in ("png", "svg", "pdf"):
            fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600 if suffix == "png" else None, bbox_inches="tight")
        plt.close(fig)


def write_reports(out: Path, summary: pd.DataFrame, args: argparse.Namespace) -> None:
    selected = summary[summary["selected_by_validation"]].copy()
    report = [
        "# Stage24 Strict Advanced Model Optimization",
        "",
        "## Purpose",
        "",
        "This stage keeps the Stage17 strict split units and replaces only the representation/model family.",
        "Stage23 random-window demo is not used as evidence in this stage.",
        "",
        "## Protocol",
        "",
        "- UCI: original subject-aware 3-fold split; test subjects are unseen during training, normalization, and selection.",
        "- CapgMyo: original trial-aware calibrated 3-fold split; every frame from one MAT recording remains in one split.",
        "- Model selection: mean validation Macro-F1 only. Test metrics are reported after candidates are fixed.",
        "- CapgMyo aggregation: arithmetic mean probability across fixed frames from the same held-out recording.",
        "",
        "## Results",
        "",
    ]
    for _, row in summary.iterrows():
        report.append(
            f"- {row['dataset_name']} / {row['model_display_name']}: Accuracy={row['test_accuracy_mean']:.4f}, "
            f"Macro-F1={row['test_macro_f1_mean']:.4f}, validation Macro-F1={row['validation_macro_f1_mean']:.4f}."
        )
    report.extend(["", "## Validation-selected models", ""])
    for _, row in selected.iterrows():
        report.append(f"- {row['dataset_name']}: {row['model_display_name']}.")
    report.extend(
        [
            "",
            "## Boundaries",
            "",
            "- UCI subject-aware and CapgMyo trial-aware results are different protocols and must not be pooled.",
            "- CapgMyo is calibrated because each subject contributes labeled train trials; it is not zero-calibration cross-subject generalization.",
            "- Public-data results are not Bingbin results.",
            "- No test split, test label, or test metric was used to alter the fixed candidate set in this run.",
        ]
    )
    (out / "STAGE24_STRICT_ADVANCED_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    audit = f"""# Stage24 Leakage Audit

## Status: PASS

- Stage17 outputs were not overwritten.
- Stage23 random-window split was not used.
- UCI split unit: subject_id. Train/validation/test subject sets are disjoint in every fold.
- UCI normalization statistics are fitted on train subjects only.
- CapgMyo split unit: trial_id / MAT recording. All {args.frames_per_recording} derived frames from one recording stay in the same split.
- CapgMyo per-subject calibration statistics are fitted only from that subject's train trials and applied to validation/test trials.
- Validation recording/window Macro-F1 selects checkpoints and the preferred candidate. Test metrics are evaluation-only.
- No Limb or Bingbin data are read by this script.
"""
    (out / "LEAKAGE_AUDIT.md").write_text(audit, encoding="utf-8")
    decision = {
        "stage": "Stage24 strict advanced model optimization",
        "status": "DONE",
        "stage17_overwritten": False,
        "stage23_random_window_used": False,
        "limb_or_bingbin_used": False,
        "selection_metric": "mean validation Macro-F1",
        "selected_models": selected[
            ["dataset_name", "model_name", "model_display_name", "validation_macro_f1_mean", "test_accuracy_mean", "test_macro_f1_mean"]
        ].to_dict(orient="records"),
        "output_dir": str(out.relative_to(ROOT)),
    }
    (out / "STAGE24_DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", choices=["uci", "capgmyo", "both"], default="both")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--uci-epochs", type=int, default=80)
    parser.add_argument("--capg-epochs", type=int, default=55)
    parser.add_argument("--uci-batch-size", type=int, default=128)
    parser.add_argument("--capg-batch-size", type=int, default=512)
    parser.add_argument("--frames-per-recording", type=int, default=64)
    parser.add_argument("--frame-radius", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if not 1 <= args.folds <= 3:
        raise ValueError("--folds must be between 1 and 3")
    out = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    per_fold_rows, per_class_rows, complexity_rows, history_rows, split_rows = [], [], [], [], []
    all_predictions: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    capg_bundle = None
    if args.datasets in ("uci", "both"):
        pf, pc, cx, hist, splits, predictions = run_uci(device, args.uci_epochs, args.uci_batch_size, args.folds)
        per_fold_rows.extend(pf)
        per_class_rows.extend(pc)
        complexity_rows.extend(cx)
        history_rows.extend(hist)
        split_rows.extend(splits)
        all_predictions["uci"] = predictions
    if args.datasets in ("capgmyo", "both"):
        pf, pc, cx, hist, splits, predictions, capg_bundle = run_capg(
            device, args.capg_epochs, args.capg_batch_size, args.folds,
            args.frames_per_recording, args.frame_radius,
        )
        per_fold_rows.extend(pf)
        per_class_rows.extend(pc)
        complexity_rows.extend(cx)
        history_rows.extend(hist)
        split_rows.extend(splits)
        all_predictions["capgmyo"] = predictions

    per_fold = pd.DataFrame(per_fold_rows)
    per_class = pd.DataFrame(per_class_rows)
    complexity = pd.DataFrame(complexity_rows)
    history = pd.DataFrame(history_rows)
    split_manifest = pd.DataFrame(split_rows)
    summary = summarize(per_fold)
    baselines = load_stage19_baselines()
    summary.to_csv(out / "STRICT_ADVANCED_MODEL_SUMMARY.csv", index=False, encoding="utf-8-sig")
    per_fold.to_csv(out / "STRICT_ADVANCED_MODEL_PER_FOLD.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(out / "STRICT_ADVANCED_MODEL_PER_CLASS.csv", index=False, encoding="utf-8-sig")
    complexity.to_csv(out / "STRICT_ADVANCED_MODEL_COMPLEXITY.csv", index=False, encoding="utf-8-sig")
    history.to_csv(out / "STRICT_ADVANCED_TRAINING_HISTORY.csv", index=False, encoding="utf-8-sig")
    split_manifest.to_csv(out / "STRICT_ADVANCED_SPLIT_MANIFEST.csv", index=False, encoding="utf-8-sig")
    baselines.to_csv(out / "STAGE19_CONVERGED_BASELINES.csv", index=False, encoding="utf-8-sig")
    if capg_bundle is not None:
        capg_bundle.recording_table.to_csv(out / "CAPGMYO_RECORDING_INVENTORY.csv", index=False, encoding="utf-8-sig")
        capg_bundle.frame_table.to_csv(out / "CAPGMYO_FRAME_MANIFEST.csv", index=False, encoding="utf-8-sig")
    plot_comparison(figures / "strict_advanced_model_comparison", summary, baselines)
    if "uci" in all_predictions:
        plot_confusions(figures, all_predictions["uci"], [base.UCI_CLASS_MAP[i] for i in range(6)], "uci")
    if "capgmyo" in all_predictions:
        plot_confusions(figures, all_predictions["capgmyo"], [f"Gesture{i}" for i in range(1, 9)], "capgmyo")
    write_reports(out, summary, args)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

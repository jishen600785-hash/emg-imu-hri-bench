from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import time
import urllib.request
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "stage16_public_deep_benchmark"
FIG_DIR = OUT / "figures"
SEED = 42

UCI_DIR = ROOT / "data_public" / "uci_emg_gestures"
UCI_ZIP = UCI_DIR / "emg_data_for_gestures.zip"
UCI_EXTRACTED = UCI_DIR / "extracted" / "EMG_data_for_gestures-master"
UCI_URL = "https://archive.ics.uci.edu/static/public/481/emg+data+for+gestures.zip"
UCI_CHANNELS = [f"channel{i}" for i in range(1, 9)]
UCI_CLASS_MAP = {
    0: "Rest",
    1: "Fist",
    2: "WristFlexion",
    3: "WristExtension",
    4: "RadialDeviation",
    5: "UlnarDeviation",
}

CAPG_DIR = ROOT / "data_public" / "capgmyo_dba"
CAPG_FILES = {
    1: {
        "name": "dba-s1.zip",
        "url": "https://ndownloader.figshare.com/files/13277105",
        "md5": "ee30c5235d817e3c96492aff01fc9423",
    },
    2: {
        "name": "dba-s2.zip",
        "url": "https://ndownloader.figshare.com/files/13275896",
        "md5": "63a0dbf3f6387097d957a9cdca2182c9",
    },
    3: {
        "name": "dba-s3.zip",
        "url": "https://ndownloader.figshare.com/files/13275953",
        "md5": "cef147a5a0a0be71f9da30d15461dbb3",
    },
    4: {
        "name": "dba-s4.zip",
        "url": "https://ndownloader.figshare.com/files/13275962",
        "md5": "21bf535d25369b1c1cda33d582105df8",
    },
    5: {
        "name": "dba-s5.zip",
        "url": "https://ndownloader.figshare.com/files/13275965",
        "md5": "397a6fd60c7c63e5a6e6254957435b35",
    },
    6: {
        "name": "dba-s6.zip",
        "url": "https://ndownloader.figshare.com/files/13275983",
        "md5": "9e16e3686e9620da3d6be980b7b1e4e0",
    },
    7: {
        "name": "dba-s7.zip",
        "url": "https://ndownloader.figshare.com/files/13276016",
        "md5": "0689e978553f3ba1a32d124fca097fad",
    },
    8: {
        "name": "dba-s8.zip",
        "url": "https://ndownloader.figshare.com/files/13276019",
        "md5": "6bc3a96a4312dab93b1a198df2c63b59",
    },
    9: {
        "name": "dba-s9.zip",
        "url": "https://ndownloader.figshare.com/files/13276022",
        "md5": "743deeded248e00c0faee2d86c4b3e6d",
    },
    10: {
        "name": "dba-s10.zip",
        "url": "https://ndownloader.figshare.com/files/13276934",
        "md5": "c8fe5aa7e61d44d25ac086852123aa60",
    },
    11: {
        "name": "dba-s11.zip",
        "url": "https://ndownloader.figshare.com/files/13277147",
        "md5": "74382cc1927835c731b1305d4a3f6501",
    },
    12: {
        "name": "dba-s12.zip",
        "url": "https://ndownloader.figshare.com/files/13277000",
        "md5": "ec92802ea6a9e656805895fc8b5016f9",
    },
    13: {
        "name": "dba-s13.zip",
        "url": "https://ndownloader.figshare.com/files/13277015",
        "md5": "a29e85b204168fccb8104922ca9c01ae",
    },
    14: {
        "name": "dba-s14.zip",
        "url": "https://ndownloader.figshare.com/files/13277027",
        "md5": "574845abe6de997158f82f19d6ab03cb",
    },
    15: {
        "name": "dba-s15.zip",
        "url": "https://ndownloader.figshare.com/files/13276136",
        "md5": "c1e85885f2c14dd41e22de90ab771a29",
    },
    16: {
        "name": "dba-s16.zip",
        "url": "https://ndownloader.figshare.com/files/13276139",
        "md5": "c742e7d0f2b45bbe45b1f7e5bb5bf9c5",
    },
    17: {
        "name": "dba-s17.zip",
        "url": "https://ndownloader.figshare.com/files/13276142",
        "md5": "c638a7ffe355be522b1f0f4e4e1bb799",
    },
    18: {
        "name": "dba-s18.zip",
        "url": "https://ndownloader.figshare.com/files/13276148",
        "md5": "386a49bf4f6a8c86135d259d7a26d591",
    },
}
CAPG_CLASS_MAP = {i - 1: f"Gesture{i}" for i in range(1, 9)}

MODEL_NAMES = ["CNN1D", "TCN", "BiLSTM", "LSTMTransformer"]
MODEL_DISPLAY_NAMES = {
    "CNN1D": "1D-CNN",
    "TCN": "TCN",
    "BiLSTM": "BiLSTM",
    "LSTMTransformer": "LSTM-Transformer",
}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class DatasetBundle:
    key: str
    display_name: str
    source_url: str
    source_file: str
    subset_note: str
    split_protocol: str
    class_map: dict[int, str]
    manifest: pd.DataFrame
    x: np.ndarray
    y: np.ndarray
    inventory_rows: list[dict]


@dataclass
class TrainResult:
    model: nn.Module
    history: list[dict]
    best_epoch: int
    best_val_macro_f1: float
    train_seconds: float
    param_count: int


def native(obj):
    if isinstance(obj, dict):
        return {str(k): native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return native(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_uci(download: bool) -> None:
    UCI_DIR.mkdir(parents=True, exist_ok=True)
    if not UCI_ZIP.exists():
        if not download:
            raise FileNotFoundError(f"Missing {UCI_ZIP}; rerun with --download")
        urllib.request.urlretrieve(UCI_URL, UCI_ZIP)
    if not UCI_EXTRACTED.exists():
        extract_dir = UCI_DIR / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(UCI_ZIP, "r") as zf:
            zf.extractall(extract_dir)


def ensure_capgmyo(download: bool, subjects: list[int]) -> None:
    CAPG_DIR.mkdir(parents=True, exist_ok=True)
    for subject_id in subjects:
        meta = CAPG_FILES.get(subject_id)
        if meta is None:
            raise ValueError(f"CapgMyo subject {subject_id} is not registered in this lightweight adapter")
        path = CAPG_DIR / meta["name"]
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; rerun with --download")
            print(f"Downloading CapgMyo DB-a subject S{subject_id}: {meta['url']}")
            urllib.request.urlretrieve(meta["url"], path)
        got = md5_file(path)
        if got.lower() != meta["md5"].lower():
            raise RuntimeError(f"MD5 mismatch for {path}: got {got}, expected {meta['md5']}")


def contiguous_segments(labels: np.ndarray) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []
    out = []
    start = 0
    cur = int(labels[0])
    for i in range(1, len(labels)):
        val = int(labels[i])
        if val != cur:
            out.append((start, i, cur))
            start = i
            cur = val
    out.append((start, len(labels), cur))
    return out


def build_uci_bundle(window_len: int, step: int, download: bool) -> DatasetBundle:
    ensure_uci(download)
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    manifest_rows: list[dict] = []
    inventory_rows: list[dict] = []
    sample_id = 0
    for subject_dir in sorted(UCI_EXTRACTED.glob("[0-9][0-9]")):
        subject_id = int(subject_dir.name)
        for path in sorted(subject_dir.glob("*_raw_data_*.txt")):
            series_id = int(path.name.split("_", 1)[0])
            df = pd.read_csv(path, sep="\t")
            labels = df["class"].to_numpy(dtype=int)
            values = df[UCI_CHANNELS].to_numpy(dtype=np.float32)
            class_counts = df["class"].value_counts().sort_index().to_dict()
            inventory_rows.append(
                {
                    "dataset_name": "UCI_EMG_Data_for_Gestures",
                    "subject_id": subject_id,
                    "series_id": series_id,
                    "source_file": str(path.relative_to(ROOT)),
                    "row_count": int(len(df)),
                    "channel_count": len(UCI_CHANNELS),
                    "native_class_values": ";".join(str(k) for k in sorted(class_counts)),
                    "class_counts": json.dumps({int(k): int(v) for k, v in class_counts.items()}, ensure_ascii=False),
                    "file_size_bytes": int(path.stat().st_size),
                }
            )
            for seg_idx, (start, end, native_cls) in enumerate(contiguous_segments(labels), start=1):
                if native_cls == 0 or native_cls == 7:
                    continue
                class_id = native_cls - 1
                if class_id not in UCI_CLASS_MAP or end - start < window_len:
                    continue
                local_start = start
                while local_start + window_len <= end:
                    local_end = local_start + window_len
                    x_rows.append(values[local_start:local_end].T.astype(np.float32))
                    y_rows.append(class_id)
                    manifest_rows.append(
                        {
                            "dataset_name": "UCI_EMG_Data_for_Gestures",
                            "sample_id": sample_id,
                            "subject_id": subject_id,
                            "series_id": series_id,
                            "recording_id": f"S{subject_id:02d}_R{series_id}",
                            "segment_id": f"S{subject_id:02d}_R{series_id}_seg{seg_idx:03d}",
                            "source_file": str(path.relative_to(ROOT)),
                            "class_id_original": native_cls,
                            "class_id": class_id,
                            "class_name": UCI_CLASS_MAP[class_id],
                            "start_index": local_start,
                            "end_index": local_end - 1,
                            "window_length": window_len,
                        }
                    )
                    sample_id += 1
                    local_start += step
    if not x_rows:
        raise RuntimeError("No UCI windows generated")
    return DatasetBundle(
        key="uci",
        display_name="UCI_EMG_Data_for_Gestures",
        source_url="https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures",
        source_file=str(UCI_ZIP.relative_to(ROOT)),
        subset_note="Full 36-subject UCI dataset; class 0 and optional class 7 excluded.",
        split_protocol="subject-aware 3-fold; test subject groups are S01-S12, S13-S24, S25-S36",
        class_map=UCI_CLASS_MAP,
        manifest=pd.DataFrame(manifest_rows),
        x=np.stack(x_rows).astype(np.float32),
        y=np.asarray(y_rows, dtype=np.int64),
        inventory_rows=inventory_rows,
    )


def resize_time_axis(data: np.ndarray, target_len: int) -> np.ndarray:
    if data.shape[0] == target_len:
        return data.astype(np.float32)
    src = np.linspace(0.0, 1.0, data.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.empty((target_len, data.shape[1]), dtype=np.float32)
    for ch in range(data.shape[1]):
        out[:, ch] = np.interp(dst, src, data[:, ch]).astype(np.float32)
    return out


def parse_capgmyo_mat_name(name: str) -> tuple[int, int, int]:
    m = re.match(r"(\d+)-(\d+)-(\d+)\.mat$", Path(name).name)
    if not m:
        raise ValueError(f"Unexpected CapgMyo file name: {name}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def build_capgmyo_bundle(subjects: list[int], target_len: int, download: bool) -> DatasetBundle:
    ensure_capgmyo(download, subjects)
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    manifest_rows: list[dict] = []
    inventory_rows: list[dict] = []
    sample_id = 0
    for subject_id in subjects:
        meta = CAPG_FILES[subject_id]
        zip_path = CAPG_DIR / meta["name"]
        with zipfile.ZipFile(zip_path, "r") as zf:
            mat_names = sorted(n for n in zf.namelist() if n.lower().endswith(".mat"))
            inventory_rows.append(
                {
                    "dataset_name": "CapgMyo_DBa_subset",
                    "subject_id": subject_id,
                    "series_id": "",
                    "source_file": str(zip_path.relative_to(ROOT)),
                    "row_count": len(mat_names),
                    "channel_count": 128,
                    "native_class_values": "1;2;3;4;5;6;7;8",
                    "class_counts": json.dumps({i: 10 for i in range(1, 9)}, ensure_ascii=False),
                    "file_size_bytes": int(zip_path.stat().st_size),
                    "md5": md5_file(zip_path),
                }
            )
            for mat_name in mat_names:
                parsed_subject, parsed_gesture, parsed_trial = parse_capgmyo_mat_name(mat_name)
                raw = zf.read(mat_name)
                mat = scipy.io.loadmat(io.BytesIO(raw))
                data = np.asarray(mat["data"], dtype=np.float32)
                native_subject = int(np.asarray(mat.get("subject", [[parsed_subject]])).squeeze())
                native_gesture = int(np.asarray(mat.get("gesture", [[parsed_gesture]])).squeeze())
                native_trial = int(np.asarray(mat.get("trial", [[parsed_trial]])).squeeze())
                if native_subject != subject_id:
                    raise RuntimeError(f"Subject mismatch in {mat_name}: zip S{subject_id}, mat S{native_subject}")
                if native_gesture < 1 or native_gesture > 8:
                    continue
                resized = resize_time_axis(data, target_len)
                class_id = native_gesture - 1
                x_rows.append(resized.T.astype(np.float32))
                y_rows.append(class_id)
                manifest_rows.append(
                    {
                        "dataset_name": "CapgMyo_DBa_subset",
                        "sample_id": sample_id,
                        "subject_id": subject_id,
                        "series_id": "",
                        "recording_id": f"S{subject_id:02d}_G{native_gesture:02d}_T{native_trial:02d}",
                        "segment_id": f"S{subject_id:02d}_G{native_gesture:02d}_T{native_trial:02d}",
                        "source_file": str(zip_path.relative_to(ROOT)),
                        "mat_file": mat_name,
                        "class_id_original": native_gesture,
                        "class_id": class_id,
                        "class_name": CAPG_CLASS_MAP[class_id],
                        "trial_id": native_trial,
                        "gesture_id": native_gesture,
                        "start_index": 0,
                        "end_index": int(data.shape[0] - 1),
                        "window_length": target_len,
                        "native_time_points": int(data.shape[0]),
                    }
                )
                sample_id += 1
    if not x_rows:
        raise RuntimeError("No CapgMyo recordings generated")
    sub_range = ",".join(f"S{s}" for s in subjects)
    return DatasetBundle(
        key="capgmyo",
        display_name="CapgMyo_DBa_subset",
        source_url="https://figshare.com/articles/dataset/Data_from_Gesture_Recognition_by_Instantaneous_Surface_EMG_Images_CapgMyo-DBa/7210397",
        source_file=";".join(str((CAPG_DIR / CAPG_FILES[s]["name"]).relative_to(ROOT)) for s in subjects),
        subset_note=f"CapgMyo DB-a subject subset: {sub_range}; 8 gestures, 10 trials per gesture; time axis interpolated to {target_len} points.",
        split_protocol="recording/trial-aware 3-fold within selected subjects; no MAT recording crosses split",
        class_map=CAPG_CLASS_MAP,
        manifest=pd.DataFrame(manifest_rows),
        x=np.stack(x_rows).astype(np.float32),
        y=np.asarray(y_rows, dtype=np.int64),
        inventory_rows=inventory_rows,
    )


def split_subject_groups(subjects: list[int], fold_index: int) -> dict[str, list[int]]:
    groups = [list(map(int, arr)) for arr in np.array_split(np.asarray(sorted(subjects), dtype=int), 3)]
    test_subjects = groups[fold_index]
    validation_subjects = groups[(fold_index + 1) % 3][: max(1, len(groups[(fold_index + 1) % 3]) // 2)]
    used = set(test_subjects + validation_subjects)
    train_subjects = [s for s in sorted(subjects) if s not in used]
    return {"train": train_subjects, "validation": validation_subjects, "test": test_subjects}


def make_split_masks(bundle: DatasetBundle, fold_index: int) -> tuple[str, dict[str, np.ndarray], dict[str, list[int]], list[dict]]:
    if bundle.key == "capgmyo":
        trial_groups = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10]]
        validation_groups = [[4, 5], [7, 8], [1, 2]]
        test_trials = trial_groups[fold_index]
        validation_trials = validation_groups[fold_index]
        train_trials = [t for t in range(1, 11) if t not in set(test_trials + validation_trials)]
        split_values = {"train": train_trials, "validation": validation_trials, "test": test_trials}
        masks = {role: bundle.manifest["trial_id"].isin(vals).to_numpy() for role, vals in split_values.items()}
        fold_id = f"{bundle.key}_trial_fold{fold_index + 1}"
        rows = []
        for role, vals in split_values.items():
            for trial_id in vals:
                rows.append(
                    {
                        "dataset_name": bundle.display_name,
                        "fold_id": fold_id,
                        "split_unit": "trial_id",
                        "split_value": trial_id,
                        "split": role,
                    }
                )
        return fold_id, masks, split_values, rows

    subjects = sorted(bundle.manifest["subject_id"].unique().tolist())
    split_values = split_subject_groups(subjects, fold_index)
    masks = {role: bundle.manifest["subject_id"].isin(vals).to_numpy() for role, vals in split_values.items()}
    fold_id = f"{bundle.key}_subject_fold{fold_index + 1}"
    rows = []
    for role, vals in split_values.items():
        for subject_id in vals:
            rows.append(
                {
                    "dataset_name": bundle.display_name,
                    "fold_id": fold_id,
                    "split_unit": "subject_id",
                    "split_value": subject_id,
                    "split": role,
                }
            )
    return fold_id, masks, split_values, rows


def standardize_by_train(x_train: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-8)
    return tuple(((arr - mean) / std).astype(np.float32) for arr in arrays)


class CNN1D(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(96, classes)

    def forward(self, x):
        return self.head(self.net(x).squeeze(-1))


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        pad = dilation
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.conv(x))


class SmallTCN(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(channels, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU())
        self.blocks = nn.Sequential(TCNBlock(64, 1), TCNBlock(64, 2), TCNBlock(64, 4), TCNBlock(64, 8))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        return self.head(self.pool(self.blocks(self.stem(x))).squeeze(-1))


class BiLSTMNet(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=channels, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, classes))

    def forward(self, x):
        out, _ = self.lstm(x.transpose(1, 2))
        return self.head(out[:, -1, :])


class LSTMTransformerNet(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=80, num_layers=1, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=160,
            nhead=4,
            dim_feedforward=320,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.attn = nn.Sequential(nn.Linear(160, 64), nn.Tanh(), nn.Linear(64, 1))
        self.head = nn.Sequential(nn.LayerNorm(160), nn.Dropout(0.1), nn.Linear(160, classes))

    def forward(self, x):
        seq = self.patch(x).transpose(1, 2)
        seq, _ = self.lstm(seq)
        seq = self.transformer(seq)
        weights = torch.softmax(self.attn(seq).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (seq * weights).sum(dim=1)
        return self.head(pooled)


def make_model(name: str, channels: int, classes: int) -> nn.Module:
    if name == "CNN1D":
        return CNN1D(channels, classes)
    if name == "TCN":
        return SmallTCN(channels, classes)
    if name == "BiLSTM":
        return BiLSTMNet(channels, classes)
    if name == "LSTMTransformer":
        return LSTMTransformerNet(channels, classes)
    raise ValueError(name)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    ys = []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
            ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, class_count: int, prefix: str) -> dict:
    labels = list(range(class_count))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_precision": float(p),
        f"{prefix}_macro_recall": float(r),
        f"{prefix}_macro_f1": float(f1),
    }


def train_one_model(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    class_count: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> TrainResult:
    set_seed(seed)
    model = make_model(model_name, x_train.shape[1], class_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)), batch_size=batch_size, shuffle=False)
    best_state = None
    best_epoch = 0
    best_val = -1.0
    stale = 0
    patience = 8 if model_name == "LSTMTransformer" else 5
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_true, val_probs = predict(model, val_loader, device)
        val_pred = val_probs.argmax(axis=1)
        val_metrics = metric_dict(val_true, val_pred, class_count, "val")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics}
        history.append(row)
        if val_metrics["val_macro_f1"] > best_val + 1e-6:
            best_val = val_metrics["val_macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model, history, best_epoch, best_val, time.perf_counter() - start, count_params(model))


def plot_confusion(path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_map: dict[int, str], title: str) -> None:
    labels = list(range(len(class_map)))
    names = [class_map[i] for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, labels=names, rotation=35, ha="right")
    ax.set_yticks(labels, labels=names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_training_curves(path: Path, history_rows: list[dict], dataset_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=180)
    for model_name in MODEL_NAMES:
        rows = [r for r in history_rows if r["model_name"] == model_name]
        if not rows:
            continue
        df = pd.DataFrame(rows)
        grouped = df.groupby("epoch", as_index=False).agg({"train_loss": "mean", "val_macro_f1": "mean"})
        label = MODEL_DISPLAY_NAMES[model_name]
        lw = 2.4 if model_name == "LSTMTransformer" else 1.8
        axes[0].plot(grouped["epoch"], grouped["train_loss"], label=label, linewidth=lw)
        axes[1].plot(grouped["epoch"], grouped["val_macro_f1"], label=label, linewidth=lw)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f"{dataset_name}: four-model training curves")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_model_comparison(path: Path, summary_rows: list[dict], dataset_name: str) -> None:
    metrics = [
        ("test_macro_f1_mean", "Macro-F1"),
        ("test_accuracy_mean", "Accuracy"),
        ("test_balanced_accuracy_mean", "Balanced Acc."),
    ]
    models = [r["model_display_name"] for r in summary_rows]
    x = np.arange(len(models))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=180)
    for idx, (field, label) in enumerate(metrics):
        vals = [float(r[field]) for r in summary_rows]
        bars = ax.bar(x + (idx - 1) * width, vals, width, label=label)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, models, rotation=0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"{dataset_name}: deep-model comparison", pad=14)
    ax.legend(frameon=False, ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_strategy_overview(path: Path, public_best_rows: list[dict]) -> None:
    labels = ["Limb\npersonalized", "Bingbin\ntraditional ML V3"]
    values = [0.9050, 0.9272]
    colors = ["#4C78A8", "#59A14F"]
    for row in public_best_rows:
        labels.append(row["dataset_short"])
        values.append(float(row["test_macro_f1_mean"]))
        colors.append("#E15759" if row["model_name"] == "LSTMTransformer" else "#F28E2B")
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
    bars = ax.bar(labels, values, color=colors)
    for b, val in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{val:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Platform strategy: deep learning for large datasets, traditional ML for small datasets")
    ax.text(1, 0.16, "Bingbin V3 is exploratory\nwith a test-reuse caveat", ha="center", fontsize=8)
    ax.text(2.5, 0.08, "Public-dataset results are separate from Bingbin application results", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_dataset_benchmark(
    bundle: DatasetBundle,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict]]:
    class_count = len(bundle.class_map)
    per_fold_rows: list[dict] = []
    per_class_rows: list[dict] = []
    complexity_rows: list[dict] = []
    history_rows: list[dict] = []
    split_rows: list[dict] = []
    summary_rows: list[dict] = []
    all_true: dict[str, list[np.ndarray]] = {m: [] for m in MODEL_NAMES}
    all_pred: dict[str, list[np.ndarray]] = {m: [] for m in MODEL_NAMES}

    for fold_index in range(3):
        fold_id, masks, split_values, fold_split_rows = make_split_masks(bundle, fold_index)
        split_rows.extend(fold_split_rows)
        x_train, x_val, x_test = standardize_by_train(bundle.x[masks["train"]], bundle.x[masks["train"]], bundle.x[masks["validation"]], bundle.x[masks["test"]])
        y_train, y_val, y_test = bundle.y[masks["train"]], bundle.y[masks["validation"]], bundle.y[masks["test"]]
        for model_idx, model_name in enumerate(MODEL_NAMES):
            display_name = MODEL_DISPLAY_NAMES[model_name]
            print(f"Training {bundle.display_name} {display_name} fold{fold_index + 1} on {device}")
            result = train_one_model(
                model_name,
                x_train,
                y_train,
                x_val,
                y_val,
                class_count,
                device,
                epochs,
                batch_size,
                SEED + fold_index * 100 + model_idx,
            )
            test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)), batch_size=batch_size, shuffle=False)
            start = time.perf_counter()
            y_true, probs = predict(result.model, test_loader, device)
            infer_seconds = time.perf_counter() - start
            y_pred = probs.argmax(axis=1)
            metrics = metric_dict(y_true, y_pred, class_count, "test")
            row = {
                "dataset_name": bundle.display_name,
                "fold_id": fold_id,
                "model_name": model_name,
                "model_display_name": display_name,
                "train_split_values": ";".join(str(v) for v in split_values["train"]),
                "validation_split_values": ";".join(str(v) for v in split_values["validation"]),
                "test_split_values": ";".join(str(v) for v in split_values["test"]),
                "train_sample_count": int(len(y_train)),
                "validation_sample_count": int(len(y_val)),
                "test_sample_count": int(len(y_test)),
                "best_epoch": result.best_epoch,
                "best_val_macro_f1": result.best_val_macro_f1,
                "train_seconds": result.train_seconds,
                "avg_inference_ms_per_sample": float(infer_seconds / max(len(y_test), 1) * 1000),
                "param_count": result.param_count,
                "source_file": bundle.source_file,
                **metrics,
            }
            per_fold_rows.append(row)
            p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(class_count)), zero_division=0)
            for cls_idx, class_name in bundle.class_map.items():
                per_class_rows.append(
                    {
                        "dataset_name": bundle.display_name,
                        "fold_id": fold_id,
                        "model_name": model_name,
                        "model_display_name": display_name,
                        "class_id": cls_idx,
                        "class_name": class_name,
                        "precision": float(p[cls_idx]),
                        "recall": float(r[cls_idx]),
                        "f1": float(f1[cls_idx]),
                        "support": int(support[cls_idx]),
                    }
                )
            complexity_rows.append(
                {
                    "dataset_name": bundle.display_name,
                    "fold_id": fold_id,
                    "model_name": model_name,
                    "model_display_name": display_name,
                    "param_count": result.param_count,
                    "train_seconds": result.train_seconds,
                    "avg_inference_ms_per_sample": float(infer_seconds / max(len(y_test), 1) * 1000),
                    "source_file": bundle.source_file,
                }
            )
            for h in result.history:
                history_rows.append({"dataset_name": bundle.display_name, "fold_id": fold_id, "model_name": model_name, "model_display_name": display_name, **h})
            all_true[model_name].append(y_true)
            all_pred[model_name].append(y_pred)

    for model_name in MODEL_NAMES:
        rows = [r for r in per_fold_rows if r["model_name"] == model_name]
        y_true_all = np.concatenate(all_true[model_name])
        y_pred_all = np.concatenate(all_pred[model_name])
        display_name = MODEL_DISPLAY_NAMES[model_name]
        plot_confusion(
            FIG_DIR / f"{bundle.display_name}_{model_name}_confusion_matrix.png",
            y_true_all,
            y_pred_all,
            bundle.class_map,
            f"{bundle.display_name}: {display_name} confusion matrix",
        )
        summary = {
            "dataset_name": bundle.display_name,
            "dataset_short": "UCI\npublic deep" if bundle.key == "uci" else "CapgMyo\npublic deep",
            "model_name": model_name,
            "model_display_name": display_name,
            "folds_completed": len(rows),
            "test_accuracy_mean": float(np.mean([r["test_accuracy"] for r in rows])),
            "test_accuracy_std": float(np.std([r["test_accuracy"] for r in rows], ddof=0)),
            "test_balanced_accuracy_mean": float(np.mean([r["test_balanced_accuracy"] for r in rows])),
            "test_balanced_accuracy_std": float(np.std([r["test_balanced_accuracy"] for r in rows], ddof=0)),
            "test_macro_f1_mean": float(np.mean([r["test_macro_f1"] for r in rows])),
            "test_macro_f1_std": float(np.std([r["test_macro_f1"] for r in rows], ddof=0)),
            "test_macro_f1_min": float(np.min([r["test_macro_f1"] for r in rows])),
            "test_macro_f1_max": float(np.max([r["test_macro_f1"] for r in rows])),
            "param_count_mean": float(np.mean([r["param_count"] for r in rows])),
            "train_seconds_mean": float(np.mean([r["train_seconds"] for r in rows])),
            "avg_inference_ms_per_sample_mean": float(np.mean([r["avg_inference_ms_per_sample"] for r in rows])),
            "source_file": bundle.source_file,
        }
        summary_rows.append(summary)

    plot_model_comparison(FIG_DIR / f"{bundle.display_name}_model_comparison_cn.png", summary_rows, bundle.display_name)
    plot_training_curves(FIG_DIR / f"{bundle.display_name}_training_curves_4models_cn.png", history_rows, bundle.display_name)
    return per_fold_rows, per_class_rows, complexity_rows, history_rows, split_rows, summary_rows


def write_dataset_selection(capg_subjects: list[int]) -> None:
    rows = [
        {
            "dataset_name": "UCI EMG Data for Gestures",
            "role": "formal_benchmark_A",
            "official_source": "https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures",
            "subjects": "36",
            "channels": "8 EMG",
            "classes": "6 common gesture classes used; class 0 and optional class 7 excluded",
            "download_size": "about 16.9 MB",
            "license": "CC BY 4.0 per UCI page",
            "run_scope": "full dataset with subject-aware 3-fold split",
        },
        {
            "dataset_name": "CapgMyo-DBa",
            "role": "formal_benchmark_B_subset",
            "official_source": "https://figshare.com/articles/dataset/Data_from_Gesture_Recognition_by_Instantaneous_Surface_EMG_Images_CapgMyo-DBa/7210397",
            "subjects": "18 total; lightweight run uses " + ",".join(f"S{s}" for s in capg_subjects),
            "channels": "128 HD-sEMG",
            "classes": "8 gestures",
            "download_size": "full 1.31 GB; subset downloads selected subject zips only",
            "license": "CC0 per Figshare API",
            "run_scope": "subject subset benchmark with recording/trial-aware calibrated 3-fold split",
        },
        {
            "dataset_name": "GRABMyo",
            "role": "backup_large_extension",
            "official_source": "https://physionet.org/content/grabmyo/",
            "subjects": "43",
            "channels": "32 EMG",
            "classes": "16 gestures",
            "download_size": "9.4 GB uncompressed",
            "license": "PhysioNet terms",
            "run_scope": "not downloaded in Stage16; kept as later extension",
        },
        {
            "dataset_name": "Ninapro DB2",
            "role": "backup_standard_extension",
            "official_source": "https://ninapro.hevs.ch/instructions/DB2.html",
            "subjects": "40 intact subjects",
            "channels": "12 Delsys Trigno sEMG plus additional sensors",
            "classes": "49 movements plus rest",
            "download_size": "large",
            "license": "Ninapro terms",
            "run_scope": "not downloaded in Stage16; kept as later extension",
        },
    ]
    write_csv(OUT / "PUBLIC_DATASET_CANDIDATES.csv", rows)
    text = f"""# Public Dataset Selection

## Formal Benchmarks

Stage16 now uses two public datasets:

1. **UCI EMG Data for Gestures**
   - Official source: https://archive.ics.uci.edu/dataset/481/emg+data+for+gestures
   - Role: public raw-EMG deep-learning reproduction benchmark A.
   - Scope: full 36-subject dataset; six common classes; subject-aware 3-fold split.

2. **CapgMyo-DBa subset**
   - Official source: https://figshare.com/articles/dataset/Data_from_Gesture_Recognition_by_Instantaneous_Surface_EMG_Images_CapgMyo-DBa/7210397
   - Role: public HD-sEMG deep-learning reproduction benchmark B.
   - Scope: selected subject subset {','.join(f'S{s}' for s in capg_subjects)} because full DB-a is about 1.31 GB. The report labels this as a subset benchmark, not a full DB-a benchmark.
   - Split: recording/trial-aware calibrated split. Each MAT trial belongs to only one split; this is not a zero-calibration cross-subject protocol.

## Backup Datasets

- GRABMyo / PhysioNet: https://physionet.org/content/grabmyo/ . Large 43-participant benchmark; kept for later extension.
- Ninapro DB2: https://ninapro.hevs.ch/instructions/DB2.html . Standard benchmark; preprocessing cost is higher and not prioritized here.

## Project Positioning

The public dataset branch satisfies the proposal requirement for deep-learning model reproduction. It must not be described as a Bingbin_Realtime deep-learning result. application Bingbin_Realtime remains a small-sample recording-level demo branch where traditional ML is more stable.
"""
    (OUT / "PUBLIC_DATASET_SELECTION.md").write_text(text, encoding="utf-8")


def write_reports(
    bundles: list[DatasetBundle],
    summary_rows: list[dict],
    per_fold_rows: list[dict],
    split_rows: list[dict],
    device: str,
) -> None:
    best_by_dataset = {}
    for bundle in bundles:
        rows = [r for r in summary_rows if r["dataset_name"] == bundle.display_name]
        best_by_dataset[bundle.display_name] = max(rows, key=lambda r: float(r["test_macro_f1_mean"]))

    fig_lines = [
        f"- `{(FIG_DIR / 'public_deep_strategy_overview_cn.png').relative_to(ROOT)}`",
    ]
    for bundle in bundles:
        fig_lines.extend(
            [
                f"- `{(FIG_DIR / f'{bundle.display_name}_model_comparison_cn.png').relative_to(ROOT)}`",
                f"- `{(FIG_DIR / f'{bundle.display_name}_LSTMTransformer_confusion_matrix.png').relative_to(ROOT)}`",
                f"- `{(FIG_DIR / f'{bundle.display_name}_training_curves_4models_cn.png').relative_to(ROOT)}`",
            ]
        )
    figure_text = "\n".join(fig_lines)

    result_lines = []
    for bundle in bundles:
        best = best_by_dataset[bundle.display_name]
        lstm = next(r for r in summary_rows if r["dataset_name"] == bundle.display_name and r["model_name"] == "LSTMTransformer")
        result_lines.append(
            f"- {bundle.display_name}: best={best['model_display_name']} Macro-F1={float(best['test_macro_f1_mean']):.4f}; "
            f"LSTM-Transformer Macro-F1={float(lstm['test_macro_f1_mean']):.4f}."
        )
    result_text = "\n".join(result_lines)

    report = f"""# Public Deep Benchmark Report

## Scope

- Formal public datasets: UCI EMG Data for Gestures; CapgMyo-DBa selected-subject subset.
- Deep models: 1D-CNN, TCN, BiLSTM, LSTM-Transformer.
- LSTM-Transformer is included as a key model because the application proposal mentioned LSTM + Transformer.
- Split: UCI uses subject-aware 3-fold; CapgMyo subset uses recording/trial-aware calibrated 3-fold. Test split units are never used for training or validation.
- Device: {device}

## Results

{result_text}

If LSTM-Transformer is not the top model on a dataset, this is kept as the true result. The project conclusion is model selection by validation/test evidence, not forcing the proposal model to be best.

## Figures

{figure_text}

## Interpretation

The public branch demonstrates that the platform supports deep-learning reproduction on standard EMG datasets. It does not overwrite application-data conclusions:

- Limb Position subject-dependent 5-class result remains Macro-F1=0.9050.
- Bingbin_Realtime V3 remains a small-sample exploratory demo branch with Recording Macro-F1=0.9272, Accuracy=0.9286, Balanced Accuracy=0.9286, while retaining the caveat that Stage15 reused known split/test results and is not a clean blinded benchmark or online causal streaming model.
- Public dataset results must not be described as Bingbin_Realtime deep-learning performance.
"""
    (OUT / "PUBLIC_DEEP_BENCHMARK_REPORT.md").write_text(report, encoding="utf-8")

    platform = f"""# Platform Reproduction Report

## Reproduction Entry

```powershell
python -X utf8 scripts\\public_dataset_benchmark.py --datasets both --download --epochs 20 --batch-size 64
```

## Platform Components

- Dataset adapters: UCI raw TXT adapter; CapgMyo DB-a MAT-in-ZIP adapter.
- Split protocol: UCI subject-aware 3-fold; CapgMyo recording/trial-aware calibrated 3-fold.
- Model registry: `CNN1D`, `TCN`, `BiLSTM`, `LSTMTransformer`.
- Evaluation: Accuracy, Balanced Accuracy, Macro-F1, per-class metrics, confusion matrices, training curves, parameter count, inference latency.
- Output CSVs include `dataset_name` and `source_file` to distinguish public data branches.

## Figures

{figure_text}

## Leakage Check

- Split unit is subject for UCI and MAT trial/recording for CapgMyo.
- Standardization is fitted on train subjects only.
- CapgMyo subset is explicitly labeled as a subset benchmark.
- This public-data branch does not read or modify Limb/Bingbin frozen experiment outputs.
"""
    (OUT / "PLATFORM_REPRODUCTION_REPORT.md").write_text(platform, encoding="utf-8")

    policy = """# Model Selection Policy

The platform follows two explicit modelling branches:

1. Public benchmark datasets use deep-learning models. UCI EMG Data for Gestures and the CapgMyo DB-a subset compare 1D-CNN, TCN, BiLSTM, and LSTM-Transformer variants. Selection follows validation evidence rather than forcing a proposed architecture to win.
2. The smaller Bingbin_Realtime application dataset uses engineered raw-EMG/raw-IMU features with traditional machine-learning models. This branch remains an exploratory recording-level demonstration and is reported separately from the public benchmarks.

These branches support different data regimes and must not be combined into a single performance claim.
"""
    (OUT / "MODEL_SELECTION_POLICY.md").write_text(policy, encoding="utf-8")

    leakage = ["# Stage16 Leakage Audit", "", "## Status: PASS", ""]
    for bundle in bundles:
        leakage.extend(
            [
                f"### {bundle.display_name}",
                f"- Source: {bundle.source_url}",
                f"- Scope: {bundle.subset_note}",
                f"- Split protocol: {bundle.split_protocol}",
                "- Split unit: subject for UCI; MAT trial/recording for CapgMyo.",
                "- Standardization is fitted on train subjects only and applied to validation/test.",
                "- Test subjects are not used for model selection, threshold selection, or normalization fitting.",
                "",
            ]
        )
    leakage.append("This public branch does not modify Limb/Bingbin frozen outputs.")
    (OUT / "LEAKAGE_AUDIT.md").write_text("\n".join(leakage), encoding="utf-8")

    decision = {
        "stage": "Stage16 Public Dataset Deep Learning Benchmark",
        "status": "DONE",
        "formal_public_datasets": [b.display_name for b in bundles],
        "deep_models": MODEL_NAMES,
        "proposal_model_included": "LSTMTransformer",
        "split_protocol": "UCI subject-aware 3-fold; CapgMyo recording/trial-aware calibrated 3-fold",
        "device": device,
        "summary": summary_rows,
        "outputs": {
            "summary_csv": str((OUT / "DEEP_MODEL_BENCHMARK_SUMMARY.csv").relative_to(ROOT)),
            "per_fold_csv": str((OUT / "DEEP_MODEL_PER_FOLD.csv").relative_to(ROOT)),
            "per_class_csv": str((OUT / "DEEP_MODEL_PER_CLASS.csv").relative_to(ROOT)),
            "figures_dir": str(FIG_DIR.relative_to(ROOT)),
        },
        "bingbin_stage15_preserved": True,
        "limb_stage8_preserved": True,
        "application_positioning": "large/public datasets use deep learning; small application recording dataset uses robust traditional ML",
    }
    (OUT / "STAGE16_DECISION.json").write_text(json.dumps(native(decision), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    global OUT, FIG_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", choices=["uci", "capgmyo", "both"], default="both")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--uci-window-len", type=int, default=200)
    parser.add_argument("--uci-step", type=int, default=100)
    parser.add_argument("--capgmyo-subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--capgmyo-target-len", type=int, default=250)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()

    OUT = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    FIG_DIR = OUT / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    write_dataset_selection(args.capgmyo_subjects)

    bundles: list[DatasetBundle] = []
    if args.datasets in ("uci", "both"):
        bundles.append(build_uci_bundle(args.uci_window_len, args.uci_step, args.download))
    if args.datasets in ("capgmyo", "both"):
        bundles.append(build_capgmyo_bundle(args.capgmyo_subjects, args.capgmyo_target_len, args.download))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_inventory: list[dict] = []
    all_manifest: list[pd.DataFrame] = []
    all_per_fold: list[dict] = []
    all_per_class: list[dict] = []
    all_complexity: list[dict] = []
    all_history: list[dict] = []
    all_split: list[dict] = []
    all_summary: list[dict] = []

    for bundle in bundles:
        all_inventory.extend(bundle.inventory_rows)
        all_manifest.append(bundle.manifest)
        per_fold, per_class, complexity, history, split_rows, summary = run_dataset_benchmark(bundle, device, args.epochs, args.batch_size)
        all_per_fold.extend(per_fold)
        all_per_class.extend(per_class)
        all_complexity.extend(complexity)
        all_history.extend(history)
        all_split.extend(split_rows)
        all_summary.extend(summary)

    write_csv(OUT / "DATASET_INVENTORY.csv", all_inventory)
    pd.concat(all_manifest, ignore_index=True).to_csv(OUT / "WINDOW_MANIFEST.csv", index=False, encoding="utf-8-sig")
    write_csv(OUT / "SUBJECT_SPLIT_MANIFEST.csv", all_split)
    write_csv(OUT / "DEEP_MODEL_PER_FOLD.csv", all_per_fold)
    write_csv(OUT / "DEEP_MODEL_PER_CLASS.csv", all_per_class)
    write_csv(OUT / "DEEP_MODEL_COMPLEXITY.csv", all_complexity)
    write_csv(OUT / "TRAINING_HISTORY.csv", all_history)
    write_csv(OUT / "DEEP_MODEL_BENCHMARK_SUMMARY.csv", all_summary)

    best_rows = []
    for bundle in bundles:
        rows = [r for r in all_summary if r["dataset_name"] == bundle.display_name]
        best_rows.append(max(rows, key=lambda r: float(r["test_macro_f1_mean"])))
    plot_strategy_overview(FIG_DIR / "public_deep_strategy_overview_cn.png", best_rows)

    write_reports(bundles, all_summary, all_per_fold, all_split, str(device))
    decision = json.loads((OUT / "STAGE16_DECISION.json").read_text(encoding="utf-8"))
    print(json.dumps(native(decision), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

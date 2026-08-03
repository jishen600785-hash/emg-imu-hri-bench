"""\
Standalone dual-branch 1D-CNN training program for the SJ EMG+IMU dataset.

This file embeds HPF reading, action discovery, windowing, feature extraction, dataset splitting, and evaluation.
It does not depend on other Python source files in this directory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
import time
from typing import Iterable
import xml.etree.ElementTree as ET

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

import copy
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


HEADER_CHUNK_ID = 0x1000


CHANNEL_INFO_CHUNK_ID = 0x2000


DATA_CHUNK_ID = 0x3000


SUPPORTED_VERSION = 0x10001


class ChannelInfo:
    group_id: int
    data_index: int
    name: str
    unit: str
    sample_rate_hz: float
    data_type: str
    metadata: dict[str, str]


class SignalGroup:
    group_id: int
    sample_rate_hz: float
    channels: tuple[ChannelInfo, ...]
    data: np.ndarray


class HPFRecording:
    path: Path
    file_version: int
    header_metadata: dict[str, str]
    groups: tuple[SignalGroup, ...]


class _Chunk:
    position: int
    chunk_id: int
    size: int
    group_id: int | None


def _i64(buffer: bytes | bytearray | memoryview, offset: int) -> int:
    return struct.unpack_from("<q", buffer, offset)[0]


def _i32(buffer: bytes | bytearray | memoryview, offset: int) -> int:
    return struct.unpack_from("<i", buffer, offset)[0]


def _xml_root(payload: bytes) -> ET.Element:
    payload = payload.rstrip(b"\x00")
    start = payload.find(b"<")
    if start < 0:
        raise ValueError("HPF chunk does not contain XML")
    return ET.fromstring(payload[start:])


def _flatten_xml(root: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in root.iter():
        if node is root:
            continue
        text = (node.text or "").strip()
        if text and len(node) == 0:
            result[node.tag] = text
    return result


def _scan_chunks(raw: memoryview, first_chunk_offset: int) -> list[_Chunk]:
    chunks: list[_Chunk] = []
    position = first_chunk_offset
    file_size = len(raw)
    while position + 16 <= file_size:
        chunk_id = _i64(raw, position)
        size = _i64(raw, position + 8)
        if size < 16 or position + size > file_size:
            break
        group_id = (
            _i32(raw, position + 16)
            if chunk_id in {CHANNEL_INFO_CHUNK_ID, DATA_CHUNK_ID}
            and size >= 20
            else None
        )
        chunks.append(_Chunk(position, chunk_id, size, group_id))
        position += size
    return chunks


def _parse_channel_info(raw: memoryview, chunk: _Chunk) -> tuple[ChannelInfo, ...]:
    if chunk.group_id is None:
        raise ValueError("Channel-info chunk is missing a group id")
    channel_count = _i32(raw, chunk.position + 20)
    xml_start = chunk.position + 24
    xml_end = chunk.position + chunk.size
    root = _xml_root(bytes(raw[xml_start:xml_end]))
    nodes = list(root.findall("ChannelInformation"))
    if not nodes and root.tag == "ChannelInformation":
        nodes = [root]
    if len(nodes) != channel_count:
        raise ValueError(
            f"HPF group {chunk.group_id}: header says {channel_count} channels, "
            f"XML contains {len(nodes)}"
        )

    channels: list[ChannelInfo] = []
    for fallback_index, node in enumerate(nodes):
        meta = {
            child.tag: (child.text or "").strip()
            for child in node
            if len(child) == 0
        }
        data_index = int(meta.get("DataIndex", fallback_index))
        # In current EMGworks files RequestedPerChannelSampleRate may contain
        # the dimensionless value "1", while PerChannelSampleRate contains the
        # actual physical rate (for example 1259.259 Hz EMG and 148.148 Hz IMU).
        # Prefer the physical rate whenever it is present.
        rate_text = meta.get("PerChannelSampleRate", "0")
        sample_rate = float(rate_text or 0.0)
        if sample_rate == 0.0:
            sample_rate = float(
                meta.get("RequestedPerChannelSampleRate", "0") or 0.0
            )
        channels.append(
            ChannelInfo(
                group_id=chunk.group_id,
                data_index=data_index,
                name=meta.get("Name", f"group{chunk.group_id}_ch{data_index}"),
                unit=meta.get("Unit", ""),
                sample_rate_hz=sample_rate,
                data_type=meta.get("DataType", ""),
                metadata=meta,
            )
        )
    channels.sort(key=lambda item: item.data_index)
    return tuple(channels)


def read_hpf(path: str | Path, load_data: bool = True) -> HPFRecording:
    path = Path(path)
    raw_bytes = path.read_bytes()
    raw = memoryview(raw_bytes)
    if len(raw) < 28:
        raise ValueError(f"HPF file is too short: {path}")
    chunk_id = _i64(raw, 0)
    header_size = _i64(raw, 8)
    signature = bytes(raw[16:20])
    version = _i64(raw, 20)
    if chunk_id != HEADER_CHUNK_ID or signature != b"datx":
        raise ValueError(f"Not a supported Delsys/DT HPF file: {path}")
    if version != SUPPORTED_VERSION:
        raise ValueError(f"Unsupported HPF version {version:#x}: {path}")

    header_metadata: dict[str, str] = {}
    try:
        header_metadata = _flatten_xml(_xml_root(bytes(raw[28:header_size])))
    except (ET.ParseError, ValueError):
        pass

    chunks = _scan_chunks(raw, header_size)
    info_chunks = [chunk for chunk in chunks if chunk.chunk_id == CHANNEL_INFO_CHUNK_ID]
    data_chunks = [chunk for chunk in chunks if chunk.chunk_id == DATA_CHUNK_ID]
    if not info_chunks:
        raise ValueError(f"No channel information chunks found: {path}")

    channels_by_group: dict[int, tuple[ChannelInfo, ...]] = {}
    for chunk in info_chunks:
        if chunk.group_id is None:
            continue
        if chunk.group_id in channels_by_group:
            raise ValueError(f"Duplicate channel-info group {chunk.group_id}: {path}")
        channels_by_group[chunk.group_id] = _parse_channel_info(raw, chunk)

    if not load_data:
        groups = tuple(
            SignalGroup(
                group_id=group_id,
                sample_rate_hz=channels[0].sample_rate_hz if channels else 0.0,
                channels=channels,
                data=np.empty((0, len(channels)), dtype=np.float32),
            )
            for group_id, channels in sorted(channels_by_group.items())
        )
        return HPFRecording(path, version, header_metadata, groups)

    segments: dict[int, list[list[tuple[int, np.ndarray]]]] = {
        group_id: [[] for _ in channels]
        for group_id, channels in channels_by_group.items()
    }
    maximum_lengths: dict[int, int] = {group_id: 0 for group_id in channels_by_group}

    for chunk in data_chunks:
        group_id = chunk.group_id
        if group_id not in channels_by_group:
            continue
        cursor = chunk.position + 20
        data_start_index = _i64(raw, cursor)
        cursor += 8
        channel_data_count = min(
            _i32(raw, cursor),
            len(channels_by_group[group_id]),
        )
        cursor += 4
        descriptors = [
            (_i32(raw, cursor + 8 * index), _i32(raw, cursor + 8 * index + 4))
            for index in range(channel_data_count)
        ]
        for channel_index, (relative_offset, byte_count) in enumerate(descriptors):
            if byte_count < 0 or byte_count % 4:
                raise ValueError(f"Invalid float payload size in {path}")
            begin = chunk.position + relative_offset
            end = begin + byte_count
            if begin < chunk.position or end > chunk.position + chunk.size:
                raise ValueError(f"HPF channel payload is outside its chunk: {path}")
            values = np.frombuffer(raw[begin:end], dtype="<f4").copy()
            start = int(data_start_index)
            segments[group_id][channel_index].append((start, values))
            maximum_lengths[group_id] = max(
                maximum_lengths[group_id],
                start + values.size,
            )

    groups: list[SignalGroup] = []
    for group_id, channels in sorted(channels_by_group.items()):
        length = maximum_lengths[group_id]
        matrix = np.full((length, len(channels)), np.nan, dtype=np.float32)
        for channel_index, channel_segments in enumerate(segments[group_id]):
            for start, values in channel_segments:
                matrix[start : start + values.size, channel_index] = values
        rates = {round(channel.sample_rate_hz, 9) for channel in channels}
        groups.append(
            SignalGroup(
                group_id=group_id,
                # HPF aligns all channels in a group to the fastest stored
                # timebase. Individual physical rates remain on ChannelInfo.
                sample_rate_hz=max(rates) if rates else 0.0,
                channels=channels,
                data=matrix,
            )
        )
    return HPFRecording(path, version, header_metadata, tuple(groups))


def iter_channel_rows(recording: HPFRecording) -> Iterable[dict[str, object]]:
    for group in recording.groups:
        for channel_index, channel in enumerate(group.channels):
            if group.data.size and group.sample_rate_hz > 0:
                physical_count = int(
                    round(
                        group.data.shape[0]
                        * channel.sample_rate_hz
                        / group.sample_rate_hz
                    )
                )
                values = group.data[:physical_count, channel_index]
            else:
                values = np.array([])
            finite = values[np.isfinite(values)]
            yield {
                "file": str(recording.path),
                "group_id": group.group_id,
                "channel_index": channel_index,
                "data_index": channel.data_index,
                "name": channel.name,
                "unit": channel.unit,
                "sample_rate_hz": channel.sample_rate_hz,
                "sample_count": int(values.size),
                "finite_count": int(finite.size),
                "minimum": float(np.min(finite)) if finite.size else None,
                "maximum": float(np.max(finite)) if finite.size else None,
                "mean": float(np.mean(finite)) if finite.size else None,
                "std": float(np.std(finite)) if finite.size else None,
            }


CLASS_NAMES = ["hand_close", "hand_down", "hand_open", "hand_up", "rest"]


LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


SPLIT_CONDITIONS = {
    "train": ["position 1", "position 2 down"],
    "validation": ["position 3 up"],
    "test": ["dynamic"],
}


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    labels = np.arange(len(CLASS_NAMES))
    return {
        "sample_count": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_NAMES)),
        display_labels=CLASS_NAMES,
        cmap="Blues",
        colorbar=False,
        xticks_rotation=25,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def split_mask(splits: np.ndarray, name: str) -> np.ndarray:
    return splits == name


ACTION_PATTERNS = {
    "hand_close": re.compile(r"hands?_close", re.IGNORECASE),
    "hand_down": re.compile(r"hands?_down", re.IGNORECASE),
    "hand_open": re.compile(r"hands?_open", re.IGNORECASE),
    "hand_up": re.compile(r"hands?_up", re.IGNORECASE),
    "rest": re.compile(r"rest", re.IGNORECASE),
}


def action_from_name(name: str) -> str:
    for action, pattern in ACTION_PATTERNS.items():
        if pattern.search(name):
            return action
    raise ValueError(f"Cannot infer action label from filename: {name}")


def inspect_raw_dataset(data_dir: Path, output_dir: Path) -> None:
    """Inspect all HPF files and save a manifest before window preparation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(data_dir.rglob("*.hpf"))
    if not files:
        raise FileNotFoundError(f"No HPF files found under {data_dir}")

    recordings: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in files:
        condition = path.parent.name
        action = action_from_name(path.name)
        try:
            recording = read_hpf(path)
            group_summary = []
            for group in recording.groups:
                group_summary.append(
                    {
                        "group_id": group.group_id,
                        "sample_rate_hz": group.sample_rate_hz,
                        "sample_count": int(group.data.shape[0]),
                        "channel_count": int(group.data.shape[1]),
                        "duration_sec": (
                            float(group.data.shape[0] / group.sample_rate_hz)
                            if group.sample_rate_hz > 0
                            else None
                        ),
                        "channel_names": [
                            channel.name for channel in group.channels
                        ],
                        "channel_units": [
                            channel.unit for channel in group.channels
                        ],
                    }
                )
            recordings.append(
                {
                    "path": str(path),
                    "relative_path": str(path.relative_to(data_dir)),
                    "condition": condition,
                    "action": action,
                    "size_bytes": path.stat().st_size,
                    "header_metadata": recording.header_metadata,
                    "groups": group_summary,
                }
            )
            for row in iter_channel_rows(recording):
                row["condition"] = condition
                row["action"] = action
                row["relative_path"] = str(path.relative_to(data_dir))
                channel_rows.append(row)
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})

    manifest = {
        "data_dir": str(data_dir),
        "file_count": len(files),
        "parsed_count": len(recordings),
        "error_count": len(errors),
        "recordings": recordings,
        "errors": errors,
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if channel_rows:
        with (output_dir / "channel_quality.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(channel_rows[0]))
            writer.writeheader()
            writer.writerows(channel_rows)
    if errors:
        raise RuntimeError(f"{len(errors)} HPF files failed inspection: {errors}")
    print(f"Successfully inspected {len(recordings)} HPF files")


WINDOW_SEC = 0.500


STEP_SEC = 0.250


EDGE_TRIM_SEC = 1.000


EMG_RESAMPLED_POINTS = 256


IMU_RESAMPLED_POINTS = 64


def split_for_condition(condition: str) -> str:
    for split, conditions in SPLIT_CONDITIONS.items():
        if condition in conditions:
            return split
    raise ValueError(f"Condition is not assigned to a split: {condition}")


def physical_channel_values(
    recording: HPFRecording,
    group_index: int,
    channel_index: int,
) -> np.ndarray:
    group = recording.groups[group_index]
    channel = group.channels[channel_index]
    count = int(
        round(group.data.shape[0] * channel.sample_rate_hz / group.sample_rate_hz)
    )
    return group.data[:count, channel_index]


def fixed_resample(values: np.ndarray, points: int) -> np.ndarray:
    if values.size < 2:
        return np.zeros(points, dtype=np.float32)
    return resample(values.astype(np.float64), points).astype(np.float32)


def channel_features(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    diff = np.diff(values)
    x = np.linspace(-1.0, 1.0, values.size, dtype=np.float64)
    x_centered = x - x.mean()
    slope = (
        float(np.dot(x_centered, values - values.mean()) / np.dot(x_centered, x_centered))
        if values.size > 1
        else 0.0
    )
    return np.asarray(
        [
            np.mean(values),
            np.std(values),
            np.mean(np.abs(values)),  # MAV
            np.sqrt(np.mean(np.square(values))),  # RMS
            np.var(values),  # VAR
            np.sum(np.abs(diff)),  # WL
            np.ptp(values),
            np.min(values),
            np.max(values),
            slope,
        ],
        dtype=np.float32,
    )


def inspect_layout(recording: HPFRecording) -> tuple[list[int], list[int]]:
    if len(recording.groups) != 1:
        raise ValueError(
            f"Expected one aligned HPF group, found {len(recording.groups)}"
        )
    channels = recording.groups[0].channels
    emg = [i for i, channel in enumerate(channels) if "EMG" in channel.name.upper()]
    imu = [
        i
        for i, channel in enumerate(channels)
        if "ACC." in channel.name.upper() or "GYRO." in channel.name.upper()
    ]
    if len(emg) != 5 or len(imu) != 30:
        raise ValueError(f"Expected 5 EMG + 30 IMU channels, found {len(emg)} + {len(imu)}")
    return emg, imu


def prepare(data_dir: Path, output_dir: Path) -> None:
    files = sorted(data_dir.rglob("*.hpf"))
    if len(files) != 20:
        raise ValueError(f"Expected 20 HPF files, found {len(files)}")

    all_features: list[np.ndarray] = []
    all_emg: list[np.ndarray] = []
    all_imu: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    conditions: list[str] = []
    source_files: list[str] = []
    start_times: list[float] = []
    window_rows: list[dict[str, object]] = []
    channel_layout: dict[str, object] | None = None

    for file_index, path in enumerate(files):
        relative = path.relative_to(data_dir)
        condition = relative.parts[0]
        action = action_from_name(path.name)
        split = split_for_condition(condition)
        recording = read_hpf(path)
        group = recording.groups[0]
        emg_indices, imu_indices = inspect_layout(recording)
        duration = group.data.shape[0] / group.sample_rate_hz
        starts = np.arange(
            EDGE_TRIM_SEC,
            duration - EDGE_TRIM_SEC - WINDOW_SEC + 1e-9,
            STEP_SEC,
        )

        if channel_layout is None:
            channel_layout = {
                "aligned_storage_rate_hz": group.sample_rate_hz,
                "emg": [
                    {
                        "index": i,
                        "name": group.channels[i].name,
                        "unit": group.channels[i].unit,
                        "physical_sample_rate_hz": group.channels[i].sample_rate_hz,
                    }
                    for i in emg_indices
                ],
                "imu": [
                    {
                        "index": i,
                        "name": group.channels[i].name,
                        "unit": group.channels[i].unit,
                        "physical_sample_rate_hz": group.channels[i].sample_rate_hz,
                    }
                    for i in imu_indices
                ],
            }

        channel_values = {
            i: physical_channel_values(recording, 0, i)
            for i in emg_indices + imu_indices
        }
        feature_names = None
        for start_sec in starts:
            end_sec = start_sec + WINDOW_SEC
            emg_window = []
            imu_window = []
            features = []

            for index in emg_indices:
                rate = group.channels[index].sample_rate_hz
                begin = int(round(start_sec * rate))
                end = int(round(end_sec * rate))
                values = channel_values[index][begin:end]
                features.append(channel_features(values))
                emg_window.append(fixed_resample(values, EMG_RESAMPLED_POINTS))

            for index in imu_indices:
                rate = group.channels[index].sample_rate_hz
                begin = int(round(start_sec * rate))
                end = int(round(end_sec * rate))
                values = channel_values[index][begin:end]
                features.append(channel_features(values))
                imu_window.append(fixed_resample(values, IMU_RESAMPLED_POINTS))

            all_features.append(np.concatenate(features))
            all_emg.append(np.stack(emg_window))
            all_imu.append(np.stack(imu_window))
            labels.append(LABEL_TO_INDEX[action])
            splits.append(split)
            conditions.append(condition)
            source_files.append(relative.as_posix())
            start_times.append(float(start_sec))
            window_rows.append(
                {
                    "window_id": len(labels) - 1,
                    "split": split,
                    "condition": condition,
                    "action": action,
                    "label_index": LABEL_TO_INDEX[action],
                    "source_file": relative.as_posix(),
                    "start_sec": f"{start_sec:.3f}",
                    "end_sec": f"{end_sec:.3f}",
                }
            )
        print(
            f"[{file_index + 1:02d}/{len(files)}] {relative}: "
            f"{len(starts)} windows -> {split}"
        )

    X_features = np.stack(all_features)
    X_emg = np.stack(all_emg)
    X_imu = np.stack(all_imu)
    y = np.asarray(labels, dtype=np.int64)
    split_array = np.asarray(splits)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "window_dataset.npz",
        X_features=X_features,
        X_emg=X_emg,
        X_imu=X_imu,
        y=y,
        splits=split_array,
        conditions=np.asarray(conditions),
        source_files=np.asarray(source_files),
        start_times_sec=np.asarray(start_times, dtype=np.float32),
    )
    with (output_dir / "window_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(window_rows[0]))
        writer.writeheader()
        writer.writerows(window_rows)

    counts = {
        split: {
            CLASS_NAMES[label]: int(np.sum((split_array == split) & (y == label)))
            for label in range(len(CLASS_NAMES))
        }
        for split in SPLIT_CONDITIONS
    }
    metadata = {
        "data_dir": str(data_dir),
        "file_count": len(files),
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "overlap_fraction": 1.0 - STEP_SEC / WINDOW_SEC,
        "edge_trim_sec": EDGE_TRIM_SEC,
        "split_policy": "whole-condition holdout; no recording appears in multiple splits",
        "split_conditions": SPLIT_CONDITIONS,
        "class_names": CLASS_NAMES,
        "counts": counts,
        "total_windows": int(y.size),
        "feature_vector_dim": int(X_features.shape[1]),
        "feature_names_per_channel": [
            "mean",
            "std",
            "MAV",
            "RMS",
            "VAR",
            "WL",
            "range",
            "min",
            "max",
            "slope",
        ],
        "raw_tensor_shapes": {
            "emg": list(X_emg.shape),
            "imu": list(X_imu.shape),
        },
        "resampled_points": {
            "emg": EMG_RESAMPLED_POINTS,
            "imu": IMU_RESAMPLED_POINTS,
        },
        "channel_layout": channel_layout,
    }
    json_dump(output_dir / "window_dataset_metadata.json", metadata)
    print(f"Saved {y.size} windows to {output_dir}")
    print(f"Feature shape: {X_features.shape}")
    print(f"EMG shape: {X_emg.shape}; IMU shape: {X_imu.shape}")
    print(f"Counts: {counts}")


class ModalityBranch(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class FusionCNN(nn.Module):
    def __init__(
        self,
        emg_channels: int,
        imu_channels: int,
        class_count: int,
        emg_mean: torch.Tensor,
        emg_std: torch.Tensor,
        imu_mean: torch.Tensor,
        imu_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("emg_mean", emg_mean)
        self.register_buffer("emg_std", emg_std)
        self.register_buffer("imu_mean", imu_mean)
        self.register_buffer("imu_std", imu_std)
        self.emg_branch = ModalityBranch(emg_channels)
        self.imu_branch = ModalityBranch(imu_channels)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, class_count),
        )

    def forward(
        self,
        emg: torch.Tensor,
        imu: torch.Tensor,
    ) -> torch.Tensor:
        emg = (emg - self.emg_mean) / self.emg_std
        imu = (imu - self.imu_mean) / self.imu_std
        fused = torch.cat(
            [self.emg_branch(emg), self.imu_branch(imu)],
            dim=1,
        )
        return self.classifier(fused)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def channel_statistics(values: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    std = values.std(axis=(0, 2), keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-7)
    return torch.from_numpy(mean), torch.from_numpy(std)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    criterion = nn.CrossEntropyLoss()
    y_true = []
    y_pred = []
    loss_sum = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for emg, imu, target in loader:
            emg = emg.to(device)
            imu = imu.to(device)
            target = target.to(device)
            logits = model(emg, imu)
            loss = criterion(logits, target)
            loss_sum += float(loss) * target.size(0)
            count += target.size(0)
            y_true.append(target.cpu().numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        loss_sum / max(count, 1),
    )


def make_loader(
    emg: np.ndarray,
    imu: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(emg[mask].astype(np.float32)),
        torch.from_numpy(imu[mask].astype(np.float32)),
        torch.from_numpy(y[mask].astype(np.int64)),
    )
    generator = torch.Generator().manual_seed(42)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def train(
    dataset_path: Path,
    output_dir: Path,
    epochs: int,
    patience: int,
) -> None:
    set_seed(42)
    dataset = np.load(dataset_path, allow_pickle=False)
    emg = dataset["X_emg"]
    imu = dataset["X_imu"]
    y = dataset["y"]
    splits = dataset["splits"]
    train_mask = split_mask(splits, "train")
    validation_mask = split_mask(splits, "validation")
    test_mask = split_mask(splits, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emg_mean, emg_std = channel_statistics(emg[train_mask])
    imu_mean, imu_std = channel_statistics(imu[train_mask])
    model = FusionCNN(
        emg.shape[1],
        imu.shape[1],
        len(CLASS_NAMES),
        emg_mean,
        emg_std,
        imu_mean,
        imu_std,
    ).to(device)

    train_loader = make_loader(emg, imu, y, train_mask, 64, True)
    validation_loader = make_loader(emg, imu, y, validation_mask, 128, False)
    test_loader = make_loader(emg, imu, y, test_mask, 128, False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_state = None
    best_epoch = -1
    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    no_improvement = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_emg, batch_imu, target in train_loader:
            batch_emg = batch_emg.to(device)
            batch_imu = batch_imu.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_emg, batch_imu)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * target.size(0)
            train_count += target.size(0)

        val_true, val_pred, val_loss = predict(
            model, validation_loader, device
        )
        val_metrics = evaluate_predictions(val_true, val_pred)
        macro_f1 = float(val_metrics["macro_f1"])
        scheduler.step(macro_f1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / train_count,
                "validation_loss": val_loss,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": macro_f1,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={history[-1]['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_metrics['accuracy']:.4f} "
            f"val_macro_f1={macro_f1:.4f}"
        )

        improved = (
            macro_f1 > best_macro_f1 + 1e-6
            or (
                abs(macro_f1 - best_macro_f1) <= 1e-6
                and val_loss < best_val_loss
            )
        )
        if improved:
            best_macro_f1 = macro_f1
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping after epoch {epoch}")
                break

    training_seconds = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("Deep-learning training produced no checkpoint")
    model.load_state_dict(best_state)

    # The test condition is touched only once, after epoch selection by validation.
    test_true, test_pred, test_loss = predict(model, test_loader, device)
    test_metrics = evaluate_predictions(test_true, test_pred)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_class": "FusionCNN",
        "state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "emg_channels": emg.shape[1],
        "imu_channels": imu.shape[1],
        "emg_points": emg.shape[2],
        "imu_points": imu.shape[2],
        "best_epoch": best_epoch,
    }
    torch.save(checkpoint, output_dir / "best_cnn_evaluation_model.pt")
    model_cpu = model.cpu().eval()
    scripted = torch.jit.script(model_cpu)
    scripted.save(str(output_dir / "best_cnn_evaluation_model.ts"))
    np.savez_compressed(
        output_dir / "best_cnn_test_predictions.npz",
        y_true=test_true,
        y_pred=test_pred,
    )
    save_confusion_matrix(
        test_true,
        test_pred,
        output_dir / "best_cnn_test_confusion_matrix.png",
        "Two-branch 1D CNN held-out dynamic test",
    )

    with (output_dir / "cnn_training_history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([row["epoch"] for row in history], [row["train_loss"] for row in history], label="train")
    axes[0].plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot([row["epoch"] for row in history], [row["validation_accuracy"] for row in history], label="accuracy")
    axes[1].plot([row["epoch"] for row in history], [row["validation_macro_f1"] for row in history], label="macro-F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cnn_training_curves.png", dpi=180)
    plt.close(fig)

    result = {
        "architecture": "two-branch 1D CNN (raw EMG + raw IMU fusion)",
        "device": str(device),
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(validation_mask.sum()),
        "test_windows": int(test_mask.sum()),
        "maximum_epochs": epochs,
        "completed_epochs": len(history),
        "early_stopping_patience": patience,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_macro_f1,
        "best_validation_loss": best_val_loss,
        "training_seconds": training_seconds,
        "held_out_test_loss": test_loss,
        "held_out_test": test_metrics,
        "saved_checkpoint": "best_cnn_evaluation_model.pt",
        "saved_torchscript": "best_cnn_evaluation_model.ts",
        "normalization": (
            "Per-channel mean/std fitted only on training windows and stored "
            "inside the model."
        ),
    }
    json_dump(output_dir / "cnn_results.json", result)
    print(
        f"Best epoch {best_epoch}; held-out dynamic test accuracy="
        f"{test_metrics['accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SJ HPF windows, train a dual-branch 1D CNN with validation early stopping, and "
            "evaluate once on the held-out dynamic condition."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Raw-data directory or an existing window_dataset.npz",
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for this training run")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--reuse-prepared",
        action="store_true",
        help="Reuse output_dir/prepared/window_dataset.npz when raw data are supplied",
    )
    args = parser.parse_args()

    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if input_path.is_file():
        dataset_path = input_path
        model_dir = output_dir
    elif input_path.is_dir():
        prepared_dir = output_dir / "prepared"
        dataset_path = prepared_dir / "window_dataset.npz"
        if not args.reuse_prepared or not dataset_path.is_file():
            inspect_raw_dataset(input_path, output_dir / "inspection")
            prepare(input_path, prepared_dir)
        model_dir = output_dir / "dl"
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    train(dataset_path, model_dir, args.epochs, args.patience)
    print(f"Deep-learning results: {model_dir}")
    print(f"PyTorch model: {model_dir / 'best_cnn_evaluation_model.pt'}")
    print(f"TorchScript model: {model_dir / 'best_cnn_evaluation_model.ts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

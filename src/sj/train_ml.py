"""\
Standalone comparison of five traditional machine-learning models for the SJ EMG+IMU dataset.

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

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


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


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "shrinkage_lda": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LinearDiscriminantAnalysis(
                        solver="lsqr",
                        shrinkage="auto",
                    ),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=5.0,
                        gamma="scale",
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }


def train(dataset_path: Path, output_dir: Path) -> None:
    dataset = np.load(dataset_path, allow_pickle=False)
    X = dataset["X_features"]
    y = dataset["y"]
    splits = dataset["splits"]
    train_mask = split_mask(splits, "train")
    validation_mask = split_mask(splits, "validation")
    test_mask = split_mask(splits, "test")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    validation_details: dict[str, object] = {}
    fitted_models: dict[str, object] = {}

    for name, model in candidate_models().items():
        started = time.perf_counter()
        model.fit(X[train_mask], y[train_mask])
        fit_seconds = time.perf_counter() - started
        validation_pred = model.predict(X[validation_mask])
        metrics = evaluate_predictions(y[validation_mask], validation_pred)
        validation_details[name] = {
            "fit_seconds": fit_seconds,
            "validation": metrics,
        }
        fitted_models[name] = model
        rows.append(
            {
                "model": name,
                "fit_seconds": fit_seconds,
                "validation_accuracy": metrics["accuracy"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
            }
        )
        print(
            f"{name}: validation accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

    rows.sort(
        key=lambda row: (
            float(row["validation_macro_f1"]),
            float(row["validation_accuracy"]),
        ),
        reverse=True,
    )
    selected_name = str(rows[0]["model"])
    selected_model = fitted_models[selected_name]
    test_pred = selected_model.predict(X[test_mask])
    test_metrics = evaluate_predictions(y[test_mask], test_pred)

    joblib.dump(selected_model, output_dir / "best_ml_evaluation_model.joblib")
    np.savez_compressed(
        output_dir / "best_ml_test_predictions.npz",
        y_true=y[test_mask],
        y_pred=test_pred,
    )
    save_confusion_matrix(
        y[test_mask],
        test_pred,
        output_dir / "best_ml_test_confusion_matrix.png",
        f"Machine learning test set: {selected_name}",
    )

    with (output_dir / "ml_validation_leaderboard.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "selection_rule": (
            "Select highest validation macro-F1; test split is evaluated once "
            "after model selection."
        ),
        "class_names": CLASS_NAMES,
        "train_windows": int(train_mask.sum()),
        "validation_windows": int(validation_mask.sum()),
        "test_windows": int(test_mask.sum()),
        "selected_model": selected_name,
        "validation_candidates": validation_details,
        "held_out_test": test_metrics,
        "saved_model": "best_ml_evaluation_model.joblib",
        "important_note": (
            "The saved evaluation model was fitted only on the training "
            "conditions. No scaler or classifier was refit on validation/test."
        ),
    }
    json_dump(output_dir / "ml_results.json", result)
    print(
        f"Selected {selected_name}; held-out dynamic test accuracy="
        f"{test_metrics['accuracy']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}"
    )


STATIC_CONDITIONS = ["position 1", "position 2 down", "position 3 up"]


TEST_CONDITION = "dynamic"


EPSILON = 1e-10


def _time_features(x: np.ndarray) -> np.ndarray:
    """Return orientation-independent time features for (N, C, T)."""
    diff = np.diff(x, axis=2)
    mean = np.mean(x, axis=2)
    std = np.std(x, axis=2)
    mav = np.mean(np.abs(x), axis=2)
    rms = np.sqrt(np.mean(np.square(x), axis=2))
    var = np.var(x, axis=2)
    wl = np.mean(np.abs(diff), axis=2)
    value_range = np.ptp(x, axis=2)
    diff_std = np.std(diff, axis=2)
    return np.stack(
        [mean, std, mav, rms, var, wl, value_range, diff_std],
        axis=2,
    )


def _emg_frequency_features(emg: np.ndarray) -> np.ndarray:
    """Frequency-shape features; amplitude is normalized per channel."""
    centered = emg - emg.mean(axis=2, keepdims=True)
    spectrum = np.abs(np.fft.rfft(centered, axis=2)) ** 2
    power = spectrum[..., 1:] + EPSILON
    frequencies = np.fft.rfftfreq(emg.shape[2], d=0.5 / emg.shape[2])[1:]
    total = power.sum(axis=2)
    centroid = (power * frequencies).sum(axis=2) / total
    cumulative = np.cumsum(power, axis=2)
    median_index = np.argmax(cumulative >= total[..., None] * 0.5, axis=2)
    median_frequency = frequencies[median_index]
    bands = []
    for low, high in ((20, 60), (60, 120), (120, 200), (200, 256)):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(power[..., mask].sum(axis=2) / total)
    return np.stack(
        [centroid / 256.0, median_frequency / 256.0, *bands],
        axis=2,
    )


def engineer_features(
    X_features: np.ndarray,
    X_emg: np.ndarray,
    X_imu: np.ndarray,
) -> dict[str, np.ndarray]:
    # prepare_windows.py stores the 5 EMG channels first, ten features each.
    emg_basic = X_features[:, :50].reshape(-1, 5, 10).astype(np.float64)
    positive_indices = [1, 2, 3, 4, 5, 6]
    log_positive = np.log(np.abs(emg_basic[:, :, positive_indices]) + EPSILON)
    # Remove global amplitude offsets while preserving relative muscle activity.
    relative_log = log_positive - log_positive.mean(axis=1, keepdims=True)
    emg_shape = np.concatenate(
        [
            relative_log.reshape(X_features.shape[0], -1),
            _emg_frequency_features(X_emg.astype(np.float64)).reshape(
                X_features.shape[0], -1
            ),
        ],
        axis=1,
    )

    # IMU ordering is sensor1(ACC xyz, GYRO xyz), ..., sensor5.
    imu = X_imu.astype(np.float64).reshape(-1, 5, 6, X_imu.shape[2])
    acceleration_magnitude = np.linalg.norm(imu[:, :, 0:3, :], axis=2)
    gyroscope_magnitude = np.linalg.norm(imu[:, :, 3:6, :], axis=2)
    # Remove each window's orientation/static offset before dynamic features.
    acceleration_dynamic = (
        acceleration_magnitude
        - acceleration_magnitude.mean(axis=2, keepdims=True)
    )
    gyro_dynamic = gyroscope_magnitude - gyroscope_magnitude.mean(
        axis=2, keepdims=True
    )
    imu_invariant = np.concatenate(
        [
            _time_features(acceleration_magnitude),
            _time_features(gyroscope_magnitude),
            _time_features(acceleration_dynamic),
            _time_features(gyro_dynamic),
        ],
        axis=2,
    ).reshape(X_features.shape[0], -1)

    emg_time = _time_features(X_emg.astype(np.float64)).reshape(
        X_features.shape[0], -1
    )
    return {
        "emg_basic": emg_basic.reshape(X_features.shape[0], -1).astype(np.float32),
        "emg_relative_frequency": emg_shape.astype(np.float32),
        "emg_time_frequency": np.concatenate(
            [emg_time, emg_shape], axis=1
        ).astype(np.float32),
        "emg_plus_imu_invariant": np.concatenate(
            [emg_time, emg_shape, imu_invariant], axis=1
        ).astype(np.float32),
    }


def model_factories() -> dict[str, callable]:
    def rbf(c: float) -> Pipeline:
        return Pipeline(
            [
                ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
                (
                    "classifier",
                    SVC(
                        C=c,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        )

    def extra_trees() -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    def random_forest() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )

    def hist_gradient() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        max_leaf_nodes=31,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        )

    def soft_vote() -> VotingClassifier:
        return VotingClassifier(
            estimators=[
                ("svm", rbf(5.0)),
                ("extra", extra_trees()),
                ("forest", random_forest()),
            ],
            voting="soft",
            weights=[2, 2, 1],
            n_jobs=-1,
        )

    return {
        "rbf_svm_c1": lambda: rbf(1.0),
        "rbf_svm_c5": lambda: rbf(5.0),
        "rbf_svm_c20": lambda: rbf(20.0),
        "extra_trees": extra_trees,
        "random_forest": random_forest,
        "hist_gradient_boosting": hist_gradient,
        "soft_voting_ensemble": soft_vote,
    }


def select_model(
    feature_sets: dict[str, np.ndarray],
    y: np.ndarray,
    conditions: np.ndarray,
) -> tuple[str, str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for feature_name, X in feature_sets.items():
        for model_name, factory in model_factories().items():
            fold_accuracies = []
            fold_f1 = []
            started = time.perf_counter()
            for held_out in STATIC_CONDITIONS:
                train_mask = np.isin(
                    conditions,
                    [name for name in STATIC_CONDITIONS if name != held_out],
                )
                validation_mask = conditions == held_out
                model = factory()
                model.fit(X[train_mask], y[train_mask])
                prediction = model.predict(X[validation_mask])
                fold_accuracies.append(
                    accuracy_score(y[validation_mask], prediction)
                )
                fold_f1.append(
                    f1_score(
                        y[validation_mask],
                        prediction,
                        average="macro",
                    )
                )
            row = {
                "feature_set": feature_name,
                "model": model_name,
                "mean_static_lopo_accuracy": float(np.mean(fold_accuracies)),
                "std_static_lopo_accuracy": float(np.std(fold_accuracies)),
                "mean_static_lopo_macro_f1": float(np.mean(fold_f1)),
                "std_static_lopo_macro_f1": float(np.std(fold_f1)),
                "position1_accuracy": float(fold_accuracies[0]),
                "position2down_accuracy": float(fold_accuracies[1]),
                "position3up_accuracy": float(fold_accuracies[2]),
                "selection_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(
                f"{feature_name:26s} {model_name:22s} "
                f"CV acc={row['mean_static_lopo_accuracy']:.4f} "
                f"macro_f1={row['mean_static_lopo_macro_f1']:.4f}"
            )

    rows.sort(
        key=lambda row: (
            float(row["mean_static_lopo_macro_f1"]),
            float(row["mean_static_lopo_accuracy"]),
        ),
        reverse=True,
    )
    return str(rows[0]["feature_set"]), str(rows[0]["model"]), rows


def train_improved_model(dataset_path: Path, output_dir: Path) -> None:
    dataset = np.load(dataset_path, allow_pickle=False)
    y = dataset["y"]
    conditions = dataset["conditions"]
    feature_sets = engineer_features(
        dataset["X_features"],
        dataset["X_emg"],
        dataset["X_imu"],
    )
    feature_name, model_name, leaderboard = select_model(
        feature_sets, y, conditions
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "strict_improvement_leaderboard.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(leaderboard[0]))
        writer.writeheader()
        writer.writerows(leaderboard)

    X = feature_sets[feature_name]
    static_mask = np.isin(conditions, STATIC_CONDITIONS)
    test_mask = conditions == TEST_CONDITION
    selected = model_factories()[model_name]()
    started = time.perf_counter()
    selected.fit(X[static_mask], y[static_mask])
    fit_seconds = time.perf_counter() - started
    test_prediction = selected.predict(X[test_mask])
    test_metrics = evaluate_predictions(y[test_mask], test_prediction)

    bundle = {
        "model": selected,
        "feature_set": feature_name,
        "class_names": CLASS_NAMES,
        "static_conditions": STATIC_CONDITIONS,
        "test_condition": TEST_CONDITION,
    }
    joblib.dump(bundle, output_dir / "improved_strict_model.joblib")
    np.savez_compressed(
        output_dir / "improved_strict_test_predictions.npz",
        y_true=y[test_mask],
        y_pred=test_prediction,
    )
    save_confusion_matrix(
        y[test_mask],
        test_prediction,
        output_dir / "improved_strict_test_confusion_matrix.png",
        f"Improved strict dynamic test: {model_name}",
    )
    result = {
        "protocol": (
            "Select features/model by leave-one-position-out CV over the three "
            "static positions, refit on all static positions, evaluate dynamic "
            "condition once."
        ),
        "selected_feature_set": feature_name,
        "selected_model": model_name,
        "selection_mean_static_lopo_accuracy": leaderboard[0][
            "mean_static_lopo_accuracy"
        ],
        "selection_mean_static_lopo_macro_f1": leaderboard[0][
            "mean_static_lopo_macro_f1"
        ],
        "fit_seconds": fit_seconds,
        "training_windows": int(static_mask.sum()),
        "dynamic_test_windows": int(test_mask.sum()),
        "dynamic_test": test_metrics,
        "leakage_controls": [
            "Model selection uses static conditions only.",
            "Each validation fold holds out a whole acquisition condition.",
            "No normalization is fitted on the dynamic test condition.",
            "The dynamic condition is evaluated after selection.",
        ],
    }
    json_dump(output_dir / "strict_improvement_results.json", result)
    print(
        f"SELECTED feature={feature_name}, model={model_name}; "
        f"dynamic test accuracy={test_metrics['accuracy']:.4f}, "
        f"macro_f1={test_metrics['macro_f1']:.4f}"
    )


WINDOW_CANDIDATES = [1, 3, 5, 7, 11, 15, 21, 31]


def causal_probability_average(
    probabilities: np.ndarray,
    source_files: np.ndarray,
    start_times: np.ndarray,
    history_windows: int,
) -> np.ndarray:
    prediction = np.empty(probabilities.shape[0], dtype=np.int64)
    for source_file in np.unique(source_files):
        indices = np.flatnonzero(source_files == source_file)
        indices = indices[np.argsort(start_times[indices])]
        local = probabilities[indices]
        cumulative = np.vstack(
            [np.zeros((1, local.shape[1])), np.cumsum(local, axis=0)]
        )
        averaged = np.empty_like(local)
        for position in range(local.shape[0]):
            begin = max(0, position + 1 - history_windows)
            averaged[position] = (
                cumulative[position + 1] - cumulative[begin]
            ) / (position + 1 - begin)
        prediction[indices] = averaged.argmax(axis=1)
    return prediction


def train_temporal_smoothing(dataset_path: Path, improvement_dir: Path) -> None:
    dataset = np.load(dataset_path, allow_pickle=False)
    y = dataset["y"]
    conditions = dataset["conditions"]
    source_files = dataset["source_files"]
    start_times = dataset["start_times_sec"]
    feature_sets = engineer_features(
        dataset["X_features"],
        dataset["X_emg"],
        dataset["X_imu"],
    )
    strict_result = __import__("json").loads(
        (improvement_dir / "strict_improvement_results.json").read_text(
            encoding="utf-8"
        )
    )
    feature_name = strict_result["selected_feature_set"]
    model_name = strict_result["selected_model"]
    X = feature_sets[feature_name]

    candidate_scores = {window: [] for window in WINDOW_CANDIDATES}
    for held_out in STATIC_CONDITIONS:
        train_mask = np.isin(
            conditions,
            [condition for condition in STATIC_CONDITIONS if condition != held_out],
        )
        validation_mask = conditions == held_out
        model = model_factories()[model_name]()
        model.fit(X[train_mask], y[train_mask])
        probabilities = model.predict_proba(X[validation_mask])
        validation_sources = source_files[validation_mask]
        validation_times = start_times[validation_mask]
        validation_y = y[validation_mask]
        for window in WINDOW_CANDIDATES:
            prediction = causal_probability_average(
                probabilities,
                validation_sources,
                validation_times,
                window,
            )
            candidate_scores[window].append(
                {
                    "accuracy": accuracy_score(validation_y, prediction),
                    "macro_f1": f1_score(
                        validation_y, prediction, average="macro"
                    ),
                }
            )

    rows = []
    for window, fold_scores in candidate_scores.items():
        rows.append(
            {
                "history_windows": window,
                "history_seconds": 0.25 * (window - 1) + 0.5,
                "mean_static_lopo_accuracy": float(
                    np.mean([score["accuracy"] for score in fold_scores])
                ),
                "mean_static_lopo_macro_f1": float(
                    np.mean([score["macro_f1"] for score in fold_scores])
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["mean_static_lopo_macro_f1"]),
            float(row["mean_static_lopo_accuracy"]),
            -int(row["history_windows"]),
        ),
        reverse=True,
    )
    selected_window = int(rows[0]["history_windows"])

    bundle = joblib.load(improvement_dir / "improved_strict_model.joblib")
    model = bundle["model"]
    test_mask = conditions == TEST_CONDITION
    raw_probabilities = model.predict_proba(X[test_mask])
    raw_prediction = raw_probabilities.argmax(axis=1)
    smoothed_prediction = causal_probability_average(
        raw_probabilities,
        source_files[test_mask],
        start_times[test_mask],
        selected_window,
    )
    raw_metrics = evaluate_predictions(y[test_mask], raw_prediction)
    smoothed_metrics = evaluate_predictions(y[test_mask], smoothed_prediction)

    bundle["causal_smoothing_history_windows"] = selected_window
    bundle["window_step_seconds"] = 0.25
    bundle["smoothing_note"] = (
        "Average only current and prior predicted probabilities. Reset occurs "
        "at the start of each independent recording, never from ground truth."
    )
    joblib.dump(
        bundle,
        improvement_dir / "improved_strict_smoothed_model.joblib",
    )
    np.savez_compressed(
        improvement_dir / "improved_smoothed_test_predictions.npz",
        y_true=y[test_mask],
        y_pred_raw=raw_prediction,
        y_pred_smoothed=smoothed_prediction,
        probabilities=raw_probabilities,
    )
    save_confusion_matrix(
        y[test_mask],
        smoothed_prediction,
        improvement_dir / "improved_smoothed_test_confusion_matrix.png",
        f"Strict dynamic test + causal smoothing ({selected_window} windows)",
    )
    with (improvement_dir / "temporal_smoothing_selection.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "selection_protocol": (
            "History length selected only by leave-one-static-position-out "
            "cross-validation."
        ),
        "selected_history_windows": selected_window,
        "effective_history_seconds": rows[0]["history_seconds"],
        "selection_mean_static_lopo_accuracy": rows[0][
            "mean_static_lopo_accuracy"
        ],
        "selection_mean_static_lopo_macro_f1": rows[0][
            "mean_static_lopo_macro_f1"
        ],
        "dynamic_test_raw": raw_metrics,
        "dynamic_test_causal_smoothed": smoothed_metrics,
        "online_behavior": (
            "Only current and past probabilities are averaged; no future "
            "window or true test label is used."
        ),
        "limitation": (
            "Each dataset recording contains one sustained action. In a "
            "continuous multi-action stream, smoothing introduces transition "
            "latency and must be evaluated separately."
        ),
    }
    json_dump(improvement_dir / "temporal_smoothing_results.json", result)
    print(
        f"selected history={selected_window} windows; "
        f"raw dynamic accuracy={raw_metrics['accuracy']:.4f}; "
        f"causal-smoothed dynamic accuracy={smoothed_metrics['accuracy']:.4f}, "
        f"macro_f1={smoothed_metrics['macro_f1']:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SJ HPF windows, compare five machine-learning models on validation data, and "
            "evaluate once on the held-out dynamic condition."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Raw-data directory or an existing window_dataset.npz",
    )
    parser.add_argument("output_dir", type=Path, help="Output directory for this training run")
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
        model_dir = output_dir / "ml"
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    train(dataset_path, model_dir)
    optimized_dir = model_dir / "optimized"
    train_improved_model(dataset_path, optimized_dir)
    train_temporal_smoothing(dataset_path, optimized_dir)
    print(f"Machine-learning results: {model_dir}")
    print(
        "Best five-model baseline: "
        f"{model_dir / 'best_ml_evaluation_model.joblib'}"
    )
    print(
        "Final optimized model: "
        f"{optimized_dir / 'improved_strict_smoothed_model.joblib'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

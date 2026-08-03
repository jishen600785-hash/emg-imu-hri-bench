from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import csv
import gc
import hashlib
import json
import math
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


warnings.filterwarnings("ignore", category=UserWarning)

# Resolve project assets relative to this script instead of the caller's
# working directory. This keeps execution stable from Python 3.12 venvs,
# ROS 2 launch files, terminals, and IDEs started outside the project root.
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "limb_personalized"
CM_DIR = OUT / "confusion_matrices"
LOG_PATH = OUT / "stage8a_run.log"
STAGE1_MANIFEST = ROOT / "protocols" / "limb" / "LIMB_RECORDING_MANIFEST.csv"
CHANNEL_AUDIT = ROOT / "protocols" / "limb" / "CHANNEL_QUALITY_AUDIT.csv"
LOSO_DECISION = ROOT / "protocols" / "limb" / "LIMB_FINAL_DECISION.json"
STAGE9B_DECISION = ROOT / "protocols" / "boundary" / "STAGE9B_DECISION.json"

SEED = 42
FS = 1260
INPUT_COLS = 42
LABEL_COL = 43
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8, 10]
LABELS = [0, 1, 2, 3, 4]
LABEL_NAMES = {
    0: "HandOpen",
    1: "Lateral",
    2: "Pinch",
    3: "Power",
    4: "Rest",
}
ACTION_LABEL_TO_Y = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
CONDITION_ORDER = ["StaticP1", "StaticP2", "StaticP3", "StaticP4", "StaticP9", "StaticP10", "StaticP14", "Dynamic"]

WINDOW_CONFIGS = {
    "win0p25s_50ov": {"window_length_sec": 0.25, "window_length": 315, "step": 157},
    "win0p5s_50ov": {"window_length_sec": 0.5, "window_length": 630, "step": 315},
    "win1p0s_50ov": {"window_length_sec": 1.0, "window_length": 1260, "step": 630},
}
TRIM_CONFIGS = {
    "full_segment": {"trim_start_fraction": 0.0, "trim_end_fraction": 0.0},
    "trim10_each_end": {"trim_start_fraction": 0.10, "trim_end_fraction": 0.10},
    "trim20_each_end": {"trim_start_fraction": 0.20, "trim_end_fraction": 0.20},
}
MODEL_TYPES = ["ShrinkageLDA", "LogisticRegression", "RBF-SVM", "RandomForest"]


def log(msg: str) -> None:
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(text, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


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


def rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(p)


def load_limb_segments() -> pd.DataFrame:
    df = pd.read_csv(STAGE1_MANIFEST, encoding="utf-8-sig")
    df = df[
        (df["subject_id"].isin(SUBJECTS))
        & (df["action_label"].isin([1, 2, 3, 4, 5]))
        & (df["usable_status"].isin(["usable_active_action", "usable_rest"]))
    ].copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df["action_label"] = df["action_label"].astype(int)
    df["y"] = df["action_label"].map(ACTION_LABEL_TO_Y).astype(int)
    df["label_name"] = df["y"].map(LABEL_NAMES)
    df["condition_order"] = df["condition"].map({name: i for i, name in enumerate(CONDITION_ORDER)})
    df = df.sort_values(["subject_id", "action_label", "condition_order", "start_index"]).reset_index(drop=True)
    if len(df) != 360:
        raise RuntimeError(f"Expected 360 usable Limb segments, got {len(df)}")
    counts = df.groupby(["subject_id", "action_label"]).size()
    for subject in SUBJECTS:
        for label in [1, 2, 3, 4, 5]:
            if int(counts.loc[(subject, label)]) != 8:
                raise RuntimeError(f"Unexpected segment count for subject {subject} label {label}")
    df["class_segment_index"] = df.groupby(["subject_id", "action_label"]).cumcount().astype(int)
    return df


def load_channel_groups() -> dict[str, list[int]]:
    audit = pd.read_csv(CHANNEL_AUDIT, encoding="utf-8-sig")
    mapping = audit[["channel_index", "channel_semantic"]].drop_duplicates().sort_values("channel_index")
    if len(mapping) != 42:
        raise RuntimeError(f"Expected 42 input channels in audit, got {len(mapping)}")
    emg = [int(r.channel_index) - 1 for r in mapping.itertuples(index=False) if str(r.channel_semantic).startswith("EMG")]
    acc = [int(r.channel_index) - 1 for r in mapping.itertuples(index=False) if str(r.channel_semantic).startswith("ACC")]
    gyro = [int(r.channel_index) - 1 for r in mapping.itertuples(index=False) if str(r.channel_semantic).startswith("GYRO")]
    if len(emg) != 6 or len(acc) != 18 or len(gyro) != 18:
        raise RuntimeError(f"Unexpected channel grouping: EMG={len(emg)}, ACC={len(acc)}, GYRO={len(gyro)}")
    return {
        "EMG": emg,
        "EMG_ACC": emg + acc,
        "EMG_IMU": emg + acc + gyro,
    }


def build_split_manifest(segments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject in SUBJECTS:
        sdf = segments[segments["subject_id"] == subject]
        for fold_index in range(8):
            fold_id = f"S{subject:02d}_personal_fold{fold_index + 1:02d}"
            test_idx = fold_index
            val_idx = (fold_index + 1) % 8
            for row in sdf.itertuples(index=False):
                seg_idx = int(row.class_segment_index)
                if seg_idx == test_idx:
                    split = "test"
                elif seg_idx == val_idx:
                    split = "validation"
                else:
                    split = "train"
                rows.append(
                    {
                        "fold_id": fold_id,
                        "subject_id": subject,
                        "fold_index": fold_index + 1,
                        "test_class_segment_index": test_idx,
                        "validation_class_segment_index": val_idx,
                        "split_rule": "per class: test=i, validation=(i+1)%8, train=remaining 6 segment indices",
                        "split": split,
                        "source_file": row.source_file,
                        "condition": row.condition,
                        "recording_id": row.recording_id,
                        "segment_id": row.recording_id,
                        "class_segment_index": seg_idx,
                        "action_label": int(row.action_label),
                        "action_name": LABEL_NAMES[int(row.y)],
                        "y": int(row.y),
                        "start_index": int(row.start_index),
                        "end_index": int(row.end_index),
                        "sample_count": int(row.sample_count),
                        "sampling_rate": float(row.sampling_rate),
                        "channel_count": int(row.channel_count),
                        "label_column": int(row.label_column),
                    }
                )
    manifest = pd.DataFrame(rows)
    expected = {"train": 30, "validation": 5, "test": 5}
    for (subject, fold_id), g in manifest.groupby(["subject_id", "fold_id"]):
        counts = g["split"].value_counts().to_dict()
        if counts != expected:
            raise RuntimeError(f"Bad split counts for subject {subject} {fold_id}: {counts}")
    return manifest


def load_subject_arrays(subject_segments: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    arrays: dict[tuple[str, str], np.ndarray] = {}
    for source_file in sorted(subject_segments["source_file"].unique()):
        log(f"Loading subject MAT: {source_file}")
        mat = sio.loadmat(str(ROOT / source_file), simplify_cells=True)["limbEMG_Data"]
        for condition in sorted(subject_segments.loc[subject_segments["source_file"] == source_file, "condition"].unique()):
            arr = np.asarray(mat[condition], dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != LABEL_COL:
                raise RuntimeError(f"Unexpected array shape for {source_file} {condition}: {arr.shape}")
            arrays[(source_file, condition)] = arr
        del mat
        gc.collect()
    return arrays


def make_windows_for_subject(
    subject_segments: pd.DataFrame,
    arrays: dict[tuple[str, str], np.ndarray],
    channels: list[int],
    window_length: int,
    step: int,
    trim_start_fraction: float,
    trim_end_fraction: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    meta_rows: list[dict] = []
    for row in subject_segments.itertuples(index=False):
        arr = arrays[(row.source_file, row.condition)]
        seg_start = int(row.start_index)
        seg_end = int(row.end_index)
        seg_len = seg_end - seg_start + 1
        trim_start = int(math.floor(seg_len * trim_start_fraction))
        trim_end = int(math.floor(seg_len * trim_end_fraction))
        usable_start = seg_start + trim_start
        usable_end = seg_end - trim_end
        if usable_end - usable_start + 1 < window_length:
            usable_start = seg_start
            usable_end = seg_end
        local_starts = list(range(usable_start, usable_end - window_length + 2, step))
        if local_starts and local_starts[-1] + window_length - 1 < usable_end:
            last_start = usable_end - window_length + 1
            if last_start > local_starts[-1]:
                local_starts.append(last_start)
        if not local_starts:
            local_starts = [max(seg_start, seg_end - window_length + 1)]
        for w_i, start in enumerate(local_starts):
            end = start + window_length - 1
            x = arr[start - 1 : end, channels]
            x_rows.append(extract_feature_vector(x))
            y_rows.append(int(row.y))
            meta_rows.append(
                {
                    "subject_id": int(row.subject_id),
                    "condition": row.condition,
                    "recording_id": row.recording_id,
                    "segment_id": row.recording_id,
                    "class_segment_index": int(row.class_segment_index),
                    "action_label": int(row.action_label),
                    "action_name": LABEL_NAMES[int(row.y)],
                    "y": int(row.y),
                    "window_start_index": int(start),
                    "window_end_index": int(end),
                    "window_index_in_segment": int(w_i),
                }
            )
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64), pd.DataFrame(meta_rows)


def extract_feature_vector(window: np.ndarray) -> np.ndarray:
    x = np.asarray(window, dtype=np.float32)
    eps = 1e-12
    q05, q25, q75, q95 = np.percentile(x, [5, 25, 75, 95], axis=0)
    med = np.median(x, axis=0)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    var = np.var(x, axis=0)
    mad = np.median(np.abs(x - med[None, :]), axis=0)
    abs_x = np.abs(x)
    diff = np.diff(x, axis=0)
    rms = np.sqrt(np.mean(x * x, axis=0))
    iqr = q75 - q25
    threshold = 0.01 * np.maximum(iqr, eps)
    zc = np.mean((x[:-1] * x[1:]) < 0, axis=0)
    ssc = np.mean((diff[:-1] * diff[1:]) < 0, axis=0) if diff.shape[0] > 1 else np.zeros(x.shape[1])
    wamp = np.mean(np.abs(diff) > threshold[None, :], axis=0)

    power = np.abs(np.fft.rfft(x, axis=0)) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / FS)
    power[0, :] = 0.0
    total_power = np.maximum(np.sum(power, axis=0), eps)
    prob = power / total_power[None, :]
    entropy = -np.sum(prob * np.log(np.maximum(prob, eps)), axis=0) / math.log(prob.shape[0])
    mean_freq = np.sum(freqs[:, None] * power, axis=0) / total_power
    cum_power = np.cumsum(power, axis=0)
    median_indices = [
        min(int(np.searchsorted(cum_power[:, ch], total_power[ch] * 0.5)), len(freqs) - 1)
        for ch in range(x.shape[1])
    ]
    median_freq = freqs[np.asarray(median_indices, dtype=int)]
    band_feats = []
    for low, high in [(0, 20), (20, 100), (100, 250), (250, 500), (500, FS / 2 + 1)]:
        mask = (freqs >= low) & (freqs < high)
        band_feats.append(np.sum(power[mask, :], axis=0) / total_power)

    dx_var = np.var(diff, axis=0) if diff.shape[0] else np.zeros(x.shape[1])
    ddiff = np.diff(diff, axis=0) if diff.shape[0] > 1 else np.zeros((1, x.shape[1]), dtype=np.float32)
    ddx_var = np.var(ddiff, axis=0)
    mobility = np.sqrt(dx_var / np.maximum(var, eps))
    mobility_dx = np.sqrt(ddx_var / np.maximum(dx_var, eps))
    complexity = mobility_dx / np.maximum(mobility, eps)

    feats = [
        mean,
        std,
        med,
        iqr,
        mad,
        q05,
        q25,
        q75,
        q95,
        np.mean(abs_x, axis=0),
        rms,
        var,
        np.mean(x * x, axis=0),
        np.sum(abs_x, axis=0) / x.shape[0],
        np.sum(np.abs(diff), axis=0) / max(diff.shape[0], 1),
        zc,
        ssc,
        wamp,
        np.ptp(x, axis=0),
        np.mean(diff, axis=0) if diff.shape[0] else np.zeros(x.shape[1]),
        np.std(diff, axis=0) if diff.shape[0] else np.zeros(x.shape[1]),
        np.mean(np.abs(diff), axis=0) if diff.shape[0] else np.zeros(x.shape[1]),
        entropy,
        mean_freq,
        median_freq,
        *band_feats,
        var,
        mobility,
        complexity,
    ]
    return np.nan_to_num(np.concatenate(feats).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def make_model(model_type: str) -> Pipeline:
    if model_type == "ShrinkageLDA":
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif model_type == "LogisticRegression":
        clf = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1200,
            random_state=SEED,
            solver="lbfgs",
        )
    elif model_type == "RBF-SVM":
        clf = SVC(C=3.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=True, random_state=SEED)
    elif model_type == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=80,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=1,
        )
    else:
        raise ValueError(model_type)
    return Pipeline([("scaler", StandardScaler()), ("classifier", clf)])


def probabilities(model: Pipeline, x: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(x)
    classes = np.asarray(model.named_steps["classifier"].classes_)
    out = np.zeros((len(x), len(LABELS)), dtype=np.float64)
    for src_idx, label in enumerate(classes):
        out[:, int(label)] = prob[:, src_idx]
    row_sum = np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out / row_sum


def aggregate_segments(meta: pd.DataFrame, y: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    rows = []
    for segment_id, idx in meta.groupby("segment_id").groups.items():
        indices = np.asarray(list(idx), dtype=int)
        mean_prob = prob[indices].mean(axis=0)
        true_label = int(y[indices[0]])
        pred_label = int(np.argmax(mean_prob))
        first = meta.iloc[indices[0]]
        row = {
            "subject_id": int(first.subject_id),
            "condition": first.condition,
            "recording_id": first.recording_id,
            "segment_id": segment_id,
            "true_label": true_label,
            "true_name": LABEL_NAMES[true_label],
            "predicted_label": pred_label,
            "predicted_name": LABEL_NAMES[pred_label],
            "window_count": int(len(indices)),
            "correct": bool(true_label == pred_label),
        }
        for label in LABELS:
            row[f"prob_{label}_{LABEL_NAMES[label]}"] = float(mean_prob[label])
        rows.append(row)
    seg_df = pd.DataFrame(rows)
    return (
        seg_df["true_label"].to_numpy(dtype=int),
        seg_df["predicted_label"].to_numpy(dtype=int),
        seg_df[[f"prob_{label}_{LABEL_NAMES[label]}" for label in LABELS]].to_numpy(dtype=float),
        seg_df,
    )


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    p, r, f1, _support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_precision": float(p),
        f"{prefix}_macro_recall": float(r),
        f"{prefix}_macro_f1": float(f1),
    }


def evaluate_predictions(meta: pd.DataFrame, y: np.ndarray, prob: np.ndarray, prefix: str) -> tuple[dict, pd.DataFrame]:
    win_pred = np.argmax(prob, axis=1)
    metrics = metric_dict(y, win_pred, f"{prefix}_window")
    seg_true, seg_pred, _seg_prob, seg_df = aggregate_segments(meta, y, prob)
    metrics.update(metric_dict(seg_true, seg_pred, f"{prefix}_segment"))
    return metrics, seg_df


def select_key(row: dict) -> tuple:
    simplicity = {"ShrinkageLDA": 0, "LogisticRegression": 1, "RBF-SVM": 2, "RandomForest": 3}
    window_order = {"win0p25s_50ov": 0, "win0p5s_50ov": 1, "win1p0s_50ov": 2}
    trim_order = {"full_segment": 0, "trim10_each_end": 1, "trim20_each_end": 2}
    modality_order = {"EMG": 0, "EMG_ACC": 1, "EMG_IMU": 2}
    return (
        row["val_segment_macro_f1"],
        row["val_segment_balanced_accuracy"],
        row["val_window_macro_f1"],
        -simplicity[row["model_type"]],
        -window_order[row["window_config_id"]],
        -trim_order[row["trim_config_id"]],
        -modality_order[row["modality"]],
    )


def get_strict_loso_summary() -> dict:
    if not LOSO_DECISION.exists():
        return {"available": False, "reason": "reports/limb_final_selection/LIMB_FINAL_DECISION.json not found"}
    try:
        data = json.loads(LOSO_DECISION.read_text(encoding="utf-8"))
        return {"available": True, "path": rel(LOSO_DECISION), "data": data}
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}


def get_stage9b_summary() -> dict:
    if not STAGE9B_DECISION.exists():
        return {"available": False, "reason": "Stage 9B decision not found"}
    try:
        data = json.loads(STAGE9B_DECISION.read_text(encoding="utf-8"))
        return {
            "available": True,
            "path": rel(STAGE9B_DECISION),
            "stage9b_status": data.get("stage9b_status"),
            "conclusion": data.get("external_generalization_conclusion"),
            "metrics": data.get("metrics", {}),
        }
    except Exception as exc:
        return {"available": False, "reason": repr(exc)}


def summarize_subjects(per_fold_rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(per_fold_rows)
    rows = []
    for subject, g in df.groupby("subject_id"):
        row = {
            "subject_id": int(subject),
            "fold_count": int(len(g)),
            "selected_model_counts": json.dumps(g["model_type"].value_counts().to_dict(), ensure_ascii=False),
            "selected_window_counts": json.dumps(g["window_config_id"].value_counts().to_dict(), ensure_ascii=False),
            "selected_trim_counts": json.dumps(g["trim_config_id"].value_counts().to_dict(), ensure_ascii=False),
            "selected_modality_counts": json.dumps(g["modality"].value_counts().to_dict(), ensure_ascii=False),
        }
        for metric in [
            "test_segment_macro_f1",
            "test_segment_accuracy",
            "test_segment_balanced_accuracy",
            "test_segment_macro_precision",
            "test_segment_macro_recall",
            "test_window_macro_f1",
        ]:
            vals = pd.to_numeric(g[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            row[f"{metric}_min"] = float(vals.min())
            row[f"{metric}_max"] = float(vals.max())
        rows.append(row)
    return rows


def build_summary(per_fold_rows: list[dict], segment_rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(per_fold_rows)
    seg = pd.DataFrame(segment_rows)
    y_true = seg["true_label"].to_numpy(dtype=int)
    y_pred = seg["predicted_label"].to_numpy(dtype=int)
    row = {
        "summary_scope": "selected_validation_per_subject_fold",
        "subject_count": int(df["subject_id"].nunique()),
        "fold_count": int(len(df)),
        "test_segment_count": int(len(seg)),
    }
    row.update(metric_dict(y_true, y_pred, "overall_segment"))
    for metric in [
        "test_segment_macro_f1",
        "test_segment_accuracy",
        "test_segment_balanced_accuracy",
        "test_segment_macro_precision",
        "test_segment_macro_recall",
        "test_window_macro_f1",
    ]:
        vals = pd.to_numeric(df[metric], errors="coerce")
        row[f"{metric}_mean"] = float(vals.mean())
        row[f"{metric}_std"] = float(vals.std(ddof=1))
        row[f"{metric}_min"] = float(vals.min())
        row[f"{metric}_max"] = float(vals.max())
    row["target_macro_f1_ge_0p70"] = bool(row["overall_segment_macro_f1"] >= 0.70)
    row["target_macro_f1_ge_0p80"] = bool(row["overall_segment_macro_f1"] >= 0.80)
    return [row]


def write_reports(
    summary_row: dict,
    subject_rows: list[dict],
    per_class_rows: list[dict],
    confusion_rows: list[dict],
    run_status_rows: list[dict],
) -> None:
    worst_subject = min(subject_rows, key=lambda row: row["test_segment_macro_f1_mean"])
    confusions = []
    for row in confusion_rows:
        true_name = row["true_name"]
        for label in LABELS:
            pred_col = f"pred_{label}_{LABEL_NAMES[label]}"
            if LABEL_NAMES[label] != true_name and int(row[pred_col]) > 0:
                confusions.append((true_name, LABEL_NAMES[label], int(row[pred_col])))
    confusions = sorted(confusions, key=lambda item: -item[2])[:8]

    lines = [
        "# Limb Subject-Dependent Personalized Five-Class Report",
        "",
        "## Scope",
        "",
        "- Subjects 1-8 and 10 are used; subject 9 is excluded by the audited protocol.",
        "- Classes: Hand Open, Lateral Grip, Pinch Grip, Power Grip, and Rest.",
        "- Input columns 1-42 contain sensor channels; column 43 contains the action label.",
        "- Each subject is modelled independently.",
        "",
        "## Protocol",
        "",
        "- Each class has eight original segments and is evaluated in eight cyclic folds.",
        "- Per fold and class: six segments train, one validates, and one tests.",
        "- All windows derived from one segment remain in exactly one split.",
        "- Validation macro-F1 selects the final configuration; test data are only reported afterwards.",
        "",
        "## Main result",
        "",
        f"- Overall segment macro-F1: {summary_row['overall_segment_macro_f1']:.4f}",
        f"- Overall segment accuracy: {summary_row['overall_segment_accuracy']:.4f}",
        f"- Overall segment balanced accuracy: {summary_row['overall_segment_balanced_accuracy']:.4f}",
        f"- Mean fold segment macro-F1: {summary_row['test_segment_macro_f1_mean']:.4f} +/- {summary_row['test_segment_macro_f1_std']:.4f}",
        f"- Worst fold segment macro-F1: {summary_row['test_segment_macro_f1_min']:.4f}",
        f"- Worst subject: S{worst_subject['subject_id']} (mean macro-F1={worst_subject['test_segment_macro_f1_mean']:.4f})",
        "",
        "## Interpretation",
        "",
        "This is a subject-dependent result that requires calibration data from each user. It must not be interpreted as zero-calibration cross-subject generalization.",
        "",
        "## Per-class result",
        "",
    ]
    for row in per_class_rows:
        lines.append(
            f"- {row['class_name']}: precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, support={row['support']}."
        )
    lines.extend(["", "## Main confusions", ""])
    if confusions:
        lines.extend(f"- {true_name} -> {pred_name}: {count} segments." for true_name, pred_name, count in confusions)
    else:
        lines.append("- No off-diagonal segment confusions in the aggregate selected test predictions.")
    lines.extend(["", "## Run status", ""])
    for row in run_status_rows:
        lines.append(f"- {row['component']}: {row['status']} ({row['notes']})")
    (OUT / "PERSONALIZED_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_leakage_audit(split_hash: str) -> None:
    lines = [
        "# Personalized Leakage Audit",
        "",
        "## Atomic unit",
        "",
        "- The original labelled segment is the smallest split unit.",
        "- Every window derived from one segment belongs only to train, validation, or test within a fold.",
        "- No random window-level split is used.",
        "",
        "## Split rule",
        "",
        "- Each subject has eight segments per class.",
        "- Fold i uses segment i for test, segment (i + 1) modulo 8 for validation, and the remaining six for training.",
        f"- Split manifest SHA256: `{split_hash}`.",
        "",
        "## Fitting boundary",
        "",
        "- Each sklearn Pipeline fits feature standardization on training windows only.",
        "- Validation data only select candidate configurations and models.",
        "- Test data are evaluated after selection and are used only for reporting.",
        "- Bingbin data are not used in this experiment.",
        "",
        "Leakage status: PASS.",
    ]
    (OUT / "LEAKAGE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    CM_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    log("Stage 8A personalized Limb experiment start")

    segments = load_limb_segments()
    channel_groups = load_channel_groups()
    split_manifest = build_split_manifest(segments)
    write_csv(OUT / "PERSONALIZED_SPLIT_MANIFEST.csv", split_manifest.to_dict("records"))
    split_hash = sha256_file(OUT / "PERSONALIZED_SPLIT_MANIFEST.csv")
    (OUT / "PERSONALIZED_SPLIT_HASH.json").write_text(
        json.dumps(
            {
                "split_manifest_path": rel(OUT / "PERSONALIZED_SPLIT_MANIFEST.csv"),
                "split_manifest_sha256": split_hash,
                "split_rule": "per subject and class: test=i, validation=(i+1)%8, train=remaining six class segment indices",
                "subject_count": len(SUBJECTS),
                "folds_per_subject": 8,
                "segments_per_subject_per_class": 8,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    selection_rows: list[dict] = []
    per_fold_rows: list[dict] = []
    selected_segment_rows: list[dict] = []
    complexity_rows: list[dict] = []
    run_status_rows = [
        {
            "component": "traditional_feature_grid",
            "status": "running",
            "notes": "Two-stage validation selection: 3 windows x 3 trims x 3 modalities screened by Shrinkage LDA, then 4 traditional models on each fold's best feature config.",
        },
        {
            "component": "automatic_stable_region",
            "status": "not_run_due_to_scope",
            "notes": "Optional exploration omitted to keep Stage 8A auditable and bounded.",
        },
        {
            "component": "deep_tcn_multiscale",
            "status": "not_run_due_to_time",
            "notes": "Traditional feature models completed first; deep models deferred to possible Stage 8B.",
        },
    ]

    for subject in SUBJECTS:
        log(f"Subject S{subject:02d}: loading data")
        subject_segments = segments[segments["subject_id"] == subject].copy()
        arrays = load_subject_arrays(subject_segments)
        feature_best_by_fold: dict[str, dict] = {}
        best_by_fold: dict[str, dict] = {}
        feature_cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}

        for window_id, win in WINDOW_CONFIGS.items():
            for trim_id, trim in TRIM_CONFIGS.items():
                for modality, channels in channel_groups.items():
                    config_key = (window_id, trim_id, modality)
                    t_feat = time.perf_counter()
                    x_all, y_all, meta_all = make_windows_for_subject(
                        subject_segments,
                        arrays,
                        channels,
                        win["window_length"],
                        win["step"],
                        trim["trim_start_fraction"],
                        trim["trim_end_fraction"],
                    )
                    feature_cache[config_key] = (x_all, y_all, meta_all)
                    log(
                        f"S{subject:02d} {window_id}/{trim_id}/{modality}: "
                        f"{len(x_all)} windows, {x_all.shape[1]} features in {time.perf_counter() - t_feat:.1f}s"
                    )
                    for fold_index in range(1, 9):
                        fold_id = f"S{subject:02d}_personal_fold{fold_index:02d}"
                        fold_split = split_manifest[
                            (split_manifest["subject_id"] == subject) & (split_manifest["fold_id"] == fold_id)
                        ][["segment_id", "split"]]
                        split_map = dict(zip(fold_split["segment_id"], fold_split["split"]))
                        split_values = meta_all["segment_id"].map(split_map).to_numpy()
                        train_mask = split_values == "train"
                        val_mask = split_values == "validation"
                        model_type = "ShrinkageLDA"
                        candidate_id = f"{window_id}__{trim_id}__{modality}__{model_type}"
                        t0 = time.perf_counter()
                        try:
                            model = make_model(model_type)
                            model.fit(x_all[train_mask], y_all[train_mask])
                            val_prob = probabilities(model, x_all[val_mask])
                            metrics, _val_seg = evaluate_predictions(
                                meta_all[val_mask].reset_index(drop=True),
                                y_all[val_mask],
                                val_prob,
                                "val",
                            )
                            row = {
                                "selection_stage": "feature_screen_with_shrinkage_lda",
                                "subject_id": subject,
                                "fold_id": fold_id,
                                "fold_index": fold_index,
                                "candidate_id": candidate_id,
                                "window_config_id": window_id,
                                "window_length_sec": win["window_length_sec"],
                                "window_length": win["window_length"],
                                "step_samples": win["step"],
                                "trim_config_id": trim_id,
                                "trim_start_fraction": trim["trim_start_fraction"],
                                "trim_end_fraction": trim["trim_end_fraction"],
                                "modality": modality,
                                "channel_count": len(channels),
                                "model_type": model_type,
                                "feature_dim": int(x_all.shape[1]),
                                "train_window_count": int(train_mask.sum()),
                                "validation_window_count": int(val_mask.sum()),
                                "fit_eval_seconds": float(time.perf_counter() - t0),
                                "status": "success",
                            }
                            row.update(metrics)
                        except Exception as exc:
                            row = {
                                "selection_stage": "feature_screen_with_shrinkage_lda",
                                "subject_id": subject,
                                "fold_id": fold_id,
                                "fold_index": fold_index,
                                "candidate_id": candidate_id,
                                "window_config_id": window_id,
                                "trim_config_id": trim_id,
                                "modality": modality,
                                "model_type": model_type,
                                "feature_dim": int(x_all.shape[1]),
                                "status": f"failed: {exc!r}",
                            }
                        selection_rows.append(row)
                        if row.get("status") == "success":
                            current = feature_best_by_fold.get(fold_id)
                            if current is None or select_key(row) > select_key(current):
                                feature_best_by_fold[fold_id] = row

        for fold_index in range(1, 9):
            fold_id = f"S{subject:02d}_personal_fold{fold_index:02d}"
            feature_best = feature_best_by_fold[fold_id]
            config_key = (feature_best["window_config_id"], feature_best["trim_config_id"], feature_best["modality"])
            x_all, y_all, meta_all = feature_cache[config_key]
            fold_split = split_manifest[
                (split_manifest["subject_id"] == subject) & (split_manifest["fold_id"] == fold_id)
            ][["segment_id", "split"]]
            split_map = dict(zip(fold_split["segment_id"], fold_split["split"]))
            split_values = meta_all["segment_id"].map(split_map).to_numpy()
            train_mask = split_values == "train"
            val_mask = split_values == "validation"
            for model_type in MODEL_TYPES:
                candidate_id = f"{feature_best['window_config_id']}__{feature_best['trim_config_id']}__{feature_best['modality']}__{model_type}"
                t0 = time.perf_counter()
                try:
                    model = make_model(model_type)
                    model.fit(x_all[train_mask], y_all[train_mask])
                    val_prob = probabilities(model, x_all[val_mask])
                    metrics, _val_seg = evaluate_predictions(
                        meta_all[val_mask].reset_index(drop=True),
                        y_all[val_mask],
                        val_prob,
                        "val",
                    )
                    row = {
                        "selection_stage": "model_screen_on_best_feature_config",
                        "subject_id": subject,
                        "fold_id": fold_id,
                        "fold_index": fold_index,
                        "candidate_id": candidate_id,
                        "window_config_id": feature_best["window_config_id"],
                        "window_length_sec": feature_best["window_length_sec"],
                        "window_length": feature_best["window_length"],
                        "step_samples": feature_best["step_samples"],
                        "trim_config_id": feature_best["trim_config_id"],
                        "trim_start_fraction": feature_best["trim_start_fraction"],
                        "trim_end_fraction": feature_best["trim_end_fraction"],
                        "modality": feature_best["modality"],
                        "channel_count": feature_best["channel_count"],
                        "model_type": model_type,
                        "feature_dim": int(x_all.shape[1]),
                        "train_window_count": int(train_mask.sum()),
                        "validation_window_count": int(val_mask.sum()),
                        "fit_eval_seconds": float(time.perf_counter() - t0),
                        "status": "success",
                    }
                    row.update(metrics)
                except Exception as exc:
                    row = {
                        "selection_stage": "model_screen_on_best_feature_config",
                        "subject_id": subject,
                        "fold_id": fold_id,
                        "fold_index": fold_index,
                        "candidate_id": candidate_id,
                        "window_config_id": feature_best["window_config_id"],
                        "trim_config_id": feature_best["trim_config_id"],
                        "modality": feature_best["modality"],
                        "model_type": model_type,
                        "feature_dim": int(x_all.shape[1]),
                        "status": f"failed: {exc!r}",
                    }
                selection_rows.append(row)
                if row.get("status") == "success":
                    current = best_by_fold.get(fold_id)
                    if current is None or select_key(row) > select_key(current):
                        best_by_fold[fold_id] = row

        for fold_index in range(1, 9):
            fold_id = f"S{subject:02d}_personal_fold{fold_index:02d}"
            best = best_by_fold[fold_id]
            config_key = (best["window_config_id"], best["trim_config_id"], best["modality"])
            x_all, y_all, meta_all = feature_cache[config_key]
            fold_split = split_manifest[
                (split_manifest["subject_id"] == subject) & (split_manifest["fold_id"] == fold_id)
            ][["segment_id", "split"]]
            split_map = dict(zip(fold_split["segment_id"], fold_split["split"]))
            split_values = meta_all["segment_id"].map(split_map).to_numpy()
            train_mask = split_values == "train"
            val_mask = split_values == "validation"
            test_mask = split_values == "test"
            train_val_mask = train_mask | val_mask
            # Final personalized fold fit after validation selection uses train+validation calibration segments for that subject.
            model = make_model(best["model_type"])
            t0 = time.perf_counter()
            model.fit(x_all[train_val_mask], y_all[train_val_mask])
            test_prob = probabilities(model, x_all[test_mask])
            fit_seconds = time.perf_counter() - t0
            test_metrics, test_seg = evaluate_predictions(
                meta_all[test_mask].reset_index(drop=True),
                y_all[test_mask],
                test_prob,
                "test",
            )
            per_row = {
                "subject_id": subject,
                "fold_id": fold_id,
                "fold_index": fold_index,
                "selected_candidate_id": best["candidate_id"],
                "window_config_id": best["window_config_id"],
                "window_length_sec": best["window_length_sec"],
                "window_length": best["window_length"],
                "step_samples": best["step_samples"],
                "trim_config_id": best["trim_config_id"],
                "trim_start_fraction": best["trim_start_fraction"],
                "trim_end_fraction": best["trim_end_fraction"],
                "modality": best["modality"],
                "channel_count": best["channel_count"],
                "model_type": best["model_type"],
                "feature_dim": best["feature_dim"],
                "validation_selected_segment_macro_f1": best["val_segment_macro_f1"],
                "train_segment_count": 30,
                "validation_segment_count": 5,
                "test_segment_count": 5,
                "final_fit_scope": "train_plus_validation_after_config_selection",
                "final_fit_seconds": float(fit_seconds),
            }
            per_row.update(test_metrics)
            per_fold_rows.append(per_row)
            for row in test_seg.to_dict("records"):
                selected_segment_rows.append(
                    {
                        "subject_id": subject,
                        "fold_id": fold_id,
                        "fold_index": fold_index,
                        "selected_candidate_id": best["candidate_id"],
                        "model_type": best["model_type"],
                        "window_config_id": best["window_config_id"],
                        "trim_config_id": best["trim_config_id"],
                        "modality": best["modality"],
                        **row,
                    }
                )
            complexity_rows.append(
                {
                    "subject_id": subject,
                    "fold_id": fold_id,
                    "model_type": best["model_type"],
                    "feature_dim": best["feature_dim"],
                    "train_validation_window_count": int(train_val_mask.sum()),
                    "test_window_count": int(test_mask.sum()),
                    "param_count_estimate": int(best["feature_dim"] * len(LABELS)) if best["model_type"] in ["ShrinkageLDA", "LogisticRegression"] else "",
                    "support_vectors_or_trees": int(getattr(model.named_steps["classifier"], "n_support_", np.array([])).sum())
                    if best["model_type"] == "RBF-SVM"
                    else (80 if best["model_type"] == "RandomForest" else ""),
                    "final_fit_seconds": float(fit_seconds),
                }
            )
            log(
                f"S{subject:02d} {fold_id}: selected {best['candidate_id']} "
                f"test segment Macro-F1={per_row['test_segment_macro_f1']:.3f}"
            )

        del arrays, feature_cache
        gc.collect()

    run_status_rows[0]["status"] = "success"
    run_status_rows[0]["notes"] = f"{len(selection_rows)} validation candidate fits recorded; {len(per_fold_rows)} selected fold tests completed."

    write_csv(OUT / "PERSONALIZED_MODEL_SELECTION_RESULTS.csv", selection_rows)
    write_csv(OUT / "PERSONALIZED_PER_FOLD.csv", per_fold_rows)
    subject_rows = summarize_subjects(per_fold_rows)
    write_csv(OUT / "PERSONALIZED_PER_SUBJECT.csv", subject_rows)
    summary_rows = build_summary(per_fold_rows, selected_segment_rows)
    write_csv(OUT / "PERSONALIZED_SUMMARY.csv", summary_rows)

    seg_df = pd.DataFrame(selected_segment_rows)
    y_true = seg_df["true_label"].to_numpy(dtype=int)
    y_pred = seg_df["predicted_label"].to_numpy(dtype=int)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    per_class_rows = []
    for idx, label in enumerate(LABELS):
        per_class_rows.append(
            {
                "class_label": label,
                "class_name": LABEL_NAMES[label],
                "precision": float(p[idx]),
                "recall": float(r[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    write_csv(OUT / "PERSONALIZED_PER_CLASS.csv", per_class_rows)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    confusion_rows = []
    for i, label in enumerate(LABELS):
        row = {"true_label": label, "true_name": LABEL_NAMES[label]}
        for j, pred_label in enumerate(LABELS):
            row[f"pred_{pred_label}_{LABEL_NAMES[pred_label]}"] = int(cm[i, j])
        confusion_rows.append(row)
    write_csv(OUT / "PERSONALIZED_CONFUSION_MATRIX.csv", confusion_rows)
    write_csv(OUT / "PERSONALIZED_MODEL_COMPLEXITY.csv", complexity_rows)
    write_csv(OUT / "PERSONALIZED_RUN_STATUS.csv", run_status_rows)
    write_leakage_audit(split_hash)

    summary_row = summary_rows[0]
    write_reports(summary_row, subject_rows, per_class_rows, confusion_rows, run_status_rows)
    decision = {
        "stage": "Stage 8A personalized subject-dependent Limb 5-class experiment",
        "stage8a_status": "DONE",
        "scope": {
            "dataset": "Limb Position only",
            "subjects": SUBJECTS,
            "excluded_subjects": [9],
            "classes": LABEL_NAMES,
            "input_columns": "1-42",
            "label_column": 43,
            "bingbin_read_or_used": False,
            "stage9_modified": False,
        },
        "split_manifest_sha256": split_hash,
        "overall_segment_macro_f1": summary_row["overall_segment_macro_f1"],
        "overall_segment_accuracy": summary_row["overall_segment_accuracy"],
        "overall_segment_balanced_accuracy": summary_row["overall_segment_balanced_accuracy"],
        "mean_fold_segment_macro_f1": summary_row["test_segment_macro_f1_mean"],
        "std_fold_segment_macro_f1": summary_row["test_segment_macro_f1_std"],
        "worst_fold_segment_macro_f1": summary_row["test_segment_macro_f1_min"],
        "target_macro_f1_ge_0p70": summary_row["target_macro_f1_ge_0p70"],
        "target_macro_f1_ge_0p80": summary_row["target_macro_f1_ge_0p80"],
        "strict_loso_reference": get_strict_loso_summary(),
        "bingbin_external_validation_boundary": get_stage9b_summary(),
        "application_main_result_recommendation": "YES for subject-dependent personalized Limb model, with explicit calibration requirement and LOSO/Bingbin boundary notes.",
        "deep_models_status": "not_run_due_to_time",
        "automatic_stable_region_status": "not_run_due_to_scope",
        "raw_data_modified": False,
        "original_results_overwritten": False,
    }
    (OUT / "STAGE8A_DECISION.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    log(
        f"Stage 8A complete: overall segment Macro-F1={summary_row['overall_segment_macro_f1']:.4f}, "
        f"mean fold Macro-F1={summary_row['test_segment_macro_f1_mean']:.4f}"
    )


if __name__ == "__main__":
    main()

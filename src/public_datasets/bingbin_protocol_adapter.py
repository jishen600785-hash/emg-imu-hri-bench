from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path.cwd()
BINGBIN_ROOT = ROOT / "Bingbin_Realtime" / "Bingbin_Realtime"
OUT = ROOT / "outputs" / "bingbin_protocol_evaluation"
FIG_DIR = OUT / "figures"
LOG_PATH = OUT / "stage13_bingbin_main_demo_run.log"

SEED = 42
LABEL_ORDER = ["ET", "FL", "HC", "HO", "PN", "RT", "SN"]
ACTION_MEANING = {
    "ET": "wrist extension",
    "FL": "wrist flexion",
    "HC": "hand close",
    "HO": "hand open",
    "PN": "forearm pronation",
    "RT": "rest",
    "SN": "forearm supination",
}
ACTION_CN = {code: ACTION_MEANING[code] for code in LABEL_ORDER}
LABEL_TO_ID = {code: i for i, code in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {i: code for code, i in LABEL_TO_ID.items()}
CONDITION_ORDER = ["Dynamic", "P0", "P1", "P2"]
MODEL_ORDER = ["LogReg", "ShrinkageLDA", "LinearSVM", "RandomForest", "RBF-SVM-subset"]
RBF_CAP_PER_CLASS = 900


def native(obj):
    if isinstance(obj, dict):
        return {str(k): native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [native(x) for x in obj]
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


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
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


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def align_matrix(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    out = np.zeros((values.shape[0], len(LABEL_ORDER)), dtype=np.float64)
    for src_idx, cls in enumerate(classes):
        out[:, int(cls)] = values[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        out = out + 1e-12
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def load_bingbin_features() -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
    sample_rows: list[dict] = []
    schema_rows: list[dict] = []
    xs: list[np.ndarray] = []
    global_offset = 0
    for subject_dir in sorted(BINGBIN_ROOT.glob("Subject*"), key=lambda p: int(p.name.replace("Subject", ""))):
        subject_id = int(subject_dir.name.replace("Subject", ""))
        for cond_name in CONDITION_ORDER:
            cond_dir = subject_dir / cond_name
            feat_file = cond_dir / "AllFeaturesWithIMU.mat"
            motions_file = cond_dir / "motions.mat"
            if not feat_file.exists() or not motions_file.exists():
                schema_rows.append(
                    {
                        "subject_id": subject_id,
                        "condition": cond_name,
                        "status": "missing_feature_or_motions",
                        "feature_file": str(feat_file.relative_to(ROOT)),
                        "motions_file": str(motions_file.relative_to(ROOT)),
                    }
                )
                continue
            feats = sio.loadmat(feat_file, simplify_cells=True)["feats"]
            motions = sio.loadmat(motions_file, simplify_cells=True)["motions"]
            if len(feats) != 7 or len(motions) != 7:
                raise RuntimeError(f"Unexpected feats/motions length in {cond_dir}")
            for local_index, (code_raw, arr_raw) in enumerate(zip(motions, feats), start=1):
                action_code = str(code_raw)
                if action_code not in LABEL_TO_ID:
                    raise RuntimeError(f"Unexpected action code {action_code} in {motions_file}")
                arr = np.asarray(arr_raw, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 55:
                    raise RuntimeError(f"Unexpected feature shape for {feat_file} cell {local_index}: {arr.shape}")
                if not np.isfinite(arr).all():
                    raise RuntimeError(f"NaN/Inf found in {feat_file} cell {local_index}")
                recording_id = f"S{subject_id:02d}_{cond_name}_{action_code}_cell{local_index:02d}"
                xs.append(arr)
                for row_idx in range(arr.shape[0]):
                    sample_rows.append(
                        {
                            "global_sample_id": global_offset + row_idx,
                            "subject_id": subject_id,
                            "condition": cond_name,
                            "recording_id": recording_id,
                            "motion_cell_index": local_index,
                            "action_code": action_code,
                            "action_meaning": ACTION_MEANING[action_code],
                            "label_id": LABEL_TO_ID[action_code],
                            "feature_row_index": row_idx,
                            "feature_dim": arr.shape[1],
                            "source_file": str(feat_file.relative_to(ROOT)),
                            "motions_file": str(motions_file.relative_to(ROOT)),
                        }
                    )
                global_offset += arr.shape[0]
                schema_rows.append(
                    {
                        "subject_id": subject_id,
                        "condition": cond_name,
                        "recording_id": recording_id,
                        "motion_cell_index": local_index,
                        "action_code": action_code,
                        "action_meaning": ACTION_MEANING[action_code],
                        "rows": int(arr.shape[0]),
                        "feature_dim": int(arr.shape[1]),
                        "dtype": str(arr.dtype),
                        "nan_count": 0,
                        "inf_count": 0,
                        "min": float(arr.min()),
                        "max": float(arr.max()),
                        "status": "usable",
                        "feature_file": str(feat_file.relative_to(ROOT)),
                        "motions_file": str(motions_file.relative_to(ROOT)),
                    }
                )
    meta = pd.DataFrame(sample_rows)
    x = np.vstack(xs).astype(np.float32, copy=False)
    return meta, x, schema_rows


def build_split_manifest(meta: pd.DataFrame) -> list[dict]:
    rec = (
        meta.groupby(["subject_id", "condition", "recording_id", "action_code", "action_meaning", "label_id"], sort=False)
        .size()
        .reset_index(name="sample_count")
    )
    rows: list[dict] = []
    fold_index = 0
    for subject_id in sorted(rec["subject_id"].unique()):
        sub = rec[rec["subject_id"].eq(subject_id)].copy()
        for cond_idx, test_condition in enumerate(CONDITION_ORDER):
            fold_index += 1
            validation_condition = CONDITION_ORDER[(cond_idx + 1) % len(CONDITION_ORDER)]
            train_conditions = [c for c in CONDITION_ORDER if c not in {test_condition, validation_condition}]
            fold_id = f"S{subject_id:02d}_fold{cond_idx+1:02d}_test{test_condition}"
            for _, row in sub.iterrows():
                condition = str(row["condition"])
                split = "train" if condition in train_conditions else "validation" if condition == validation_condition else "test"
                rows.append(
                    {
                        "subject_id": int(subject_id),
                        "fold_id": fold_id,
                        "fold_index": fold_index,
                        "test_condition": test_condition,
                        "validation_condition": validation_condition,
                        "train_conditions": ";".join(train_conditions),
                        "split": split,
                        "condition": condition,
                        "recording_id": row["recording_id"],
                        "action_code": row["action_code"],
                        "action_meaning": row["action_meaning"],
                        "label_id": int(row["label_id"]),
                        "sample_count": int(row["sample_count"]),
                    }
                )
    return rows


def fit_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    fit_idx = np.arange(len(y_train))
    if model_name == "RBF-SVM-subset":
        chosen = []
        for label in range(len(LABEL_ORDER)):
            idx = np.where(y_train == label)[0]
            if len(idx) > RBF_CAP_PER_CLASS:
                idx = rng.choice(idx, size=RBF_CAP_PER_CLASS, replace=False)
            chosen.extend(idx.tolist())
        fit_idx = np.asarray(sorted(chosen), dtype=int)

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_train[fit_idx])
    y_fit = y_train[fit_idx]

    if model_name == "LogReg":
        model = LogisticRegression(max_iter=1200, C=1.0, class_weight="balanced", random_state=seed, n_jobs=1)
    elif model_name == "ShrinkageLDA":
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    elif model_name == "LinearSVM":
        model = LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=6000)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=180,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    elif model_name == "RBF-SVM-subset":
        model = SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)
    else:
        raise ValueError(model_name)
    model.fit(x_fit, y_fit)
    return {"model_name": model_name, "scaler": scaler, "model": model, "train_samples_used": int(len(y_fit))}


def predict_proba(bundle: dict, x_eval: np.ndarray) -> np.ndarray:
    x_scaled = bundle["scaler"].transform(x_eval)
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        return align_matrix(model.predict_proba(x_scaled), model.classes_)
    scores = model.decision_function(x_scaled)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    probs = softmax(scores)
    return align_matrix(probs, model.classes_)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    p, r, f1, _support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABEL_ORDER))), average="macro", zero_division=0
    )
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_precision": float(p),
        f"{prefix}_macro_recall": float(r),
        f"{prefix}_macro_f1": float(f1),
    }


def aggregate_recordings(meta_split: pd.DataFrame, probs: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    tmp = meta_split[["subject_id", "condition", "recording_id", "action_code", "action_meaning", "label_id", "feature_row_index"]].reset_index(drop=True).copy()
    tmp["_row"] = np.arange(len(tmp))
    rows: list[dict] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    for recording_id, group in tmp.groupby("recording_id", sort=False):
        idx = group["_row"].to_numpy(dtype=int)
        avg_probs = probs[idx].mean(axis=0)
        pred = int(np.argmax(avg_probs))
        true = int(group["label_id"].iloc[0])
        row = {
            "subject_id": int(group["subject_id"].iloc[0]),
            "condition": group["condition"].iloc[0],
            "recording_id": recording_id,
            "true_label_id": true,
            "true_action_code": ID_TO_LABEL[true],
            "pred_label_id": pred,
            "pred_action_code": ID_TO_LABEL[pred],
            "sample_count": int(len(group)),
        }
        for label_id, code in ID_TO_LABEL.items():
            row[f"prob_{code}"] = float(avg_probs[label_id])
        rows.append(row)
        y_true.append(true)
        y_pred.append(pred)
    return pd.DataFrame(rows), np.asarray(y_true), np.asarray(y_pred)


def per_class_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(LABEL_ORDER))), zero_division=0)
    rows = []
    for idx, code in ID_TO_LABEL.items():
        rows.append(
            {
                "class_id": idx,
                "action_code": code,
                "action_meaning": ACTION_MEANING[code],
                "precision": float(p[idx]),
                "recall": float(r[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    return rows


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))
    rows = []
    for idx, code in ID_TO_LABEL.items():
        row = {"true_label_id": idx, "true_action_code": code}
        for j, pred_code in ID_TO_LABEL.items():
            row[f"pred_{j}_{pred_code}"] = int(cm[idx, j])
        rows.append(row)
    return rows


def mean_std(rows: list[dict], key: str) -> tuple[float, float, float, float]:
    vals = np.asarray([float(r[key]) for r in rows], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=0)), float(vals.min()), float(vals.max())


def setup_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Microsoft JhengHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def make_confusion_figure(cm_path: Path) -> None:
    setup_chinese_font()
    df = pd.read_csv(cm_path, encoding="utf-8-sig")
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    cm = df[pred_cols].to_numpy(dtype=int)
    labels = [f"{code}\n{ACTION_CN[code]}" for code in LABEL_ORDER]
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=180)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Bingbin_Realtime calibrated 7-class recording confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bingbin_realtime_demo_confusion_cn.png", bbox_inches="tight")
    plt.close(fig)


def make_timeline_figure(timeline_rows: list[dict]) -> None:
    if not timeline_rows:
        return
    setup_chinese_font()
    df = pd.DataFrame(timeline_rows)
    label_codes = [ID_TO_LABEL[int(x)] for x in df["pred_label_id"]]
    y = [LABEL_TO_ID[c] for c in label_codes]
    x = np.arange(len(y))
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=180)
    ax.plot(x, y, linewidth=1.5, color="#4C78A8")
    true_code = df["true_action_code"].iloc[0]
    ax.axhline(LABEL_TO_ID[true_code], color="#2CA02C", linestyle="--", linewidth=1.2, label=f"True action {true_code}")
    ax.set_yticks(range(len(LABEL_ORDER)), [f"{c}-{ACTION_CN[c]}" for c in LABEL_ORDER])
    ax.set_xlabel("Sample order / simulated replay time")
    ax.set_ylabel("Predicted action")
    ax.set_title("Real-time replay demonstration: per-sample prediction trace")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bingbin_realtime_timeline_demo_cn.png", bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict, selected_counts: Counter, target_met: bool) -> None:
    status = "GO_DEMO_MAIN_DATASET" if target_met else "NO_GO_FOR_MAIN_DEMO_HIGH_PERFORMANCE"
    report = f"""# Stage 13 Bingbin_Realtime Standalone Main-Demo Calibrated Experiment

## Scope

- Dataset: Bingbin_Realtime only.
- Limb Position was not used for training, preprocessing, model selection, or threshold selection.
- Task: standalone 7-class classification for ET, FL, HC, HO, PN, RT, SN.
- Protocol: subject-dependent calibrated split. For each subject, each fold holds out one complete condition as test, one condition as validation, and uses the remaining two conditions for training.
- Minimum split unit: recording. No recording is shared across train, validation, and test within a fold.

## Result

- Overall recording Macro-F1: {summary['overall_recording_macro_f1']:.4f}
- Overall recording Accuracy: {summary['overall_recording_accuracy']:.4f}
- Overall recording Balanced Accuracy: {summary['overall_recording_balanced_accuracy']:.4f}
- Mean fold recording Macro-F1: {summary['fold_recording_macro_f1_mean']:.4f} +/- {summary['fold_recording_macro_f1_std']:.4f}
- Overall sample Macro-F1: {summary['overall_sample_macro_f1']:.4f}
- Overall sample Accuracy: {summary['overall_sample_accuracy']:.4f}
- Selected model counts: {dict(selected_counts)}
- Gate Macro-F1 >= 0.70: {target_met}

## Decision

- Stage 13 status: {status}.
- If used for demonstration, this is a Bingbin-only calibrated/subject-dependent result.
- It does not mean Limb and Bingbin can be merged.
- It does not mean Stage 9B cross-dataset external validation passed.
- It does not mean a new user can use the model without calibration.

## Leakage Control

- Recording IDs are split atomically.
- Scalers and models are fitted only on train for validation selection.
- After selecting the model using validation Macro-F1, the final fold model is fitted on train+validation and evaluated once on held-out test recordings.
- Test recordings are not used for model or parameter selection.

## Demo Use

The optional figures can support a Realtime standalone demo:

- `figures/bingbin_realtime_demo_confusion_cn.png`
- `figures/bingbin_realtime_timeline_demo_cn.png`

The demo can show current action, predicted action, confidence/probability, and a 7-class action mapping. It remains an offline data playback demo unless hardware integration is separately implemented.
"""
    (OUT / "BINGBIN_REALTIME_MAIN_DEMO_REPORT.md").write_text(report, encoding="utf-8")


def write_leakage_audit(split_rows: list[dict], fold_rows: list[dict]) -> None:
    df = pd.DataFrame(split_rows)
    overlap_issues = []
    for (subject_id, fold_id), group in df.groupby(["subject_id", "fold_id"]):
        split_sets = {
            split: set(group[group["split"].eq(split)]["recording_id"])
            for split in ["train", "validation", "test"]
        }
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            inter = split_sets[a].intersection(split_sets[b])
            if inter:
                overlap_issues.append({"subject_id": subject_id, "fold_id": fold_id, "splits": f"{a}/{b}", "recordings": sorted(inter)})
    text = [
        "# Stage 13 Bingbin Main-Demo Leakage Audit",
        "",
        f"- Fold count: {len(fold_rows)}.",
        f"- Split manifest rows: {len(split_rows)}.",
        "- Atomic unit: recording_id.",
        "- Split protocol: condition-level holdout within each subject.",
        "- Model selection: validation recording Macro-F1 only.",
        "- Final fold fit: train+validation after model selection; test held out until final evaluation.",
        "- Limb Position data used: false.",
        "- Bingbin used for Limb tuning: false.",
        "",
    ]
    if overlap_issues:
        text.append("## Status: FAIL")
        text.append(json.dumps(native(overlap_issues), ensure_ascii=False, indent=2))
    else:
        text.append("## Status: PASS")
        text.append("No recording_id overlap across train/validation/test within any subject/fold.")
    (OUT / "BINGBIN_MAIN_DEMO_LEAKAGE_AUDIT.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_PATH.exists():
        LOG_PATH.unlink()
    log("Loading Bingbin precomputed features")
    meta, x, schema_rows = load_bingbin_features()
    y = meta["label_id"].to_numpy(dtype=int)
    log(f"Loaded x={x.shape}, samples={len(meta)}, recordings={meta['recording_id'].nunique()}")

    split_rows = build_split_manifest(meta)
    write_csv(OUT / "BINGBIN_MAIN_DEMO_SPLIT_MANIFEST.csv", split_rows)

    fold_rows: list[dict] = []
    selection_rows: list[dict] = []
    all_sample_true: list[int] = []
    all_sample_pred: list[int] = []
    all_rec_true: list[int] = []
    all_rec_pred: list[int] = []
    all_recording_result_rows: list[dict] = []
    timeline_rows: list[dict] = []

    split_df = pd.DataFrame(split_rows)
    for (subject_id, fold_id), split_group in split_df.groupby(["subject_id", "fold_id"], sort=False):
        fold_meta = {
            "subject_id": int(subject_id),
            "fold_id": fold_id,
            "fold_index": int(split_group["fold_index"].iloc[0]),
            "test_condition": split_group["test_condition"].iloc[0],
            "validation_condition": split_group["validation_condition"].iloc[0],
            "train_conditions": split_group["train_conditions"].iloc[0],
        }
        log(f"Fold {fold_id}: selecting model")
        train_recs = set(split_group[split_group["split"].eq("train")]["recording_id"])
        val_recs = set(split_group[split_group["split"].eq("validation")]["recording_id"])
        test_recs = set(split_group[split_group["split"].eq("test")]["recording_id"])
        train_mask = meta["recording_id"].isin(train_recs).to_numpy()
        val_mask = meta["recording_id"].isin(val_recs).to_numpy()
        test_mask = meta["recording_id"].isin(test_recs).to_numpy()
        trainval_mask = train_mask | val_mask

        candidates: list[dict] = []
        for model_idx, model_name in enumerate(MODEL_ORDER):
            start = time.perf_counter()
            bundle = fit_model(model_name, x[train_mask], y[train_mask], SEED + fold_meta["fold_index"] * 100 + model_idx)
            fit_seconds = time.perf_counter() - start
            val_probs = predict_proba(bundle, x[val_mask])
            val_pred = val_probs.argmax(axis=1)
            val_sample_metrics = metric_dict(y[val_mask], val_pred, "val_sample")
            val_rec_df, val_rec_true, val_rec_pred = aggregate_recordings(meta[val_mask], val_probs)
            val_rec_metrics = metric_dict(val_rec_true, val_rec_pred, "val_recording")
            row = {
                **fold_meta,
                "model_name": model_name,
                "selection_fit_seconds": fit_seconds,
                "selection_train_samples_used": bundle["train_samples_used"],
                **val_sample_metrics,
                **val_rec_metrics,
            }
            selection_rows.append(row)
            candidates.append(row)
        candidates.sort(key=lambda r: (-float(r["val_recording_macro_f1"]), MODEL_ORDER.index(r["model_name"])))
        best = candidates[0]
        selected_model = best["model_name"]

        log(f"Fold {fold_id}: selected {selected_model} val_recording_macro_f1={best['val_recording_macro_f1']:.4f}")
        start = time.perf_counter()
        final_bundle = fit_model(selected_model, x[trainval_mask], y[trainval_mask], SEED + fold_meta["fold_index"] * 1000)
        final_fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        test_probs = predict_proba(final_bundle, x[test_mask])
        inference_seconds = time.perf_counter() - start
        test_pred = test_probs.argmax(axis=1)
        test_sample_metrics = metric_dict(y[test_mask], test_pred, "test_sample")
        test_rec_df, test_rec_true, test_rec_pred = aggregate_recordings(meta[test_mask], test_probs)
        test_rec_metrics = metric_dict(test_rec_true, test_rec_pred, "test_recording")

        for _, row in test_rec_df.iterrows():
            out_row = {**fold_meta, "model_name": selected_model, **row.to_dict()}
            all_recording_result_rows.append(out_row)

        if not timeline_rows:
            first_rec = test_rec_df["recording_id"].iloc[0]
            rec_mask_local = meta[test_mask]["recording_id"].reset_index(drop=True).eq(first_rec).to_numpy()
            rec_meta = meta[test_mask].reset_index(drop=True)[rec_mask_local].copy()
            rec_probs = test_probs[rec_mask_local]
            rec_pred = rec_probs.argmax(axis=1)
            conf = rec_probs.max(axis=1)
            stride = max(1, len(rec_pred) // 180)
            for i in range(0, len(rec_pred), stride):
                timeline_rows.append(
                    {
                        "sample_order": int(i),
                        "recording_id": first_rec,
                        "true_action_code": rec_meta["action_code"].iloc[i],
                        "pred_label_id": int(rec_pred[i]),
                        "pred_action_code": ID_TO_LABEL[int(rec_pred[i])],
                        "confidence": float(conf[i]),
                    }
                )

        all_sample_true.extend(y[test_mask].tolist())
        all_sample_pred.extend(test_pred.tolist())
        all_rec_true.extend(test_rec_true.tolist())
        all_rec_pred.extend(test_rec_pred.tolist())

        fold_rows.append(
            {
                **fold_meta,
                "selected_model": selected_model,
                "validation_selected_recording_macro_f1": best["val_recording_macro_f1"],
                "train_recording_count": int(split_group["split"].eq("train").sum()),
                "validation_recording_count": int(split_group["split"].eq("validation").sum()),
                "test_recording_count": int(split_group["split"].eq("test").sum()),
                "train_sample_count": int(train_mask.sum()),
                "validation_sample_count": int(val_mask.sum()),
                "test_sample_count": int(test_mask.sum()),
                "final_fit_scope": "train_plus_validation_after_model_selection",
                "final_fit_seconds": final_fit_seconds,
                "avg_inference_ms_per_sample": float(inference_seconds / max(int(test_mask.sum()), 1) * 1000.0),
                "train_samples_used": final_bundle["train_samples_used"],
                **test_sample_metrics,
                **test_rec_metrics,
            }
        )

    all_sample_true_arr = np.asarray(all_sample_true, dtype=int)
    all_sample_pred_arr = np.asarray(all_sample_pred, dtype=int)
    all_rec_true_arr = np.asarray(all_rec_true, dtype=int)
    all_rec_pred_arr = np.asarray(all_rec_pred, dtype=int)

    sample_overall = metric_dict(all_sample_true_arr, all_sample_pred_arr, "overall_sample")
    rec_overall = metric_dict(all_rec_true_arr, all_rec_pred_arr, "overall_recording")
    rec_mean, rec_std, rec_min, rec_max = mean_std(fold_rows, "test_recording_macro_f1")
    sample_mean, sample_std, sample_min, sample_max = mean_std(fold_rows, "test_sample_macro_f1")
    selected_counts = Counter(row["selected_model"] for row in fold_rows)
    target_met = rec_overall["overall_recording_macro_f1"] >= 0.70
    stage_status = "GO_DEMO_MAIN_DATASET" if target_met else "NO_GO_FOR_MAIN_DEMO_HIGH_PERFORMANCE"

    summary = {
        "stage": "Stage 13 Bingbin_Realtime standalone calibrated main-demo experiment",
        "stage13_status": stage_status,
        "dataset": "Bingbin_Realtime",
        "limb_data_used": False,
        "mixed_with_limb": False,
        "task": "standalone 7-class ET/FL/HC/HO/PN/RT/SN",
        "protocol": "subject-dependent condition/recording-level calibrated split",
        "subject_count": int(meta["subject_id"].nunique()),
        "fold_count": len(fold_rows),
        "recording_count": int(meta["recording_id"].nunique()),
        "test_recording_count": len(all_rec_true_arr),
        "test_sample_count": len(all_sample_true_arr),
        "selected_model_counts": dict(selected_counts),
        **sample_overall,
        **rec_overall,
        "fold_recording_macro_f1_mean": rec_mean,
        "fold_recording_macro_f1_std": rec_std,
        "fold_recording_macro_f1_min": rec_min,
        "fold_recording_macro_f1_max": rec_max,
        "fold_sample_macro_f1_mean": sample_mean,
        "fold_sample_macro_f1_std": sample_std,
        "fold_sample_macro_f1_min": sample_min,
        "fold_sample_macro_f1_max": sample_max,
        "target_recording_macro_f1_ge_0p70": bool(target_met),
    }

    write_csv(OUT / "BINGBIN_MAIN_DEMO_SUMMARY.csv", [summary])
    write_csv(OUT / "BINGBIN_MAIN_DEMO_PER_FOLD_OR_BLOCK.csv", fold_rows)
    write_csv(OUT / "BINGBIN_MAIN_DEMO_MODEL_SELECTION.csv", selection_rows)
    write_csv(OUT / "BINGBIN_MAIN_DEMO_RECORDING_RESULTS.csv", all_recording_result_rows)
    write_csv(OUT / "BINGBIN_MAIN_DEMO_PER_CLASS.csv", per_class_rows(all_rec_true_arr, all_rec_pred_arr))
    write_csv(OUT / "BINGBIN_MAIN_DEMO_CONFUSION_MATRIX.csv", confusion_rows(all_rec_true_arr, all_rec_pred_arr))

    per_subject_rows = []
    rec_df = pd.DataFrame({"subject_id": [r["subject_id"] for r in all_recording_result_rows], "y_true": all_rec_true_arr, "y_pred": all_rec_pred_arr})
    for subject_id, group in rec_df.groupby("subject_id"):
        per_subject_rows.append({"subject_id": int(subject_id), **metric_dict(group["y_true"].to_numpy(), group["y_pred"].to_numpy(), "recording")})
    write_csv(OUT / "BINGBIN_MAIN_DEMO_PER_SUBJECT.csv", per_subject_rows)

    write_leakage_audit(split_rows, fold_rows)
    write_report(summary, selected_counts, target_met)
    make_confusion_figure(OUT / "BINGBIN_MAIN_DEMO_CONFUSION_MATRIX.csv")
    make_timeline_figure(timeline_rows)

    decision = {
        **summary,
        "decision": {
            "can_use_realtime_as_application_demo_main_dataset": bool(target_met),
            "reason": "recording Macro-F1 >= 0.70 under Bingbin-only calibrated protocol" if target_met else "recording Macro-F1 below 0.70 under Bingbin-only calibrated protocol",
            "not_cross_dataset_generalization": True,
            "not_zero_calibration_new_user": True,
            "do_not_merge_with_limb": True,
        },
        "outputs": {
            "summary": str(OUT / "BINGBIN_MAIN_DEMO_SUMMARY.csv"),
            "per_subject": str(OUT / "BINGBIN_MAIN_DEMO_PER_SUBJECT.csv"),
            "per_class": str(OUT / "BINGBIN_MAIN_DEMO_PER_CLASS.csv"),
            "per_fold_or_block": str(OUT / "BINGBIN_MAIN_DEMO_PER_FOLD_OR_BLOCK.csv"),
            "split_manifest": str(OUT / "BINGBIN_MAIN_DEMO_SPLIT_MANIFEST.csv"),
            "leakage_audit": str(OUT / "BINGBIN_MAIN_DEMO_LEAKAGE_AUDIT.md"),
            "report": str(OUT / "BINGBIN_REALTIME_MAIN_DEMO_REPORT.md"),
        },
    }
    (OUT / "BINGBIN_MAIN_DEMO_DECISION.json").write_text(json.dumps(native(decision), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Stage 13 complete: {stage_status}, recording Macro-F1={rec_overall['overall_recording_macro_f1']:.4f}")


if __name__ == "__main__":
    main()

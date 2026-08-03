from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import time
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
STAGE13_DIR = ROOT / "protocols" / "bingbin"
STAGE13_BASE_SCRIPT = Path(__file__).resolve().parent / "bingbin_protocol_adapter.py"
STAGE14_DIR = ROOT / "protocols" / "bingbin"
OUT = ROOT / "outputs" / "bingbin_realtime"
LOG_PATH = OUT / "stage15_bingbin_v3_run.log"
SEED = 42
V1_MACRO_F1 = 0.753406181631988
TARGET_MACRO_F1 = 0.83

spec = importlib.util.spec_from_file_location("stage13_base", STAGE13_BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

LABEL_ORDER = base.LABEL_ORDER
LABEL_TO_ID = base.LABEL_TO_ID
ID_TO_LABEL = base.ID_TO_LABEL
ACTION_MEANING = base.ACTION_MEANING
ACTION_NAME = {
    "ET": "wrist extension",
    "FL": "wrist flexion",
    "HC": "hand close",
    "HO": "hand open",
    "PN": "forearm pronation",
    "RT": "rest",
    "SN": "forearm supination",
}


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict:
    p, r, f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
        average="macro",
        zero_division=0,
    )
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_precision": float(p),
        f"{prefix}_macro_recall": float(r),
        f"{prefix}_macro_f1": float(f1),
    }


def per_class_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_ORDER))),
        zero_division=0,
    )
    return [
        {
            "class_id": idx,
            "action_code": ID_TO_LABEL[idx],
            "action_name": ACTION_NAME[ID_TO_LABEL[idx]],
            "action_meaning": ACTION_MEANING[ID_TO_LABEL[idx]],
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx in range(len(LABEL_ORDER))
    ]


def confusion_rows(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_ORDER))))
    rows = []
    for i, code in ID_TO_LABEL.items():
        row = {"true_label_id": i, "true_action_code": code, "true_action_name": ACTION_NAME[code]}
        for j, pred_code in ID_TO_LABEL.items():
            row[f"pred_{j}_{pred_code}"] = int(cm[i, j])
        rows.append(row)
    return rows


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
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


def predict_proba_estimator(estimator, x_eval: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return align_matrix(estimator.predict_proba(x_eval), estimator.classes_)
    scores = estimator.decision_function(x_eval)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    return align_matrix(softmax(scores), estimator.classes_)


def robust_scale_recording(arr: np.ndarray, clip: float = 8.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    med = np.median(arr, axis=0, keepdims=True)
    q25 = np.percentile(arr, 25, axis=0, keepdims=True)
    q75 = np.percentile(arr, 75, axis=0, keepdims=True)
    iqr = np.maximum(q75 - q25, 1e-8)
    return np.clip((arr - med) / iqr, -clip, clip).astype(np.float32)


def chunk_bounds(length: int, n_chunks: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, length, n_chunks + 1).round().astype(int)
    out = []
    for i in range(n_chunks):
        start = int(edges[i])
        end = int(edges[i + 1])
        if end <= start:
            end = min(length, start + 1)
        out.append((start, end))
    return out


def channel_features(seg: np.ndarray) -> np.ndarray:
    if seg.shape[0] < 2:
        diff = np.zeros_like(seg)
    else:
        diff = np.diff(seg, axis=0)
    mean = seg.mean(axis=0)
    std = seg.std(axis=0)
    median = np.median(seg, axis=0)
    q25 = np.percentile(seg, 25, axis=0)
    q75 = np.percentile(seg, 75, axis=0)
    iqr = q75 - q25
    mav = np.mean(np.abs(seg), axis=0)
    rms = np.sqrt(np.mean(seg * seg, axis=0))
    ptp = np.ptp(seg, axis=0)
    wl = np.mean(np.abs(diff), axis=0) if diff.size else np.zeros(seg.shape[1], dtype=np.float32)
    sign = np.signbit(seg)
    zcr = np.mean(sign[1:] != sign[:-1], axis=0) if seg.shape[0] > 1 else np.zeros(seg.shape[1], dtype=np.float32)
    sdiff = np.signbit(diff)
    ssc = np.mean(sdiff[1:] != sdiff[:-1], axis=0) if diff.shape[0] > 1 else np.zeros(seg.shape[1], dtype=np.float32)
    return np.concatenate([mean, std, median, iqr, mav, rms, ptp, wl, zcr, ssc]).astype(np.float32)


def pooled_features(seg: np.ndarray) -> np.ndarray:
    if seg.shape[0] < 2:
        diff = np.zeros_like(seg)
    else:
        diff = np.diff(seg, axis=0)
    metrics = [
        seg.mean(axis=0),
        seg.std(axis=0),
        np.mean(np.abs(seg), axis=0),
        np.sqrt(np.mean(seg * seg, axis=0)),
        np.ptp(seg, axis=0),
        np.mean(np.abs(diff), axis=0) if diff.size else np.zeros(seg.shape[1], dtype=np.float32),
    ]
    pooled = []
    for values in metrics:
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        pooled.extend([values.mean(), values.std(), np.median(values), q75 - q25, values.min(), values.max()])
    return np.asarray(pooled, dtype=np.float32)


def extract_raw_window_features(meta: pd.DataFrame, x_precomp: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    tmp = meta.reset_index(drop=True).copy()
    tmp["_row"] = np.arange(len(tmp))
    raw_emg = np.zeros((len(tmp), 160), dtype=np.float32)
    raw_imu_pool = np.zeros((len(tmp), 36), dtype=np.float32)
    audit_rows: list[dict] = []
    for rec_idx, (recording_id, group) in enumerate(tmp.groupby("recording_id", sort=False), start=1):
        if rec_idx % 20 == 1:
            log(f"Extracting raw features: recording {rec_idx}")
        source_file = ROOT / str(group["source_file"].iloc[0])
        action_code = str(group["action_code"].iloc[0])
        action_file = source_file.parent / f"{action_code}.mat"
        mat = sio.loadmat(action_file, simplify_cells=True)
        emg = robust_scale_recording(np.asarray(mat["rawEMG"], dtype=np.float32))
        imu = robust_scale_recording(np.asarray(mat["rawIMUData"], dtype=np.float32))
        n = len(group)
        emg_bounds = chunk_bounds(emg.shape[0], n)
        imu_bounds = chunk_bounds(imu.shape[0], n)
        rows = group["_row"].to_numpy(dtype=int)
        for local_i, row_idx in enumerate(rows):
            es, ee = emg_bounds[local_i]
            is_, ie = imu_bounds[local_i]
            raw_emg[row_idx] = channel_features(emg[es:ee])
            raw_imu_pool[row_idx] = pooled_features(imu[is_:ie])
        audit_rows.append(
            {
                "recording_id": recording_id,
                "subject_id": int(group["subject_id"].iloc[0]),
                "condition": group["condition"].iloc[0],
                "action_code": action_code,
                "feature_rows": int(n),
                "raw_emg_shape": f"{emg.shape[0]}x{emg.shape[1]}",
                "raw_imu_shape": f"{imu.shape[0]}x{imu.shape[1]}",
                "source_action_file": str(action_file.relative_to(ROOT)),
            }
        )
    x_raw_all = np.concatenate([x_precomp.astype(np.float32), raw_emg, raw_imu_pool], axis=1)
    return x_raw_all.astype(np.float32, copy=False), audit_rows


def recording_summary_features(meta: pd.DataFrame, x: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    tmp = meta.reset_index(drop=True).copy()
    tmp["_row"] = np.arange(len(tmp))
    rows = []
    feats = []
    for recording_id, group in tmp.groupby("recording_id", sort=False):
        idx = group["_row"].to_numpy(dtype=int)
        arr = x[idx].astype(np.float64, copy=False)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        feats.append(np.concatenate([arr.mean(axis=0), arr.std(axis=0), np.median(arr, axis=0), q75 - q25]).astype(np.float32))
        rows.append(
            {
                "subject_id": int(group["subject_id"].iloc[0]),
                "condition": group["condition"].iloc[0],
                "recording_id": recording_id,
                "action_code": group["action_code"].iloc[0],
                "label_id": int(group["label_id"].iloc[0]),
                "sample_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows), np.vstack(feats).astype(np.float32)


def make_model(model_key: str, seed: int):
    if model_key == "logreg":
        return LogisticRegression(max_iter=2500, C=1.0, class_weight="balanced", random_state=seed, n_jobs=1)
    if model_key == "logreg_c3":
        return LogisticRegression(max_iter=2500, C=3.0, class_weight="balanced", random_state=seed, n_jobs=1)
    if model_key == "lda":
        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if model_key == "linear_svm":
        return LinearSVC(C=0.7, class_weight="balanced", random_state=seed, max_iter=9000)
    if model_key == "rbf_svm":
        return SVC(C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=seed)
    if model_key == "extra_trees":
        return ExtraTreesClassifier(n_estimators=260, max_features="sqrt", min_samples_leaf=1, class_weight="balanced", random_state=seed, n_jobs=-1)
    if model_key == "random_forest":
        return RandomForestClassifier(n_estimators=220, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    if model_key == "hist_gb":
        return HistGradientBoostingClassifier(max_iter=220, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=0.03, random_state=seed)
    if model_key == "knn1":
        return KNeighborsClassifier(n_neighbors=1, weights="distance")
    if model_key == "knn3":
        return KNeighborsClassifier(n_neighbors=3, weights="distance")
    if model_key == "xgb":
        if XGBClassifier is None:
            raise RuntimeError("xgboost unavailable")
        return XGBClassifier(
            n_estimators=160,
            max_depth=3,
            learning_rate=0.045,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(model_key)


def candidate_specs() -> list[dict]:
    return [
        {"name": "raw_all_logreg_pca80", "level": "sample", "model": "logreg", "pca": 80},
        {"name": "raw_all_linear_svm_pca80", "level": "sample", "model": "linear_svm", "pca": 80},
        {"name": "raw_all_lda_k120", "level": "sample", "model": "lda", "select_k": 120},
        {"name": "raw_all_extra_trees_subset", "level": "sample", "model": "extra_trees", "select_k": 100, "subset_per_class": 1800},
        {"name": "raw_all_hist_gb_subset", "level": "sample", "model": "hist_gb", "select_k": 80, "subset_per_class": 1600},
        {"name": "recording_raw_logreg_k120", "level": "recording", "model": "logreg", "select_k": 120},
        {"name": "recording_raw_rbf_k80", "level": "recording", "model": "rbf_svm", "select_k": 80},
        {"name": "recording_raw_extra_trees", "level": "recording", "model": "extra_trees", "select_k": 160},
        {"name": "recording_raw_xgb_k120", "level": "recording", "model": "xgb", "select_k": 120},
        {"name": "recording_raw_knn1_k80", "level": "recording", "model": "knn1", "select_k": 80},
        {"name": "recording_raw_vote_k120", "level": "recording_ensemble", "select_k": 120},
    ]


def build_pipeline(cand: dict, seed: int, x_train: np.ndarray) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if cand.get("select_k"):
        steps.append(("select", SelectKBest(f_classif, k=min(int(cand["select_k"]), x_train.shape[1]))))
    if cand.get("pca"):
        n = min(int(cand["pca"]), x_train.shape[1], max(1, x_train.shape[0] - 1))
        steps.append(("pca", PCA(n_components=n, random_state=seed)))
    if cand["level"] == "recording_ensemble":
        voters = [
            ("logreg", make_model("logreg", seed)),
            ("lda", make_model("lda", seed)),
            ("extra", make_model("extra_trees", seed)),
        ]
        steps.append(("model", VotingClassifier(estimators=voters, voting="soft", n_jobs=1)))
    else:
        steps.append(("model", make_model(cand["model"], seed)))
    return Pipeline(steps)


def choose_subset(y_train: np.ndarray, per_class: int | None, seed: int) -> np.ndarray:
    if not per_class:
        return np.arange(len(y_train), dtype=int)
    rng = np.random.default_rng(seed)
    selected = []
    for label in range(len(LABEL_ORDER)):
        idx = np.where(y_train == label)[0]
        if len(idx) > per_class:
            idx = rng.choice(idx, size=per_class, replace=False)
        selected.extend(idx.tolist())
    return np.asarray(sorted(selected), dtype=int)


def aggregate_recording(meta_split: pd.DataFrame, probs: np.ndarray, method: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    tmp = meta_split[["subject_id", "condition", "recording_id", "action_code", "label_id", "feature_row_index"]].reset_index(drop=True).copy()
    tmp["_row"] = np.arange(len(tmp))
    sample_pred = probs.argmax(axis=1)
    rows = []
    y_true = []
    y_pred = []
    for recording_id, group in tmp.groupby("recording_id", sort=False):
        idx = group["_row"].to_numpy(dtype=int)
        local = probs[idx]
        if method == "mean_prob":
            rec_probs = local.mean(axis=0)
        elif method == "confidence_weighted":
            w = np.maximum(local.max(axis=1), 1e-6)
            rec_probs = np.average(local, axis=0, weights=w)
        elif method == "majority_vote":
            counts = np.bincount(sample_pred[idx], minlength=len(LABEL_ORDER)).astype(float)
            rec_probs = counts / np.maximum(counts.sum(), 1)
        else:
            raise ValueError(method)
        rec_probs = rec_probs / np.maximum(rec_probs.sum(), 1e-12)
        true = int(group["label_id"].iloc[0])
        pred = int(rec_probs.argmax())
        row = {
            "subject_id": int(group["subject_id"].iloc[0]),
            "condition": group["condition"].iloc[0],
            "recording_id": recording_id,
            "true_label_id": true,
            "true_action_code": ID_TO_LABEL[true],
            "true_action_name": ACTION_NAME[ID_TO_LABEL[true]],
            "pred_label_id": pred,
            "pred_action_code": ID_TO_LABEL[pred],
            "pred_action_name": ACTION_NAME[ID_TO_LABEL[pred]],
            "sample_count": int(len(group)),
            "aggregation_method": method,
        }
        for label_id, code in ID_TO_LABEL.items():
            row[f"prob_{code}"] = float(rec_probs[label_id])
        rows.append(row)
        y_true.append(true)
        y_pred.append(pred)
    return pd.DataFrame(rows), np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)


def predict_recording_direct(estimator, rec_meta: pd.DataFrame, rec_x: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    probs = predict_proba_estimator(estimator, rec_x)
    pred = probs.argmax(axis=1).astype(int)
    true = rec_meta["label_id"].to_numpy(dtype=int)
    rows = []
    for i, (_, row) in enumerate(rec_meta.reset_index(drop=True).iterrows()):
        out = {
            "subject_id": int(row["subject_id"]),
            "condition": row["condition"],
            "recording_id": row["recording_id"],
            "true_label_id": int(true[i]),
            "true_action_code": ID_TO_LABEL[int(true[i])],
            "true_action_name": ACTION_NAME[ID_TO_LABEL[int(true[i])]],
            "pred_label_id": int(pred[i]),
            "pred_action_code": ID_TO_LABEL[int(pred[i])],
            "pred_action_name": ACTION_NAME[ID_TO_LABEL[int(pred[i])]],
            "sample_count": int(row["sample_count"]),
            "aggregation_method": "recording_direct",
        }
        for label_id, code in ID_TO_LABEL.items():
            out[f"prob_{code}"] = float(probs[i, label_id])
        rows.append(out)
    return pd.DataFrame(rows), true, pred


def sample_metrics_from_recording(meta_split: pd.DataFrame, rec_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred_map = dict(zip(rec_df["recording_id"], rec_df["pred_label_id"]))
    return meta_split["label_id"].to_numpy(dtype=int), meta_split["recording_id"].map(pred_map).to_numpy(dtype=int)


def validate_split(split_df: pd.DataFrame) -> tuple[bool, list[dict]]:
    issues = []
    for (subject_id, fold_id), group in split_df.groupby(["subject_id", "fold_id"], sort=False):
        sets = {s: set(group[group["split"].eq(s)]["recording_id"]) for s in ["train", "validation", "test"]}
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            inter = sorted(sets[a].intersection(sets[b]))
            if inter:
                issues.append({"subject_id": int(subject_id), "fold_id": fold_id, "splits": f"{a}/{b}", "recordings": ";".join(inter)})
    return not issues, issues


def evaluate(meta: pd.DataFrame, x_raw: np.ndarray, rec_meta: pd.DataFrame, rec_x: np.ndarray, split_df: pd.DataFrame) -> dict:
    y = meta["label_id"].to_numpy(dtype=int)
    rec_index = {rid: i for i, rid in enumerate(rec_meta["recording_id"])}
    specs = candidate_specs()
    aggregations = ["mean_prob", "confidence_weighted", "majority_vote"]

    comparison_rows = []
    fold_rows = []
    recording_rows = []
    all_sample_true = []
    all_sample_pred = []
    all_rec_true = []
    all_rec_pred = []

    for (subject_id, fold_id), split_group in split_df.groupby(["subject_id", "fold_id"], sort=False):
        fold_index = int(split_group["fold_index"].iloc[0])
        fold_meta = {"subject_id": int(subject_id), "fold_id": fold_id, "fold_index": fold_index}
        train_recs = set(split_group[split_group["split"].eq("train")]["recording_id"])
        val_recs = set(split_group[split_group["split"].eq("validation")]["recording_id"])
        test_recs = set(split_group[split_group["split"].eq("test")]["recording_id"])
        trainval_recs = train_recs | val_recs
        train_mask = meta["recording_id"].isin(train_recs).to_numpy()
        val_mask = meta["recording_id"].isin(val_recs).to_numpy()
        test_mask = meta["recording_id"].isin(test_recs).to_numpy()
        trainval_mask = train_mask | val_mask
        train_rec_idx = np.asarray([rec_index[r] for r in rec_meta[rec_meta["recording_id"].isin(train_recs)]["recording_id"]], dtype=int)
        val_rec_idx = np.asarray([rec_index[r] for r in rec_meta[rec_meta["recording_id"].isin(val_recs)]["recording_id"]], dtype=int)
        test_rec_idx = np.asarray([rec_index[r] for r in rec_meta[rec_meta["recording_id"].isin(test_recs)]["recording_id"]], dtype=int)
        trainval_rec_idx = np.asarray([rec_index[r] for r in rec_meta[rec_meta["recording_id"].isin(trainval_recs)]["recording_id"]], dtype=int)

        log(f"{fold_id}: selecting Stage15 V3 candidate")
        candidates = []
        for c_idx, cand in enumerate(specs):
            seed = SEED + fold_index * 100 + c_idx
            try:
                start = time.perf_counter()
                if cand["level"].startswith("sample"):
                    fit_idx = choose_subset(y[train_mask], cand.get("subset_per_class"), seed)
                    estimator = build_pipeline(cand, seed, x_raw[train_mask][fit_idx])
                    estimator.fit(x_raw[train_mask][fit_idx], y[train_mask][fit_idx])
                    fit_seconds = time.perf_counter() - start
                    val_probs = predict_proba_estimator(estimator, x_raw[val_mask])
                    val_sample_pred = val_probs.argmax(axis=1)
                    sample_metrics = metric_dict(y[val_mask], val_sample_pred, "val_sample")
                    for agg in aggregations:
                        _rec_df, rec_true, rec_pred = aggregate_recording(meta[val_mask], val_probs, agg)
                        rec_metrics = metric_dict(rec_true, rec_pred, "val_recording")
                        row = {**fold_meta, "candidate_name": cand["name"], "candidate_level": cand["level"], "aggregation_method": agg, "selection_status": "ok", "selection_fit_seconds": fit_seconds, **sample_metrics, **rec_metrics}
                        comparison_rows.append(row)
                        candidates.append(row)
                else:
                    estimator = build_pipeline(cand, seed, rec_x[train_rec_idx])
                    estimator.fit(rec_x[train_rec_idx], rec_meta.iloc[train_rec_idx]["label_id"].to_numpy(dtype=int))
                    fit_seconds = time.perf_counter() - start
                    rec_df, rec_true, rec_pred = predict_recording_direct(estimator, rec_meta.iloc[val_rec_idx], rec_x[val_rec_idx])
                    rec_metrics = metric_dict(rec_true, rec_pred, "val_recording")
                    val_sample_true, val_sample_pred = sample_metrics_from_recording(meta[val_mask], rec_df)
                    sample_metrics = metric_dict(val_sample_true, val_sample_pred, "val_sample")
                    row = {**fold_meta, "candidate_name": cand["name"], "candidate_level": cand["level"], "aggregation_method": "recording_direct", "selection_status": "ok", "selection_fit_seconds": fit_seconds, **sample_metrics, **rec_metrics}
                    comparison_rows.append(row)
                    candidates.append(row)
            except Exception as exc:
                comparison_rows.append({**fold_meta, "candidate_name": cand["name"], "candidate_level": cand["level"], "aggregation_method": "failed", "selection_status": f"failed: {type(exc).__name__}: {exc}"})
        if not candidates:
            raise RuntimeError(f"No Stage15 candidate succeeded for {fold_id}")
        candidates.sort(key=lambda r: (-float(r["val_recording_macro_f1"]), -float(r["val_recording_balanced_accuracy"]), float(r["selection_fit_seconds"]), r["candidate_name"]))
        best = candidates[0]
        cand = next(s for s in specs if s["name"] == best["candidate_name"])
        log(f"{fold_id}: selected {best['candidate_name']} / {best['aggregation_method']} val_rec_macro_f1={float(best['val_recording_macro_f1']):.4f}")

        final_seed = SEED + fold_index * 1000
        start = time.perf_counter()
        if cand["level"].startswith("sample"):
            fit_idx = choose_subset(y[trainval_mask], cand.get("subset_per_class"), final_seed)
            estimator = build_pipeline(cand, final_seed, x_raw[trainval_mask][fit_idx])
            estimator.fit(x_raw[trainval_mask][fit_idx], y[trainval_mask][fit_idx])
            fit_seconds = time.perf_counter() - start
            infer_start = time.perf_counter()
            test_probs = predict_proba_estimator(estimator, x_raw[test_mask])
            infer_seconds = time.perf_counter() - infer_start
            test_rec_df, rec_true, rec_pred = aggregate_recording(meta[test_mask], test_probs, best["aggregation_method"])
            sample_true = y[test_mask]
            sample_pred = test_probs.argmax(axis=1).astype(int)
        else:
            estimator = build_pipeline(cand, final_seed, rec_x[trainval_rec_idx])
            estimator.fit(rec_x[trainval_rec_idx], rec_meta.iloc[trainval_rec_idx]["label_id"].to_numpy(dtype=int))
            fit_seconds = time.perf_counter() - start
            infer_start = time.perf_counter()
            test_rec_df, rec_true, rec_pred = predict_recording_direct(estimator, rec_meta.iloc[test_rec_idx], rec_x[test_rec_idx])
            infer_seconds = time.perf_counter() - infer_start
            sample_true, sample_pred = sample_metrics_from_recording(meta[test_mask], test_rec_df)
        for _, row in test_rec_df.iterrows():
            recording_rows.append({**fold_meta, "selected_candidate": best["candidate_name"], "candidate_level": cand["level"], **row.to_dict()})
        all_rec_true.extend(rec_true.tolist())
        all_rec_pred.extend(rec_pred.tolist())
        all_sample_true.extend(sample_true.tolist())
        all_sample_pred.extend(sample_pred.tolist())
        fold_rows.append(
            {
                **fold_meta,
                "selected_candidate": best["candidate_name"],
                "candidate_level": cand["level"],
                "selected_aggregation_method": best["aggregation_method"],
                "validation_selected_recording_macro_f1": float(best["val_recording_macro_f1"]),
                "final_fit_scope": "train_plus_validation_after_validation_selection",
                "test_recording_count": int(len(rec_true)),
                "test_sample_count": int(test_mask.sum()),
                "final_fit_seconds": float(fit_seconds),
                "avg_inference_ms_per_recording": float(infer_seconds / max(len(rec_true), 1) * 1000),
                **metric_dict(sample_true, sample_pred, "test_sample"),
                **metric_dict(rec_true, rec_pred, "test_recording"),
            }
        )
    rec_true = np.asarray(all_rec_true, dtype=int)
    rec_pred = np.asarray(all_rec_pred, dtype=int)
    sample_true = np.asarray(all_sample_true, dtype=int)
    sample_pred = np.asarray(all_sample_pred, dtype=int)
    fold_vals = np.asarray([float(r["test_recording_macro_f1"]) for r in fold_rows])
    summary = {
        "stage": "Stage 15 Bingbin_Realtime V3 raw-feature exploratory model search",
        "dataset": "Bingbin_Realtime",
        "limb_data_used": False,
        "mixed_with_limb": False,
        "test_reuse_warning": True,
        "protocol_id": "stage13_stratified_recording_rescue_split_reused",
        "v1_recording_macro_f1": V1_MACRO_F1,
        "target_recording_macro_f1": TARGET_MACRO_F1,
        "subject_count": int(meta["subject_id"].nunique()),
        "fold_count": int(len(fold_rows)),
        "recording_count": int(meta["recording_id"].nunique()),
        **metric_dict(sample_true, sample_pred, "overall_sample"),
        **metric_dict(rec_true, rec_pred, "overall_recording"),
        "fold_recording_macro_f1_mean": float(fold_vals.mean()),
        "fold_recording_macro_f1_std": float(fold_vals.std(ddof=0)),
        "fold_recording_macro_f1_min": float(fold_vals.min()),
        "fold_recording_macro_f1_max": float(fold_vals.max()),
    }
    summary["absolute_macro_f1_delta_vs_v1"] = float(summary["overall_recording_macro_f1"] - V1_MACRO_F1)
    summary["target_0p83_met"] = bool(summary["overall_recording_macro_f1"] >= TARGET_MACRO_F1)
    summary["can_replace_v1_formally"] = False
    return {
        "summary": summary,
        "comparison_rows": comparison_rows,
        "fold_rows": fold_rows,
        "recording_rows": recording_rows,
        "rec_true": rec_true,
        "rec_pred": rec_pred,
        "sample_true": sample_true,
        "sample_pred": sample_pred,
    }


def per_subject_rows(recording_rows: list[dict]) -> list[dict]:
    df = pd.DataFrame(recording_rows)
    rows = []
    for subject_id, group in df.groupby("subject_id", sort=True):
        rows.append({"subject_id": int(subject_id), "recording_count": int(len(group)), **metric_dict(group["true_label_id"].to_numpy(dtype=int), group["pred_label_id"].to_numpy(dtype=int), "recording")})
    return rows


def write_leakage_audit(split_ok: bool, issues: list[dict]) -> None:
    lines = [
        "# Stage 15 Bingbin V3 Leakage Audit",
        "",
        "- Dataset: Bingbin_Realtime only.",
        "- Limb Position used: false.",
        "- Split source: Stage 13 frozen `BINGBIN_MAIN_DEMO_SPLIT_MANIFEST.csv`.",
        "- Atomic split unit: `recording_id`.",
        "- Raw feature extraction is deterministic per recording and does not fit on test labels.",
        "- Scaler/feature selection/PCA/models are fitted only on train for validation selection, then train+validation for final held-out test.",
        "- Warning: Stage 13/14 test results were already known before this Stage 15 exploration, so this is exploratory and not a clean new formal benchmark.",
        "",
    ]
    if split_ok:
        lines += ["## Status: PASS", "", "No recording overlap across train/validation/test was found."]
    else:
        lines += ["## Status: FAIL", "", "```json", json.dumps(native(issues), ensure_ascii=False, indent=2), "```"]
    (OUT / "BINGBIN_V3_LEAKAGE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary: dict, per_subject: list[dict], per_class: list[dict]) -> None:
    s3 = next((r for r in per_subject if int(r["subject_id"]) == 3), None)
    weak = sorted(per_class, key=lambda r: float(r["f1"]))[:3]
    weak_text = "; ".join([f"{r['action_code']} ({r['action_name']}) F1={float(r['f1']):.4f}" for r in weak])
    status = "target met" if summary["target_0p83_met"] else "target not met"
    report = f"""# Stage 15 Bingbin_Realtime V3 Report

## Purpose

This stage derives window features from raw EMG and IMU recordings and compares a set of lightweight classifiers.

## Evaluation boundary

- Only Bingbin_Realtime data are used; Limb Position data are not mixed in.
- The frozen Stage 13 recording-level split is reused.
- No random sample-level or window-level split is used.
- Validation selects the classifier and aggregation method; test data are used for reporting.
- Because earlier Stage 13/14 test results were already inspected, this stage is exploratory and is not presented as a new blinded benchmark.

## Features and models

- Raw EMG: robust scaling within each recording followed by channel-wise statistical and waveform features.
- Raw IMU: robust scaling followed by pooled statistical features.
- Candidate models: logistic regression, LDA, linear SVM, Extra Trees, histogram gradient boosting, RBF-SVM, KNN, XGBoost, and soft voting.

## Result

- Recording accuracy: {summary['overall_recording_accuracy']:.4f}
- Recording balanced accuracy: {summary['overall_recording_balanced_accuracy']:.4f}
- Recording macro-F1: {summary['overall_recording_macro_f1']:.4f}
- Absolute macro-F1 change from V1: {summary['absolute_macro_f1_delta_vs_v1']:+.4f}
- Target 0.83: {status}
- Weakest class: {weak_text}

## Conclusion

The result must retain its exploratory test-reuse caveat. Test splits, difficult subjects, or post-hoc tuning must not be altered to manufacture a higher score.
"""
    (OUT / "BINGBIN_V3_RAW_FEATURE_SEARCH_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log("Stage 15 V3 raw-feature search started")
    meta, x_precomp, _schema_rows = base.load_bingbin_features()
    split_df = pd.read_csv(STAGE13_DIR / "BINGBIN_MAIN_DEMO_SPLIT_MANIFEST.csv", encoding="utf-8-sig")
    split_ok, issues = validate_split(split_df)
    write_leakage_audit(split_ok, issues)
    if not split_ok:
        raise RuntimeError("Split validation failed")
    x_raw, raw_audit = extract_raw_window_features(meta, x_precomp)
    write_csv(OUT / "BINGBIN_V3_RAW_FEATURE_AUDIT.csv", raw_audit)
    rec_meta, rec_x = recording_summary_features(meta, x_raw)
    result = evaluate(meta, x_raw, rec_meta, rec_x, split_df)
    write_csv(OUT / "BINGBIN_V3_MODEL_COMPARISON.csv", result["comparison_rows"])
    write_csv(OUT / "BINGBIN_V3_PER_FOLD_OR_BLOCK.csv", result["fold_rows"])
    write_csv(OUT / "BINGBIN_V3_RECORDING_RESULTS.csv", result["recording_rows"])
    per_subject = per_subject_rows(result["recording_rows"])
    per_class = per_class_rows(result["rec_true"], result["rec_pred"])
    write_csv(OUT / "BINGBIN_V3_PER_SUBJECT.csv", per_subject)
    write_csv(OUT / "BINGBIN_V3_PER_CLASS.csv", per_class)
    write_csv(OUT / "BINGBIN_V3_CONFUSION_MATRIX.csv", confusion_rows(result["rec_true"], result["rec_pred"]))
    summary_rows = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in result["summary"].items()}]
    write_csv(OUT / "BINGBIN_V3_SUMMARY.csv", summary_rows)
    write_report(result["summary"], per_subject, per_class)
    decision = {
        **result["summary"],
        "stage15_status": "DONE",
        "stage13_outputs_modified": False,
        "stage14_outputs_modified": False,
        "limb_data_used": False,
        "mixed_with_limb": False,
        "test_used_for_tuning": False,
        "split_leakage_audit_pass": True,
        "decision": {
            "target_0p83_met": bool(result["summary"]["target_0p83_met"]),
            "can_claim_clean_formal_improvement": False,
            "reason_clean_formal_improvement_false": "Stage13/14 test results were already known; Stage15 is exploratory even if the score improves.",
            "recommended_positioning": "Use only as exploratory V3 evidence; retain V1 unless application accepts exploratory test-reuse caveat.",
        },
        "outputs": {
            "summary": str(OUT / "BINGBIN_V3_SUMMARY.csv"),
            "model_comparison": str(OUT / "BINGBIN_V3_MODEL_COMPARISON.csv"),
            "per_subject": str(OUT / "BINGBIN_V3_PER_SUBJECT.csv"),
            "per_class": str(OUT / "BINGBIN_V3_PER_CLASS.csv"),
            "recording_results": str(OUT / "BINGBIN_V3_RECORDING_RESULTS.csv"),
            "report": str(OUT / "BINGBIN_V3_RAW_FEATURE_SEARCH_REPORT.md"),
            "leakage_audit": str(OUT / "BINGBIN_V3_LEAKAGE_AUDIT.md"),
        },
        "source_hashes": {
            "stage13_split_manifest_sha256": sha256_file(STAGE13_DIR / "BINGBIN_MAIN_DEMO_SPLIT_MANIFEST.csv"),
            "stage14_final_decision_sha256": sha256_file(STAGE14_DIR / "BINGBIN_STAGE14_FINAL_DECISION.json"),
        },
    }
    (OUT / "BINGBIN_V3_DECISION.json").write_text(json.dumps(native(decision), ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Stage 15 complete: recording Macro-F1={result['summary']['overall_recording_macro_f1']:.4f}, target0.83={result['summary']['target_0p83_met']}")


if __name__ == "__main__":
    main()

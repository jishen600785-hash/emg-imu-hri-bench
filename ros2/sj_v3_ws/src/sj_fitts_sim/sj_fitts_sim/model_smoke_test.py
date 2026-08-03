from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .constants import CLASS_NAMES, HELD_OUT_CONDITION
from .feature_extraction import extract_selected_feature
from .model_adapter import OptimizedSjModel


def causal_prediction(
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


def evaluate(model_path: Path, dataset_path: Path) -> dict[str, float]:
    model = OptimizedSjModel(model_path)
    dataset = np.load(dataset_path, allow_pickle=False)
    conditions = np.asarray(dataset["conditions"]).astype(str)
    sources = np.asarray(dataset["source_files"]).astype(str)
    starts = np.asarray(dataset["start_times_sec"], dtype=np.float32)
    labels = np.asarray(dataset["y"], dtype=np.int64)
    basic_all = np.asarray(dataset["X_features"], dtype=np.float32)
    emg_all = np.asarray(dataset["X_emg"], dtype=np.float32)
    imu_all = np.asarray(dataset["X_imu"], dtype=np.float32)
    held_out = conditions == HELD_OUT_CONDITION

    selected_indices = np.flatnonzero(held_out)
    started = time.perf_counter()
    features = extract_selected_feature(
        model.feature_set,
        basic_all[selected_indices],
        emg_all[selected_indices],
        imu_all[selected_indices],
    )
    model_probability = np.asarray(
        model.model.predict_proba(features), dtype=np.float64
    )
    compiled_probability = model._predict_proba(features[:1])
    probability_max_abs_error = float(
        np.max(np.abs(compiled_probability - model_probability[:1]))
    )
    probabilities = np.zeros(
        (len(selected_indices), len(CLASS_NAMES)), dtype=np.float64
    )
    probabilities[:, np.asarray(model.model.classes_, dtype=np.int64)] = (
        model_probability
    )
    raw_prediction = probabilities.argmax(axis=1)
    smoothed_prediction = causal_prediction(
        probabilities,
        sources[selected_indices],
        starts[selected_indices],
        model.history_windows,
    )
    elapsed = time.perf_counter() - started
    truth = labels[selected_indices]

    latency_indices = selected_indices[: min(10, len(selected_indices))]
    latency_started = time.perf_counter()
    model.reset()
    measured_feature_ms: list[float] = []
    measured_model_ms: list[float] = []
    for index in latency_indices:
        measured = model.predict(
            basic_all[index],
            emg_all[index],
            imu_all[index],
        )
        measured_feature_ms.append(measured.feature_ms)
        measured_model_ms.append(measured.model_ms)
    latency_seconds = time.perf_counter() - latency_started

    return {
        "window_count": float(len(selected_indices)),
        "raw_accuracy": float(accuracy_score(truth, raw_prediction)),
        "raw_macro_f1": float(
            f1_score(truth, raw_prediction, average="macro")
        ),
        "smoothed_accuracy": float(
            accuracy_score(truth, smoothed_prediction)
        ),
        "smoothed_macro_f1": float(
            f1_score(truth, smoothed_prediction, average="macro")
        ),
        "batch_total_seconds": float(elapsed),
        "sampled_mean_inference_ms": float(
            latency_seconds / max(len(latency_indices), 1) * 1000.0
        ),
        "probability_max_abs_error": probability_max_abs_error,
        "sampled_mean_feature_ms": float(np.mean(measured_feature_ms)),
        "sampled_mean_model_ms": float(np.mean(measured_model_ms)),
    }


def main(args=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--minimum-accuracy", type=float, default=0.90)
    parsed = parser.parse_args(args)
    root = parsed.project_root.expanduser().resolve()
    dataset_path = parsed.dataset or (
        root
        / "artifacts"
        / "prepared"
        / "window_dataset.npz"
    )
    model_path = parsed.model or (
        root
        / "artifacts"
        / "ml"
        / "optimized"
        / "improved_strict_smoothed_model.joblib"
    )
    metrics = evaluate(model_path, dataset_path)
    print(f"classes={CLASS_NAMES}")
    print(f"held_out_condition={HELD_OUT_CONDITION}")
    print(f"windows={int(metrics['window_count'])}")
    print(
        f"raw_accuracy={metrics['raw_accuracy']:.4f}, "
        f"raw_macro_f1={metrics['raw_macro_f1']:.4f}"
    )
    print(
        f"smoothed_accuracy={metrics['smoothed_accuracy']:.4f}, "
        f"smoothed_macro_f1={metrics['smoothed_macro_f1']:.4f}"
    )
    print(
        f"batch_total_seconds={metrics['batch_total_seconds']:.3f}, "
        f"sampled_mean_inference_ms={metrics['sampled_mean_inference_ms']:.3f}"
    )
    print(
        "compiled_probability_max_abs_error="
        f"{metrics['probability_max_abs_error']:.3e}"
    )
    print(
        f"sampled_mean_feature_ms={metrics['sampled_mean_feature_ms']:.3f}, "
        f"sampled_mean_model_ms={metrics['sampled_mean_model_ms']:.3f}"
    )
    if metrics["probability_max_abs_error"] > 1e-12:
        raise SystemExit(
            "vectorized inference does not match scikit-learn probability"
        )
    if metrics["smoothed_accuracy"] < parsed.minimum_accuracy:
        raise SystemExit(
            f"smoke test failed: {metrics['smoothed_accuracy']:.4f} "
            f"< {parsed.minimum_accuracy:.4f}"
        )


if __name__ == "__main__":
    main()

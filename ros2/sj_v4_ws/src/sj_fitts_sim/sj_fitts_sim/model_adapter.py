from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np

from .constants import CLASS_NAMES
from .feature_extraction import extract_selected_feature


@dataclass(frozen=True)
class Prediction:
    label: int
    raw_label: int
    confidence: float
    raw_confidence: float
    probabilities: list[float]
    raw_probabilities: list[float]
    history_count: int
    feature_ms: float
    model_ms: float

    @property
    def inference_ms(self) -> float:
        return self.feature_ms + self.model_ms


class OptimizedSjModel:
    """Read-only adapter for the optimized Extra Trees SJ model.

    It never fits preprocessing or a classifier during inference. The saved
    estimator and the causal history length selected on static validation
    positions are reused exactly.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        bundle: dict[str, Any] = joblib.load(self.model_path)
        required = {
            "model",
            "feature_set",
            "class_names",
            "causal_smoothing_history_windows",
        }
        if not isinstance(bundle, dict) or not required.issubset(bundle):
            raise ValueError(
                f"Expected optimized SJ bundle keys {sorted(required)}, "
                f"got {sorted(bundle) if isinstance(bundle, dict) else type(bundle)}"
            )
        self.model = bundle["model"]
        # The estimator was trained with n_jobs=-1 for fast batch fitting.
        # A ROS callback predicts one window at a time; spawning a parallel
        # job for every callback is much slower than single-thread inference.
        if hasattr(self.model, "n_jobs"):
            self.model.n_jobs = 1
        estimators = tuple(getattr(self.model, "estimators_", ()))
        self._forest_roots = np.empty(0, dtype=np.int64)
        self._forest_left = np.empty(0, dtype=np.int64)
        self._forest_right = np.empty(0, dtype=np.int64)
        self._forest_feature = np.empty(0, dtype=np.int64)
        self._forest_threshold = np.empty(0, dtype=np.float64)
        self._forest_value = np.empty((0, len(CLASS_NAMES)), dtype=np.float64)
        if estimators:
            self._compile_forest(estimators)
        self.feature_set = str(bundle["feature_set"])
        self.class_names = [str(value) for value in bundle["class_names"]]
        if self.class_names != CLASS_NAMES:
            raise ValueError(
                f"Class order mismatch: model={self.class_names}, expected={CLASS_NAMES}"
            )
        self.history_windows = int(bundle["causal_smoothing_history_windows"])
        if self.history_windows < 1:
            raise ValueError("causal_smoothing_history_windows must be positive")
        self.window_step_seconds = float(bundle.get("window_step_seconds", 0.25))
        self.probability_history: deque[np.ndarray] = deque(
            maxlen=self.history_windows
        )

        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()

    def reset(self) -> None:
        self.probability_history.clear()

    def _compile_forest(self, estimators: tuple[Any, ...]) -> None:
        roots: list[int] = []
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        feature_parts: list[np.ndarray] = []
        threshold_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        offset = 0
        for estimator in estimators:
            tree = estimator.tree_
            roots.append(offset)
            left = np.asarray(tree.children_left, dtype=np.int64).copy()
            right = np.asarray(tree.children_right, dtype=np.int64).copy()
            left[left >= 0] += offset
            right[right >= 0] += offset
            values = np.asarray(tree.value[:, 0, :], dtype=np.float64)
            if values.shape[1] != len(CLASS_NAMES):
                raise ValueError(
                    f"Tree has {values.shape[1]} classes; expected {len(CLASS_NAMES)}"
                )
            left_parts.append(left)
            right_parts.append(right)
            feature_parts.append(
                np.asarray(tree.feature, dtype=np.int64).copy()
            )
            threshold_parts.append(
                np.asarray(tree.threshold, dtype=np.float64).copy()
            )
            value_parts.append(values)
            offset += tree.node_count
        self._forest_roots = np.asarray(roots, dtype=np.int64)
        self._forest_left = np.concatenate(left_parts)
        self._forest_right = np.concatenate(right_parts)
        self._forest_feature = np.concatenate(feature_parts)
        self._forest_threshold = np.concatenate(threshold_parts)
        self._forest_value = np.concatenate(value_parts, axis=0)

    def _vectorized_single_probability(
        self, features: np.ndarray
    ) -> np.ndarray:
        current = self._forest_roots.copy()
        sample = np.asarray(features[0], dtype=np.float32)
        while True:
            left = self._forest_left[current]
            active = left >= 0
            if not np.any(active):
                break
            active_nodes = current[active]
            feature_indices = self._forest_feature[active_nodes]
            go_left = (
                sample[feature_indices]
                <= self._forest_threshold[active_nodes]
            )
            next_nodes = np.where(
                go_left,
                left[active],
                self._forest_right[active_nodes],
            )
            current[active] = next_nodes
        leaf_values = self._forest_value[current]
        leaf_values = leaf_values / np.maximum(
            leaf_values.sum(axis=1, keepdims=True), 1e-12
        )
        return leaf_values.mean(axis=0, keepdims=True)

    def _predict_proba(self, features: np.ndarray) -> np.ndarray:
        if features.shape[0] == 1 and self._forest_roots.size:
            return self._vectorized_single_probability(features)
        return np.asarray(
            self.model.predict_proba(features), dtype=np.float64
        )

    def predict(
        self,
        basic_features: np.ndarray,
        emg: np.ndarray,
        imu: np.ndarray,
    ) -> Prediction:
        t0 = time.perf_counter_ns()
        selected = extract_selected_feature(
            self.feature_set,
            basic_features,
            emg,
            imu,
        )
        t1 = time.perf_counter_ns()
        raw_model_probability = self._predict_proba(selected)[0]
        classes = np.asarray(self.model.classes_, dtype=np.int64)
        raw_probability = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        raw_probability[classes] = raw_model_probability
        raw_probability /= max(float(raw_probability.sum()), 1e-12)
        self.probability_history.append(raw_probability)
        smoothed_probability = np.mean(
            np.stack(tuple(self.probability_history), axis=0), axis=0
        )
        t2 = time.perf_counter_ns()

        raw_label = int(np.argmax(raw_probability))
        label = int(np.argmax(smoothed_probability))
        return Prediction(
            label=label,
            raw_label=raw_label,
            confidence=float(smoothed_probability[label]),
            raw_confidence=float(raw_probability[raw_label]),
            probabilities=smoothed_probability.tolist(),
            raw_probabilities=raw_probability.tolist(),
            history_count=len(self.probability_history),
            feature_ms=(t1 - t0) / 1e6,
            model_ms=(t2 - t1) / 1e6,
        )

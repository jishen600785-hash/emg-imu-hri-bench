from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np

from .constants import INPUT_CHANNELS
from .feature_extraction import extract_feature_vector


@dataclass(frozen=True)
class Prediction:
    label: int
    confidence: float
    probabilities: list[float]
    feature_ms: float
    model_ms: float

    @property
    def inference_ms(self) -> float:
        return self.feature_ms + self.model_ms


class PersonalizedLimbModel:
    """Read-only adapter for a fitted personalized Limb artifact.

    This class never calls fit, partial_fit or fit_transform.  The fitted
    StandardScaler and classifier embedded in the joblib pipeline are reused.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        payload: dict[str, Any] = joblib.load(self.model_path)
        if not isinstance(payload, dict) or "pipeline" not in payload or "metadata" not in payload:
            raise ValueError("Expected a deployment dict containing 'pipeline' and 'metadata'")

        self.pipeline = payload["pipeline"]
        self.metadata = dict(payload["metadata"])
        digest = hashlib.sha256()
        with self.model_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.sha256 = digest.hexdigest()
        self.channels = np.asarray(self.metadata["channel_indices_zero_based"], dtype=np.int64)
        self.window_samples = int(self.metadata["window_length_samples"])
        self.step_samples = int(self.metadata["step_samples"])
        self.feature_dim = int(self.metadata["feature_dim"])
        self.subject_id = int(self.metadata["subject_id"])
        self.class_names = {int(k): str(v) for k, v in self.metadata["class_names"].items()}

        if self.channels.ndim != 1 or len(self.channels) == 0:
            raise ValueError("Model metadata must contain a non-empty 1-D channel index list")
        if len(np.unique(self.channels)) != len(self.channels):
            raise ValueError("Model metadata contains duplicate channel indices")
        if int(self.channels.min()) < 0 or int(self.channels.max()) >= INPUT_CHANNELS:
            raise ValueError(
                f"Model channel indices must be within 0..{INPUT_CHANNELS - 1}, got "
                f"{self.channels.tolist()}"
            )
        expected_dim = 33 * len(self.channels)
        if expected_dim != self.feature_dim:
            raise ValueError(f"Model metadata mismatch: 33 x {len(self.channels)} != {self.feature_dim}")

    def _select_channels(self, raw_window: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw_window, dtype=np.float32)
        if raw.ndim != 2:
            raise ValueError(f"Expected samples x channels, got {raw.shape}")
        if raw.shape[0] < self.window_samples:
            raise ValueError(f"Need {self.window_samples} samples, got {raw.shape[0]}")
        raw = raw[-self.window_samples :, :]

        # The ROS stream always carries the 42 raw sensor channels.  A model
        # may also select all 42 channels in a non-identity order (EMG first,
        # then IMU), so equality with len(self.channels) does not mean the
        # input is already selected.  Apply the saved training order first.
        if raw.shape[1] == INPUT_CHANNELS:
            return raw[:, self.channels]

        # Retain support for callers that intentionally pass a preselected
        # window (for example, a six-channel EMG-only array).
        if raw.shape[1] == len(self.channels):
            return raw
        if raw.shape[1] <= int(self.channels.max()):
            raise ValueError(
                f"Model requests channel {int(self.channels.max())}, but input has {raw.shape[1]} channels"
            )
        return raw[:, self.channels]

    def predict(self, raw_window: np.ndarray) -> Prediction:
        selected = self._select_channels(raw_window)

        t0 = time.perf_counter_ns()
        features = extract_feature_vector(selected)
        t1 = time.perf_counter_ns()
        if features.size != self.feature_dim:
            raise ValueError(f"Feature dimension {features.size} != trained dimension {self.feature_dim}")

        raw_prob = np.asarray(self.pipeline.predict_proba(features[None, :])[0], dtype=np.float64)
        t2 = time.perf_counter_ns()
        classes = np.asarray(self.pipeline.classes_, dtype=np.int64)
        probabilities = np.zeros(5, dtype=np.float64)
        probabilities[classes] = raw_prob
        probabilities /= max(float(probabilities.sum()), 1e-12)
        label = int(np.argmax(probabilities))

        return Prediction(
            label=label,
            confidence=float(probabilities[label]),
            probabilities=probabilities.tolist(),
            feature_ms=(t1 - t0) / 1e6,
            model_ms=(t2 - t1) / 1e6,
        )

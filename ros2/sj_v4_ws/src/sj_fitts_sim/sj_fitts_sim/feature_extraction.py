from __future__ import annotations

import numpy as np

from .constants import (
    EMG_CHANNELS,
    EMG_POINTS,
    IMU_CHANNELS,
    IMU_POINTS,
    INPUT_FEATURES,
)


EPSILON = 1e-10


def _time_features(x: np.ndarray) -> np.ndarray:
    """Exact deployment copy of the optimized SJ training features."""
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
    basic_features: np.ndarray,
    emg: np.ndarray,
    imu: np.ndarray,
) -> dict[str, np.ndarray]:
    basic = np.asarray(basic_features, dtype=np.float32).reshape(-1, INPUT_FEATURES)
    emg_array = np.asarray(emg, dtype=np.float32).reshape(
        -1, EMG_CHANNELS, EMG_POINTS
    )
    imu_array = np.asarray(imu, dtype=np.float32).reshape(
        -1, IMU_CHANNELS, IMU_POINTS
    )

    emg_basic = basic[:, :50].reshape(-1, 5, 10).astype(np.float64)
    positive_indices = [1, 2, 3, 4, 5, 6]
    log_positive = np.log(np.abs(emg_basic[:, :, positive_indices]) + EPSILON)
    relative_log = log_positive - log_positive.mean(axis=1, keepdims=True)
    emg_shape = np.concatenate(
        [
            relative_log.reshape(basic.shape[0], -1),
            _emg_frequency_features(emg_array.astype(np.float64)).reshape(
                basic.shape[0], -1
            ),
        ],
        axis=1,
    )

    imu_by_sensor = imu_array.astype(np.float64).reshape(
        -1, 5, 6, imu_array.shape[2]
    )
    acceleration_magnitude = np.linalg.norm(
        imu_by_sensor[:, :, 0:3, :], axis=2
    )
    gyroscope_magnitude = np.linalg.norm(
        imu_by_sensor[:, :, 3:6, :], axis=2
    )
    acceleration_dynamic = (
        acceleration_magnitude
        - acceleration_magnitude.mean(axis=2, keepdims=True)
    )
    gyro_dynamic = (
        gyroscope_magnitude
        - gyroscope_magnitude.mean(axis=2, keepdims=True)
    )
    imu_invariant = np.concatenate(
        [
            _time_features(acceleration_magnitude),
            _time_features(gyroscope_magnitude),
            _time_features(acceleration_dynamic),
            _time_features(gyro_dynamic),
        ],
        axis=2,
    ).reshape(basic.shape[0], -1)

    emg_time = _time_features(emg_array.astype(np.float64)).reshape(
        basic.shape[0], -1
    )
    return {
        "emg_basic": emg_basic.reshape(basic.shape[0], -1).astype(np.float32),
        "emg_relative_frequency": emg_shape.astype(np.float32),
        "emg_time_frequency": np.concatenate(
            [emg_time, emg_shape], axis=1
        ).astype(np.float32),
        "emg_plus_imu_invariant": np.concatenate(
            [emg_time, emg_shape, imu_invariant], axis=1
        ).astype(np.float32),
    }


def extract_selected_feature(
    feature_set: str,
    basic_features: np.ndarray,
    emg: np.ndarray,
    imu: np.ndarray,
) -> np.ndarray:
    feature_sets = engineer_features(basic_features, emg, imu)
    if feature_set not in feature_sets:
        raise ValueError(
            f"Unknown feature set {feature_set!r}; available={sorted(feature_sets)}"
        )
    return feature_sets[feature_set]

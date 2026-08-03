from __future__ import annotations

import math

import numpy as np

from .constants import FS


def extract_feature_vector(window: np.ndarray) -> np.ndarray:
    """Exact deployment copy of train_limb_personalized.extract_feature_vector.

    The order is deliberately unchanged: 33 features are concatenated per
    selected channel.  In particular, MAV, RMS, VAR and WL are present.
    """

    x = np.asarray(window, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 3:
        raise ValueError(f"Expected a 2-D window with at least 3 samples, got {x.shape}")

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
        np.mean(abs_x, axis=0),                 # MAV
        rms,                                    # RMS
        var,                                    # VAR
        np.mean(x * x, axis=0),
        np.sum(abs_x, axis=0) / x.shape[0],
        np.sum(np.abs(diff), axis=0) / max(diff.shape[0], 1),  # normalized WL
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
    return np.nan_to_num(
        np.concatenate(feats).astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

import numpy as np

from sj_fitts_sim.feature_extraction import extract_selected_feature


def _inputs(seed: int = 42):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=350).astype(np.float32),
        rng.normal(size=(5, 256)).astype(np.float32),
        rng.normal(size=(30, 64)).astype(np.float32),
    )


def test_optimized_feature_dimension():
    basic, emg, imu = _inputs()
    features = extract_selected_feature(
        "emg_plus_imu_invariant", basic, emg, imu
    )
    assert features.shape == (1, 260)
    assert np.isfinite(features).all()


def test_feature_extraction_is_deterministic():
    basic, emg, imu = _inputs()
    first = extract_selected_feature(
        "emg_plus_imu_invariant", basic, emg, imu
    )
    second = extract_selected_feature(
        "emg_plus_imu_invariant", basic, emg, imu
    )
    assert np.array_equal(first, second)

import numpy as np

from limb_fitts_sim.feature_extraction import extract_feature_vector


def test_feature_dimension_is_33_per_channel():
    rng = np.random.default_rng(42)
    window = rng.normal(size=(630, 6)).astype(np.float32)
    features = extract_feature_vector(window)
    assert features.shape == (198,)
    assert np.isfinite(features).all()


def test_feature_extraction_is_deterministic():
    window = np.arange(630 * 6, dtype=np.float32).reshape(630, 6) / 1000.0
    assert np.array_equal(extract_feature_vector(window), extract_feature_vector(window))

import math

import pytest

from limb_fitts_sim.fitts_metrics import (
    condition_metrics,
    endpoint_components,
    is_fresh_movement_prediction,
    linear_regression,
    nominal_id,
    projected_amplitude,
    ring_radius,
    ring_targets,
    target_order,
    task_axis,
)


def test_replay_movement_gate_rejects_inter_trial_residue():
    rest = 4
    # Old movement is still applied while the task is in its Rest pause.
    assert not is_fresh_movement_prediction(
        autopilot=True,
        desired_label=rest,
        predicted_label=0,
        ground_truth_label=0,
        confidence=0.99,
        confidence_threshold=0.50,
        rest_label=rest,
    )
    # The new direction was requested, but the replay stream is still Rest.
    assert not is_fresh_movement_prediction(
        autopilot=True,
        desired_label=2,
        predicted_label=rest,
        ground_truth_label=rest,
        confidence=0.99,
        confidence_threshold=0.50,
        rest_label=rest,
    )
    # Only a ground-truth chunk from this trial's requested direction opens
    # the selection gate. A wrong Rest prediction is then a real model error.
    assert is_fresh_movement_prediction(
        autopilot=True,
        desired_label=2,
        predicted_label=rest,
        ground_truth_label=2,
        confidence=0.99,
        confidence_threshold=0.50,
        rest_label=rest,
    )


def test_live_movement_gate_uses_new_confident_non_rest_prediction():
    assert is_fresh_movement_prediction(
        autopilot=False,
        desired_label=4,
        predicted_label=1,
        ground_truth_label=-1,
        confidence=0.80,
        confidence_threshold=0.50,
        rest_label=4,
    )
    assert not is_fresh_movement_prediction(
        autopilot=False,
        desired_label=4,
        predicted_label=4,
        ground_truth_label=-1,
        confidence=0.99,
        confidence_threshold=0.50,
        rest_label=4,
    )


def test_target_order_is_full_near_opposite_sequence():
    order = target_order(7)
    assert order == [0, 3, 6, 2, 5, 1, 4, 0]
    assert sorted(order[:-1]) == list(range(7))


def test_ring_chords_match_configured_amplitude_and_anchor_home():
    home = (-0.3, 0.42)
    amplitude = 0.28
    targets = ring_targets(home, amplitude, 7)
    order = target_order(7)
    assert targets[0] == pytest.approx(home)
    for start, end in zip(order, order[1:]):
        assert math.dist(targets[start], targets[end]) == pytest.approx(amplitude)
    assert ring_radius(amplitude, 7) < 0.144


def test_endpoint_projection_and_effective_amplitude():
    axis = task_axis((0.0, 0.0), (1.0, 0.0))
    parallel, perpendicular = endpoint_components((1.1, -0.2), (1.0, 0.0), axis)
    assert parallel == pytest.approx(0.1)
    assert perpendicular == pytest.approx(-0.2)
    assert projected_amplitude((0.05, 0.0), (1.1, -0.2), axis) == pytest.approx(1.05)


def test_condition_metrics_use_first_selection_endpoints():
    rows = [
        {
            "selection_made": True,
            "hit": True,
            "timeout": False,
            "endpoint_axis_error_m": value,
            "effective_amplitude_m": 0.16 + value,
            "movement_time_s": 2.0,
        }
        for value in (-0.01, 0.0, 0.01)
    ]
    metrics = condition_metrics(rows)
    assert metrics["selection_count"] == 3
    assert metrics["hit_count"] == 3
    assert metrics["effective_width_m"] == pytest.approx(4.133 * 0.01)
    assert metrics["effective_amplitude_m"] == pytest.approx(0.16)
    assert metrics["throughput_bits_per_s"] > 0.0


def test_nominal_id_and_regression():
    assert nominal_id(0.10, 0.06) == pytest.approx(math.log2(0.10 / 0.06 + 1.0))
    regression = linear_regression([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert regression["intercept_s"] == pytest.approx(0.0)
    assert regression["slope_s_per_bit"] == pytest.approx(2.0)
    assert regression["r_squared"] == pytest.approx(1.0)

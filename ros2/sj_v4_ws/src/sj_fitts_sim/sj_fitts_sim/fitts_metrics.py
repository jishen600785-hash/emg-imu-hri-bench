from __future__ import annotations

from collections.abc import Iterable
import math
import statistics


def hysteretic_cardinal_direction(
    dx: float,
    dy: float,
    previous_direction: str | None,
    axis_deadband_m: float,
) -> str | None:
    """Keep one Cartesian direction until its own axis is nearly complete."""
    if axis_deadband_m <= 0.0:
        raise ValueError("axis_deadband_m must be positive")
    valid = {None, "x+", "x-", "y+", "y-"}
    if previous_direction not in valid:
        raise ValueError(f"Unsupported previous_direction: {previous_direction!r}")

    if previous_direction == "x+" and dx > axis_deadband_m:
        return previous_direction
    if previous_direction == "x-" and dx < -axis_deadband_m:
        return previous_direction
    if previous_direction == "y+" and dy > axis_deadband_m:
        return previous_direction
    if previous_direction == "y-" and dy < -axis_deadband_m:
        return previous_direction

    if abs(dx) <= axis_deadband_m and abs(dy) <= axis_deadband_m:
        return None
    if abs(dx) >= abs(dy):
        return "x+" if dx > 0.0 else "x-"
    return "y+" if dy > 0.0 else "y-"


def width_aware_stop_radius(
    minimum_radius_m: float,
    target_width_m: float,
    width_fraction: float,
    maximum_radius_m: float,
) -> float:
    """Choose a stop radius that stays safely inside the active target."""
    if minimum_radius_m <= 0.0:
        raise ValueError("minimum_radius_m must be positive")
    if target_width_m <= 0.0:
        raise ValueError("target_width_m must be positive")
    if width_fraction <= 0.0:
        raise ValueError("width_fraction must be positive")
    if maximum_radius_m < minimum_radius_m:
        raise ValueError("maximum_radius_m must be at least minimum_radius_m")
    return min(
        max(minimum_radius_m, target_width_m * width_fraction),
        maximum_radius_m,
    )


def adaptive_speed_scale(
    radial_error_m: float,
    *,
    stop_requested: bool,
    near_threshold_m: float,
    far_threshold_m: float,
    near_scale: float,
    mid_scale: float,
    far_scale: float,
) -> float:
    """Return a coarse-to-fine Cartesian speed multiplier."""
    if radial_error_m < 0.0:
        raise ValueError("radial_error_m must be non-negative")
    if not 0.0 < near_threshold_m < far_threshold_m:
        raise ValueError("speed thresholds must satisfy 0 < near < far")
    if min(near_scale, mid_scale, far_scale) <= 0.0:
        raise ValueError("movement speed scales must be positive")
    if stop_requested:
        return 0.0
    if radial_error_m > far_threshold_m:
        return far_scale
    if radial_error_m > near_threshold_m:
        return mid_scale
    return near_scale


def is_fresh_movement_prediction(
    *,
    autopilot: bool,
    desired_label: int,
    predicted_label: int,
    ground_truth_label: int,
    confidence: float,
    confidence_threshold: float,
    rest_label: int,
) -> bool:
    """Reject predictions that belong to the pause before a new trial.

    In replay mode the ground-truth stream changes immediately when the task
    requests a new gesture.  Requiring it to match the current non-Rest
    request prevents an old applied movement label followed by a pause Rest
    prediction from selecting the next target at time zero.
    """
    if autopilot:
        return (
            desired_label != rest_label
            and ground_truth_label == desired_label
        )
    return (
        predicted_label != rest_label
        and confidence >= confidence_threshold
    )


def nominal_id(distance_m: float, width_m: float) -> float:
    """Return Shannon's formulation of the nominal index of difficulty."""
    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")
    if width_m <= 0.0:
        raise ValueError("width_m must be positive")
    return math.log2(distance_m / width_m + 1.0)


def target_order(target_count: int = 7) -> list[int]:
    """Return one serial, near-opposite multidirectional target sequence."""
    if target_count < 3 or target_count % 2 == 0:
        raise ValueError("target_count must be an odd integer of at least 3")
    step = target_count // 2
    return [(index * step) % target_count for index in range(target_count + 1)]


def ring_radius(distance_m: float, target_count: int = 7) -> float:
    """Radius whose near-opposite chord has length ``distance_m``."""
    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")
    if target_count < 3 or target_count % 2 == 0:
        raise ValueError("target_count must be an odd integer of at least 3")
    step = target_count // 2
    return distance_m / (2.0 * math.sin(math.pi * step / target_count))


def ring_targets(
    home_xy: tuple[float, float],
    distance_m: float,
    target_count: int = 7,
) -> list[tuple[float, float]]:
    """Build a ring with target 0 anchored at ``home_xy``."""
    radius = ring_radius(distance_m, target_count)
    center_x = home_xy[0]
    center_y = home_xy[1] - radius
    targets: list[tuple[float, float]] = []
    for index in range(target_count):
        angle = math.pi / 2.0 - 2.0 * math.pi * index / target_count
        targets.append(
            (
                center_x + radius * math.cos(angle),
                center_y + radius * math.sin(angle),
            )
        )
    return targets


def task_axis(
    start_target_xy: tuple[float, float],
    target_xy: tuple[float, float],
) -> tuple[float, float]:
    dx = target_xy[0] - start_target_xy[0]
    dy = target_xy[1] - start_target_xy[1]
    length = math.hypot(dx, dy)
    if length <= 0.0:
        raise ValueError("start target and target must be different")
    return dx / length, dy / length


def endpoint_components(
    endpoint_xy: tuple[float, float],
    target_xy: tuple[float, float],
    axis_xy: tuple[float, float],
) -> tuple[float, float]:
    """Return endpoint error parallel and perpendicular to the task axis."""
    error_x = endpoint_xy[0] - target_xy[0]
    error_y = endpoint_xy[1] - target_xy[1]
    parallel = error_x * axis_xy[0] + error_y * axis_xy[1]
    perpendicular = -error_x * axis_xy[1] + error_y * axis_xy[0]
    return parallel, perpendicular


def projected_amplitude(
    actual_start_xy: tuple[float, float],
    endpoint_xy: tuple[float, float],
    axis_xy: tuple[float, float],
) -> float:
    movement_x = endpoint_xy[0] - actual_start_xy[0]
    movement_y = endpoint_xy[1] - actual_start_xy[1]
    return movement_x * axis_xy[0] + movement_y * axis_xy[1]


def condition_metrics(rows: Iterable[dict]) -> dict:
    """Compute condition-level effective Fitts metrics from first selections."""
    selected = [row for row in rows if bool(row.get("selection_made", False))]
    timed_out = [row for row in rows if bool(row.get("timeout", False))]
    hits = [row for row in selected if bool(row.get("hit", False))]
    misses = [row for row in selected if not bool(row.get("hit", False))]
    endpoint_errors = [float(row["endpoint_axis_error_m"]) for row in selected]
    amplitudes = [float(row["effective_amplitude_m"]) for row in selected]
    movement_times = [float(row["movement_time_s"]) for row in selected]

    sd_x = statistics.stdev(endpoint_errors) if len(endpoint_errors) > 1 else float("nan")
    effective_width = 4.133 * sd_x if math.isfinite(sd_x) else float("nan")
    effective_amplitude = (
        statistics.fmean(amplitudes) if amplitudes else float("nan")
    )
    mean_movement_time = (
        statistics.fmean(movement_times) if movement_times else float("nan")
    )
    if (
        math.isfinite(effective_width)
        and effective_width > 0.0
        and math.isfinite(effective_amplitude)
        and effective_amplitude > 0.0
    ):
        effective_id = math.log2(effective_amplitude / effective_width + 1.0)
    else:
        effective_id = float("nan")
    throughput = (
        effective_id / mean_movement_time
        if math.isfinite(effective_id) and mean_movement_time > 0.0
        else float("nan")
    )
    total = len(selected) + len(timed_out)
    return {
        "trial_count": total,
        "selection_count": len(selected),
        "hit_count": len(hits),
        "miss_count": len(misses),
        "timeout_count": len(timed_out),
        "success_rate": len(hits) / total if total else 0.0,
        "error_rate": (
            (len(misses) + len(timed_out)) / total if total else 0.0
        ),
        "selection_error_rate": (
            len(misses) / len(selected) if selected else 0.0
        ),
        "total_failure_rate": (
            (len(misses) + len(timed_out)) / total if total else 0.0
        ),
        "mean_movement_time_s": mean_movement_time,
        "endpoint_axis_sd_m": sd_x,
        "effective_width_m": effective_width,
        "effective_amplitude_m": effective_amplitude,
        "effective_id_bits": effective_id,
        "throughput_bits_per_s": throughput,
    }


def linear_regression(xs: Iterable[float], ys: Iterable[float]) -> dict:
    x_values = [float(value) for value in xs]
    y_values = [float(value) for value in ys]
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return {"intercept_s": float("nan"), "slope_s_per_bit": float("nan"), "r_squared": float("nan")}
    mean_x = statistics.fmean(x_values)
    mean_y = statistics.fmean(y_values)
    denominator = sum((value - mean_x) ** 2 for value in x_values)
    if denominator <= 0.0:
        return {"intercept_s": float("nan"), "slope_s_per_bit": float("nan"), "r_squared": float("nan")}
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    intercept = mean_y - slope * mean_x
    residual = sum(
        (y_value - (intercept + slope * x_value)) ** 2
        for x_value, y_value in zip(x_values, y_values)
    )
    total = sum((value - mean_y) ** 2 for value in y_values)
    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
    return {
        "intercept_s": intercept,
        "slope_s_per_bit": slope,
        "r_squared": r_squared,
    }

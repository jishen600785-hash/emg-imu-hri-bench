from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

from geometry_msgs.msg import Point, Pose
import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from std_msgs.msg import Float64, Int32, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .constants import HAND_OPEN, LATERAL_GRIP, PINCH_GRIP, POWER_GRIP, REST
from .fitts_metrics import (
    adaptive_speed_scale,
    condition_metrics,
    endpoint_components,
    hysteretic_cardinal_direction,
    is_fresh_movement_prediction,
    linear_regression,
    nominal_id,
    projected_amplitude,
    ring_radius,
    ring_targets,
    target_order,
    task_axis,
    width_aware_stop_radius,
)


TASK_PROTOCOL = "short_multidirectional_EMG_robot_v4_adaptive_control"
EVALUATION_PROTOCOL = (
    "model_and_history_selected_by_leave_one_static_position_out; "
    "refit_on_all_three_static_positions; dynamic_condition_held_out_test"
)
EVALUATION_SPLIT_ID = "sj_static_to_dynamic_v1"
CONTROL_POLICY = "width_aware_stop_and_coarse_to_fine_speed_v1"


@dataclass(frozen=True)
class TrialSpec:
    distance_m: float
    width_m: float
    block_index: int
    sequence_number: int
    condition_order: int
    sequence_trial_index: int
    from_target_index: int
    to_target_index: int
    practice: bool = False

    @property
    def condition_id(self) -> str:
        prefix = "PRACTICE" if self.practice else "FORMAL"
        return (
            f"{prefix}_A{round(self.distance_m * 1000):03d}"
            f"_W{round(self.width_m * 1000):03d}"
        )


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _p95(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return float(ordered[min(math.ceil(0.95 * len(ordered)) - 1, len(ordered) - 1)])


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


class FittsTaskNode(Node):
    """Runs an adapted serial multidirectional Fitts task in a fixed XY plane."""

    CSV_FIELDS = [
        "trial_index",
        "subject_id",
        "task_protocol",
        "input_mode",
        "evaluation_mode",
        "fold_index",
        "held_out_condition",
        "block_index",
        "sequence_number",
        "condition_order",
        "condition_id",
        "sequence_trial_index",
        "target_count",
        "from_target_index",
        "to_target_index",
        "model_sha256",
        "dataset_sha256",
        "split_manifest_sha256",
        "provenance_valid",
        "provenance_errors",
        "input_segment_ids",
        "input_source_files",
        "input_sample_ranges",
        "stream_instance_ids",
        "replay_wrap_count",
        "distance_config_m",
        "distance_center_m",
        "distance_actual_m",
        "target_width_m",
        "effective_stop_radius_m",
        "index_of_difficulty_bits",
        "timing_clock",
        "selection_made",
        "hit",
        "miss",
        "timeout",
        "success",
        "movement_time_s",
        "confirmation_time_s",
        "selection_confirmation_delay_s",
        "wall_movement_time_s",
        "observed_real_time_factor",
        "throughput_bits_per_s",
        "classification_errors",
        "direction_errors",
        "prediction_count",
        "classification_accuracy",
        "mean_inference_ms",
        "p95_inference_ms",
        "mean_control_latency_ms",
        "p95_control_latency_ms",
        "path_length_m",
        "direction_switches",
        "endpoint_radial_error_m",
        "endpoint_axis_error_m",
        "endpoint_orthogonal_error_m",
        "effective_amplitude_m",
        "start_target_x",
        "start_target_y",
        "start_x",
        "start_y",
        "target_x",
        "target_y",
        "first_selection_x",
        "first_selection_y",
        "confirmed_end_x",
        "confirmed_end_y",
        "end_x",
        "end_y",
        "timestamp",
    ]

    def __init__(self) -> None:
        super().__init__("fitts_task")
        self.declare_parameter("subject_id", 1)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("distances_m", [0.08, 0.18, 0.28])
        self.declare_parameter("widths_m", [0.025, 0.060])
        self.declare_parameter("target_count", 7)
        self.declare_parameter("sequences_per_condition", 1)
        self.declare_parameter("practice_enabled", True)
        self.declare_parameter("practice_distance_m", 0.18)
        self.declare_parameter("practice_width_m", 0.06)
        self.declare_parameter("randomize_conditions", True)
        self.declare_parameter("random_seed", 42)
        self.declare_parameter("trial_timeout_sec", 35.0)
        self.declare_parameter("homing_timeout_sec", 35.0)
        self.declare_parameter("rest_confirmation_predictions", 2)
        self.declare_parameter("movement_confidence_threshold", 0.20)
        self.declare_parameter("selection_confidence_threshold", 0.30)
        self.declare_parameter("autopilot_stop_radius_m", 0.0115)
        self.declare_parameter("width_stop_fraction", 0.45)
        self.declare_parameter("max_stop_radius_m", 0.027)
        self.declare_parameter("axis_hysteresis_fraction", 0.50)
        self.declare_parameter("speed_near_threshold_m", 0.05)
        self.declare_parameter("speed_far_threshold_m", 0.12)
        self.declare_parameter("speed_scale_near", 0.70)
        self.declare_parameter("speed_scale_mid", 1.00)
        self.declare_parameter("speed_scale_far", 1.35)
        self.declare_parameter("inter_trial_pause_sec", 0.50)
        self.declare_parameter("results_dir", "./results")
        self.declare_parameter("base_world_z", 0.72)
        self.declare_parameter("workspace_radius_m", 0.34)
        self.declare_parameter("autopilot", True)
        self.declare_parameter("evaluation_mode", False)
        self.declare_parameter("fold_index", 0)
        self.declare_parameter("held_out_condition", "")
        self.declare_parameter("model_path", "")

        self.subject_id = int(self.get_parameter("subject_id").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.distances = [float(value) for value in self.get_parameter("distances_m").value]
        self.widths = [float(value) for value in self.get_parameter("widths_m").value]
        self.target_count = int(self.get_parameter("target_count").value)
        self.sequences_per_condition = int(
            self.get_parameter("sequences_per_condition").value
        )
        self.practice_enabled = bool(self.get_parameter("practice_enabled").value)
        self.practice_distance = float(self.get_parameter("practice_distance_m").value)
        self.practice_width = float(self.get_parameter("practice_width_m").value)
        self.randomize_conditions = bool(
            self.get_parameter("randomize_conditions").value
        )
        self.random_seed = int(self.get_parameter("random_seed").value)
        self.timeout_sec = float(self.get_parameter("trial_timeout_sec").value)
        self.homing_timeout_sec = float(self.get_parameter("homing_timeout_sec").value)
        self.rest_confirmation_predictions = int(
            self.get_parameter("rest_confirmation_predictions").value
        )
        self.movement_confidence_threshold = float(
            self.get_parameter("movement_confidence_threshold").value
        )
        self.selection_confidence_threshold = float(
            self.get_parameter("selection_confidence_threshold").value
        )
        self.autopilot_stop_radius = float(
            self.get_parameter("autopilot_stop_radius_m").value
        )
        self.width_stop_fraction = float(
            self.get_parameter("width_stop_fraction").value
        )
        self.max_stop_radius = float(
            self.get_parameter("max_stop_radius_m").value
        )
        self.axis_hysteresis_fraction = float(
            self.get_parameter("axis_hysteresis_fraction").value
        )
        self.speed_near_threshold = float(
            self.get_parameter("speed_near_threshold_m").value
        )
        self.speed_far_threshold = float(
            self.get_parameter("speed_far_threshold_m").value
        )
        self.speed_scale_near = float(
            self.get_parameter("speed_scale_near").value
        )
        self.speed_scale_mid = float(
            self.get_parameter("speed_scale_mid").value
        )
        self.speed_scale_far = float(
            self.get_parameter("speed_scale_far").value
        )
        self.inter_trial_pause_sec = float(
            self.get_parameter("inter_trial_pause_sec").value
        )
        self.base_world_z = float(self.get_parameter("base_world_z").value)
        self.workspace_radius = float(self.get_parameter("workspace_radius_m").value)
        self.autopilot = bool(self.get_parameter("autopilot").value)
        self.evaluation_mode = bool(self.get_parameter("evaluation_mode").value)
        self.fold_index = int(self.get_parameter("fold_index").value)
        self.held_out_condition = str(self.get_parameter("held_out_condition").value)
        self.model_path = Path(str(self.get_parameter("model_path").value))

        self._validate_configuration()
        self.order = target_order(self.target_count)
        self.task_specs = self._build_task_specs()
        self.total_formal_trials = sum(not spec.practice for spec in self.task_specs)
        self.total_practice_trials = sum(spec.practice for spec in self.task_specs)

        self.model_sha256 = ""
        if self.evaluation_mode:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Held-out model not found: {self.model_path}")
            digest = hashlib.sha256()
            with self.model_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self.model_sha256 = digest.hexdigest()

        results_dir = Path(str(self.get_parameter("results_dir").value))
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_tag = (
            f"_fold{self.fold_index:02d}_{self.held_out_condition}"
            if self.evaluation_mode
            else "_demo"
        )
        stem = f"fitts_trials_subject{self.subject_id:02d}{fold_tag}_{stamp}"
        self.csv_path = results_dir / f"{stem}.csv"
        self.condition_csv_path = results_dir / f"{stem}_conditions.csv"
        self.summary_path = results_dir / stem.replace("trials", "summary")
        self.summary_path = self.summary_path.with_suffix(".json")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.marker_pub = self.create_publisher(MarkerArray, "/fitts/markers", 10)
        self.desired_pub = self.create_publisher(
            Int32, "/fitts/demo_requested_label", 10
        )
        self.speed_scale_pub = self.create_publisher(
            Float64, "/fitts/speed_scale", 10
        )
        self.status_pub = self.create_publisher(String, "/fitts/status", 10)
        self.summary_pub = self.create_publisher(String, "/fitts/summary", 10)
        self.create_subscription(String, "/sj/prediction", self._prediction_cb, 20)
        self.create_subscription(Int32, "/sj/applied_label", self._applied_cb, 20)
        self.create_subscription(String, "/fitts/control_event", self._control_cb, 20)
        self.create_subscription(
            String, "/sj/replay_provenance", self._provenance_cb, 50
        )

        self.pose_client = self.create_client(
            SetEntityPose, "/world/fitts_world/set_pose"
        )
        self.target_models = {
            0.025: "fitts_target_w025",
            0.040: "fitts_target_w040",
            0.060: "fitts_target_w060",
        }
        self.previous_models = {
            0.025: "fitts_previous_w025",
            0.040: "fitts_previous_w040",
            0.060: "fitts_previous_w060",
        }

        self.home_xy: tuple[float, float] | None = None
        self.plane_z = 0.0
        self.ring_points: list[tuple[float, float]] = []
        self.task_cursor = 0
        self.formal_trial_index = 0
        self.practice_completed = 0
        self.phase = "initializing"
        self.active_spec: TrialSpec | None = None
        self.pending_spec: TrialSpec | None = None
        self.current_target_index = 0
        self.previous_target_index: int | None = None
        self.start_target_xy = (0.0, 0.0)
        self.start_xy = (0.0, 0.0)
        self.target_xy = (0.0, 0.0)
        self.actual_distance = 0.0
        self.id_bits = 0.0
        self.phase_start_ns = 0
        self.trial_start_ns = 0
        self.trial_start_wall_ns = 0
        self.next_phase_at_ns = 0
        self.applied_label = REST
        self.desired_label = REST
        self.autopilot_rest_latched = False
        self.movement_started = False
        self.last_movement_label: int | None = None
        self.direction_switches = 0
        self.rest_prediction_streak = 0
        self.first_rest_ns = 0
        self.first_rest_pose: tuple[float, float, float] | None = None
        self.confirmation_ns = 0
        self.confirmation_pose: tuple[float, float, float] | None = None
        self.selection_ready = False
        self.path_length = 0.0
        self.last_path_xy: tuple[float, float] | None = None
        self.classification_errors = 0
        self.direction_errors = 0
        self.prediction_count = 0
        self.inference_latencies: list[float] = []
        self.control_latencies: list[float] = []
        self.seen_control_ids: set[int] = set()
        self.dataset_sha256 = ""
        self.split_manifest_sha256 = ""
        self.provenance_errors = 0
        self.trial_segment_ids: set[str] = set()
        self.trial_source_files: set[str] = set()
        self.trial_source_ranges: dict[str, list[int]] = {}
        self.trial_stream_ids: set[int] = set()
        self.trial_replay_wrap_count = 0
        self.rows: list[dict] = []
        self.finished = False
        self.last_trial_outcome = ""

        self.create_timer(1.0 / 30.0, self._timer_cb)
        self.get_logger().info(
            f"Prepared {self.total_formal_trials} measured movements and "
            f"{self.total_practice_trials} practice movements; "
            f"input={'held-out replay benchmark' if self.autopilot else 'live EMG'}; "
            f"output={self.csv_path}"
        )

    def _validate_configuration(self) -> None:
        if self.target_count < 3 or self.target_count % 2 == 0:
            raise ValueError("target_count must be an odd integer of at least 3")
        if not self.distances or any(value <= 0.0 for value in self.distances):
            raise ValueError("distances_m must contain positive values")
        if not self.widths or any(value <= 0.0 for value in self.widths):
            raise ValueError("widths_m must contain positive values")
        if self.sequences_per_condition < 1:
            raise ValueError("sequences_per_condition must be at least 1")
        if self.rest_confirmation_predictions < 1:
            raise ValueError("rest_confirmation_predictions must be at least 1")
        if not 0.0 <= self.movement_confidence_threshold <= 1.0:
            raise ValueError(
                "movement_confidence_threshold must be between 0 and 1"
            )
        if not 0.0 <= self.selection_confidence_threshold <= 1.0:
            raise ValueError(
                "selection_confidence_threshold must be between 0 and 1"
            )
        width_aware_stop_radius(
            self.autopilot_stop_radius,
            min(self.widths),
            self.width_stop_fraction,
            self.max_stop_radius,
        )
        if not 0.0 < self.axis_hysteresis_fraction <= 1.0 / math.sqrt(2.0):
            raise ValueError(
                "axis_hysteresis_fraction must be in (0, 1/sqrt(2)]"
            )
        adaptive_speed_scale(
            0.0,
            stop_requested=False,
            near_threshold_m=self.speed_near_threshold,
            far_threshold_m=self.speed_far_threshold,
            near_scale=self.speed_scale_near,
            mid_scale=self.speed_scale_mid,
            far_scale=self.speed_scale_far,
        )
        largest_extent = max(
            2.0 * ring_radius(distance, self.target_count) + max(self.widths) / 2.0
            for distance in self.distances
        )
        if largest_extent > self.workspace_radius:
            raise ValueError(
                f"Configured ring extent {largest_extent:.3f} m exceeds "
                f"workspace_radius_m={self.workspace_radius:.3f}"
            )

    def _build_task_specs(self) -> list[TrialSpec]:
        specs: list[TrialSpec] = []
        sequence_number = 0
        pairs = list(zip(self.order, self.order[1:]))
        if self.practice_enabled:
            for trial_index, (start, end) in enumerate(pairs, start=1):
                specs.append(
                    TrialSpec(
                        distance_m=self.practice_distance,
                        width_m=self.practice_width,
                        block_index=0,
                        sequence_number=0,
                        condition_order=0,
                        sequence_trial_index=trial_index,
                        from_target_index=start,
                        to_target_index=end,
                        practice=True,
                    )
                )
        conditions = [
            (distance, width)
            for distance in self.distances
            for width in self.widths
        ]
        for block_index in range(1, self.sequences_per_condition + 1):
            ordered_conditions = list(conditions)
            if self.randomize_conditions:
                random.Random(self.random_seed + block_index - 1).shuffle(
                    ordered_conditions
                )
            for condition_order, (distance, width) in enumerate(
                ordered_conditions, start=1
            ):
                sequence_number += 1
                for trial_index, (start, end) in enumerate(pairs, start=1):
                    specs.append(
                        TrialSpec(
                            distance_m=distance,
                            width_m=width,
                            block_index=block_index,
                            sequence_number=sequence_number,
                            condition_order=condition_order,
                            sequence_trial_index=trial_index,
                            from_target_index=start,
                            to_target_index=end,
                        )
                    )
        return specs

    def _tool_pose(self) -> tuple[float, float, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, rclpy.time.Time()
            )
            point = transform.transform.translation
            return float(point.x), float(point.y), float(point.z)
        except TransformException:
            return None

    def _prediction_cb(self, msg: String) -> None:
        if self.phase not in {"homing", "trial"}:
            return
        try:
            data = json.loads(msg.data)
            label = int(data["label"])
            confidence = float(data.get("confidence", 1.0))
            if self.phase == "trial":
                truth = int(data.get("ground_truth", -1))
                self.prediction_count += 1
                self.inference_latencies.append(float(data["inference_ms"]))
                if truth >= 0 and label != truth:
                    self.classification_errors += 1
                if self.autopilot and label != self.desired_label:
                    self.direction_errors += 1
                if not self.movement_started:
                    self.movement_started = is_fresh_movement_prediction(
                        autopilot=self.autopilot,
                        desired_label=self.desired_label,
                        predicted_label=label,
                        ground_truth_label=truth,
                        confidence=confidence,
                        confidence_threshold=self.movement_confidence_threshold,
                        rest_label=REST,
                    )

            eligible = self.phase == "homing" or self.movement_started
            if (
                eligible
                and label == REST
                and confidence >= self.selection_confidence_threshold
            ):
                if self.rest_prediction_streak == 0:
                    pose = self._tool_pose()
                    if pose is None:
                        return
                    self.first_rest_ns = self.get_clock().now().nanoseconds
                    self.first_rest_pose = pose
                self.rest_prediction_streak += 1
                if (
                    self.rest_prediction_streak
                    >= self.rest_confirmation_predictions
                    and not self.selection_ready
                ):
                    self.confirmation_ns = self.get_clock().now().nanoseconds
                    self.confirmation_pose = self._tool_pose() or self.first_rest_pose
                    self.selection_ready = True
            else:
                self._reset_rest_detection()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

    def _applied_cb(self, msg: Int32) -> None:
        new_label = int(msg.data)
        self.applied_label = new_label
        if (
            self.phase != "trial"
            or not self.movement_started
            or new_label == REST
        ):
            return
        if self.last_movement_label is None:
            self.last_movement_label = new_label
        elif new_label != self.last_movement_label:
            self.direction_switches += 1
            self.last_movement_label = new_label

    def _control_cb(self, msg: String) -> None:
        if self.phase != "trial":
            return
        try:
            data = json.loads(msg.data)
            prediction_id = int(data.get("prediction_id", 0))
            latency = data.get("control_latency_ms")
            if (
                prediction_id
                and prediction_id not in self.seen_control_ids
                and latency is not None
            ):
                self.seen_control_ids.add(prediction_id)
                self.control_latencies.append(float(latency))
        except (TypeError, ValueError, json.JSONDecodeError):
            return

    def _provenance_cb(self, msg: String) -> None:
        if self.phase != "trial":
            return
        try:
            data = json.loads(msg.data)
            if self.evaluation_mode and self.autopilot:
                valid = (
                    bool(data.get("evaluation_mode"))
                    and int(data.get("subject_id", -1)) == self.subject_id
                    and int(data.get("fold_index", -1)) == self.fold_index
                    and str(data.get("condition", "")) == self.held_out_condition
                    and str(data.get("model_sha256", "")) == self.model_sha256
                )
                if not valid:
                    self.provenance_errors += 1
            segment_id = str(data["segment_id"])
            source_file = str(data["source_file"])
            start_index = int(data["source_start_index"])
            end_index = int(data["source_end_index"])
            self.trial_segment_ids.add(segment_id)
            self.trial_source_files.add(source_file)
            current = self.trial_source_ranges.setdefault(
                segment_id, [start_index, end_index]
            )
            current[0] = min(current[0], start_index)
            current[1] = max(current[1], end_index)
            self.trial_stream_ids.add(int(data.get("stream_instance_id", 0)))
            self.trial_replay_wrap_count = max(
                self.trial_replay_wrap_count, int(data.get("loop_count", 0))
            )
            for attribute, key in (
                ("dataset_sha256", "dataset_sha256"),
                ("split_manifest_sha256", "split_manifest_sha256"),
            ):
                incoming_hash = str(data.get(key, ""))
                current_hash = str(getattr(self, attribute))
                if not incoming_hash:
                    self.provenance_errors += 1
                elif not current_hash:
                    setattr(self, attribute, incoming_hash)
                elif incoming_hash != current_hash:
                    self.provenance_errors += 1
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.provenance_errors += 1

    def _timer_cb(self) -> None:
        pose = self._tool_pose()
        if pose is None:
            return
        now_ns = self.get_clock().now().nanoseconds
        if self.home_xy is None:
            self.home_xy = (pose[0], pose[1])
            self.plane_z = pose[2]
            self.phase = "inter_trial"
            self.next_phase_at_ns = now_ns + int(1.0e9)
            self.get_logger().info(
                f"Home target anchored at ({pose[0]:.3f}, {pose[1]:.3f}, "
                f"{pose[2]:.3f}); fixed XY task plane ready"
            )

        if self.finished:
            self._publish_desired(REST)
            self._publish_speed_scale(0.0)
            self._publish_markers(pose)
            return

        if self.phase in {"inter_trial", "waiting_trial"}:
            self._publish_desired(REST)
            self._publish_speed_scale(0.0 if self.autopilot else 1.0)
            if now_ns >= self.next_phase_at_ns:
                if self.phase == "waiting_trial":
                    assert self.pending_spec is not None
                    self._start_trial(self.pending_spec, pose)
                else:
                    self._prepare_next_spec(pose)
            self._publish_markers(pose)
            return

        dx = self.target_xy[0] - pose[0]
        dy = self.target_xy[1] - pose[1]
        radial_error = math.hypot(dx, dy)
        if self.autopilot:
            desired = self._desired_from_error(dx, dy, radial_error)
        else:
            desired = REST
        self._publish_desired(desired)
        if self.autopilot:
            speed_scale = adaptive_speed_scale(
                radial_error,
                stop_requested=desired == REST,
                near_threshold_m=self.speed_near_threshold,
                far_threshold_m=self.speed_far_threshold,
                near_scale=self.speed_scale_near,
                mid_scale=self.speed_scale_mid,
                far_scale=self.speed_scale_far,
            )
        else:
            speed_scale = 1.0
        self._publish_speed_scale(speed_scale)

        if self.phase == "homing":
            if self.selection_ready:
                if radial_error <= self.pending_spec.width_m / 2.0:
                    self._complete_homing()
                else:
                    self.get_logger().warning(
                        "Home selection was outside target; keep moving to target 0"
                    )
                    self.autopilot_rest_latched = False
                    self._reset_rest_detection()
            elif (now_ns - self.phase_start_ns) / 1e9 >= self.homing_timeout_sec:
                self.get_logger().warning("Homing timed out; restarting homing timer")
                self.phase_start_ns = now_ns
            self._publish_markers(pose)
            return

        if self.phase == "trial":
            if self.last_path_xy is not None:
                self.path_length += math.dist(self.last_path_xy, (pose[0], pose[1]))
            self.last_path_xy = (pose[0], pose[1])
            if self.selection_ready and self.first_rest_pose is not None:
                self._complete_trial(
                    selection_made=True,
                    endpoint=self.first_rest_pose,
                    confirmed_pose=self.confirmation_pose or pose,
                    completion_ns=self.confirmation_ns or now_ns,
                )
            elif (now_ns - self.trial_start_ns) / 1e9 >= self.timeout_sec:
                self._complete_trial(
                    selection_made=False,
                    endpoint=pose,
                    confirmed_pose=pose,
                    completion_ns=now_ns,
                )
            self._publish_markers(pose)

    def _prepare_next_spec(self, pose: tuple[float, float, float]) -> None:
        if self.task_cursor >= len(self.task_specs):
            self._finish_experiment()
            return
        spec = self.task_specs[self.task_cursor]
        self.pending_spec = spec
        if spec.sequence_trial_index == 1:
            self._start_homing(spec, pose)
        else:
            self._start_trial(spec, pose)

    def _start_homing(
        self, spec: TrialSpec, pose: tuple[float, float, float]
    ) -> None:
        assert self.home_xy is not None
        self.ring_points = ring_targets(
            self.home_xy, spec.distance_m, self.target_count
        )
        self.current_target_index = 0
        self.previous_target_index = None
        self.target_xy = self.ring_points[0]
        self.phase = "homing"
        self.phase_start_ns = self.get_clock().now().nanoseconds
        self.autopilot_rest_latched = False
        self.movement_started = False
        self._reset_rest_detection()
        self._set_gazebo_targets(
            self.target_xy, None, self.plane_z, spec.width_m
        )
        stage = "PRACTICE" if spec.practice else f"SEQUENCE {spec.sequence_number}"
        self.get_logger().info(
            f"{stage} homing: A={spec.distance_m:.3f} m, "
            f"W={spec.width_m:.3f} m, move/select target 0"
        )
        self._publish_status("homing_started")

    def _complete_homing(self) -> None:
        self.phase = "waiting_trial"
        self.next_phase_at_ns = self.get_clock().now().nanoseconds + int(
            self.inter_trial_pause_sec * 1e9
        )
        self._reset_rest_detection()
        self._publish_status("homing_completed")

    def _start_trial(
        self, spec: TrialSpec, pose: tuple[float, float, float]
    ) -> None:
        if not self.ring_points:
            assert self.home_xy is not None
            self.ring_points = ring_targets(
                self.home_xy, spec.distance_m, self.target_count
            )
        self.active_spec = spec
        self.pending_spec = spec
        self.phase = "trial"
        self.current_target_index = spec.to_target_index
        self.previous_target_index = spec.from_target_index
        self.start_target_xy = self.ring_points[spec.from_target_index]
        self.start_xy = (pose[0], pose[1])
        self.target_xy = self.ring_points[spec.to_target_index]
        self.actual_distance = math.dist(self.start_xy, self.target_xy)
        self.id_bits = nominal_id(spec.distance_m, spec.width_m)
        self.trial_start_ns = self.get_clock().now().nanoseconds
        self.trial_start_wall_ns = time.perf_counter_ns()
        if not spec.practice:
            self.formal_trial_index += 1
        self.autopilot_rest_latched = False
        self.movement_started = False
        self.last_movement_label = None
        self.direction_switches = 0
        self.path_length = 0.0
        self.last_path_xy = self.start_xy
        self._reset_rest_detection()
        self._reset_trial_measurements()
        self._set_gazebo_targets(
            self.target_xy,
            self.start_target_xy,
            self.plane_z,
            spec.width_m,
        )
        stage = "PRACTICE" if spec.practice else "MEASURED"
        shown_index = (
            self.practice_completed + 1
            if spec.practice
            else self.formal_trial_index
        )
        shown_total = (
            self.total_practice_trials if spec.practice else self.total_formal_trials
        )
        self.get_logger().info(
            f"{stage} movement {shown_index}/{shown_total}: "
            f"target {spec.from_target_index}->{spec.to_target_index}, "
            f"A={spec.distance_m:.3f} m, W={spec.width_m:.3f} m, "
            f"ID={self.id_bits:.3f} bit"
        )
        self._publish_status("trial_started")

    def _active_stop_radius(self) -> float:
        spec = self.active_spec or self.pending_spec
        target_width = spec.width_m if spec is not None else min(self.widths)
        return width_aware_stop_radius(
            self.autopilot_stop_radius,
            target_width,
            self.width_stop_fraction,
            self.max_stop_radius,
        )

    def _desired_from_error(self, dx: float, dy: float, radial_error: float) -> int:
        if self.autopilot_rest_latched:
            return REST
        stop_radius = self._active_stop_radius()
        if radial_error <= stop_radius:
            # Once the replay policy asks for Rest it remains latched until the
            # model produces a stable Rest selection.  Without this latch the
            # moving robot can leave the small trigger region while the next
            # inference window is still being assembled.
            self.autopilot_rest_latched = True
            return REST
        label_to_direction = {
            HAND_OPEN: "x+",
            LATERAL_GRIP: "x-",
            PINCH_GRIP: "y+",
            POWER_GRIP: "y-",
        }
        direction_to_label = {
            value: key for key, value in label_to_direction.items()
        }
        direction = hysteretic_cardinal_direction(
            dx,
            dy,
            label_to_direction.get(self.desired_label),
            stop_radius * self.axis_hysteresis_fraction,
        )
        return direction_to_label[direction]

    def _complete_trial(
        self,
        *,
        selection_made: bool,
        endpoint: tuple[float, float, float],
        confirmed_pose: tuple[float, float, float],
        completion_ns: int,
    ) -> None:
        assert self.active_spec is not None
        spec = self.active_spec
        selection_ns = self.first_rest_ns if selection_made else completion_ns
        movement_time = max((selection_ns - self.trial_start_ns) / 1e9, 0.0)
        confirmation_time = max((completion_ns - self.trial_start_ns) / 1e9, 0.0)
        confirmation_delay = max((completion_ns - selection_ns) / 1e9, 0.0)
        wall_movement_time = max(
            (time.perf_counter_ns() - self.trial_start_wall_ns) / 1e9, 0.0
        )
        observed_real_time_factor = (
            confirmation_time / wall_movement_time
            if wall_movement_time > 0.0
            else float("nan")
        )
        radial_error = math.dist((endpoint[0], endpoint[1]), self.target_xy)
        hit = bool(selection_made and radial_error <= spec.width_m / 2.0)
        miss = bool(selection_made and not hit)
        timeout = not selection_made
        axis = task_axis(self.start_target_xy, self.target_xy)
        axis_error, orthogonal_error = endpoint_components(
            (endpoint[0], endpoint[1]), self.target_xy, axis
        )
        effective_amplitude = projected_amplitude(
            self.start_xy, (endpoint[0], endpoint[1]), axis
        )
        trial_throughput = (
            self.id_bits / movement_time if movement_time > 0.0 else float("nan")
        )
        classification_accuracy = (
            1.0 - self.classification_errors / self.prediction_count
            if self.prediction_count
            else float("nan")
        )
        if self.evaluation_mode and self.autopilot:
            provenance_valid = bool(self.trial_segment_ids) and self.provenance_errors == 0
        else:
            provenance_valid = self.provenance_errors == 0

        row = {
            "trial_index": self.formal_trial_index,
            "subject_id": self.subject_id,
            "task_protocol": TASK_PROTOCOL,
            "input_mode": "heldout_replay_autopilot" if self.autopilot else "live_emg",
            "evaluation_mode": int(self.evaluation_mode),
            "fold_index": self.fold_index,
            "held_out_condition": self.held_out_condition,
            "block_index": spec.block_index,
            "sequence_number": spec.sequence_number,
            "condition_order": spec.condition_order,
            "condition_id": spec.condition_id,
            "sequence_trial_index": spec.sequence_trial_index,
            "target_count": self.target_count,
            "from_target_index": spec.from_target_index,
            "to_target_index": spec.to_target_index,
            "model_sha256": self.model_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "provenance_valid": int(provenance_valid),
            "provenance_errors": self.provenance_errors,
            "input_segment_ids": json.dumps(
                sorted(self.trial_segment_ids), ensure_ascii=False
            ),
            "input_source_files": json.dumps(
                sorted(self.trial_source_files), ensure_ascii=False
            ),
            "input_sample_ranges": json.dumps(
                self.trial_source_ranges, ensure_ascii=False
            ),
            "stream_instance_ids": json.dumps(sorted(self.trial_stream_ids)),
            "replay_wrap_count": self.trial_replay_wrap_count,
            "distance_config_m": spec.distance_m,
            "distance_center_m": math.dist(self.start_target_xy, self.target_xy),
            "distance_actual_m": self.actual_distance,
            "target_width_m": spec.width_m,
            "effective_stop_radius_m": width_aware_stop_radius(
                self.autopilot_stop_radius,
                spec.width_m,
                self.width_stop_fraction,
                self.max_stop_radius,
            ),
            "index_of_difficulty_bits": self.id_bits,
            "timing_clock": "ros_time",
            "selection_made": int(selection_made),
            "hit": int(hit),
            "miss": int(miss),
            "timeout": int(timeout),
            "success": int(hit),
            "movement_time_s": movement_time,
            "confirmation_time_s": confirmation_time,
            "selection_confirmation_delay_s": confirmation_delay,
            "wall_movement_time_s": wall_movement_time,
            "observed_real_time_factor": observed_real_time_factor,
            "throughput_bits_per_s": trial_throughput,
            "classification_errors": self.classification_errors,
            "direction_errors": self.direction_errors,
            "prediction_count": self.prediction_count,
            "classification_accuracy": classification_accuracy,
            "mean_inference_ms": _mean(self.inference_latencies),
            "p95_inference_ms": _p95(self.inference_latencies),
            "mean_control_latency_ms": _mean(self.control_latencies),
            "p95_control_latency_ms": _p95(self.control_latencies),
            "path_length_m": self.path_length,
            "direction_switches": self.direction_switches,
            "endpoint_radial_error_m": radial_error,
            "endpoint_axis_error_m": axis_error,
            "endpoint_orthogonal_error_m": orthogonal_error,
            "effective_amplitude_m": effective_amplitude,
            "start_target_x": self.start_target_xy[0],
            "start_target_y": self.start_target_xy[1],
            "start_x": self.start_xy[0],
            "start_y": self.start_xy[1],
            "target_x": self.target_xy[0],
            "target_y": self.target_xy[1],
            "first_selection_x": endpoint[0] if selection_made else "",
            "first_selection_y": endpoint[1] if selection_made else "",
            "confirmed_end_x": confirmed_pose[0],
            "confirmed_end_y": confirmed_pose[1],
            "end_x": endpoint[0],
            "end_y": endpoint[1],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

        if spec.practice:
            self.practice_completed += 1
        else:
            self.rows.append(row)
            self._write_csv()
        outcome = "HIT" if hit else "MISS" if miss else "TIMEOUT"
        self.last_trial_outcome = outcome
        self.get_logger().info(
            f"{'PRACTICE ' if spec.practice else ''}{outcome}: "
            f"MT={movement_time:.3f}s ROS, endpoint error={radial_error:.4f}m, "
            f"classification errors={self.classification_errors}"
        )
        self.task_cursor += 1
        self.active_spec = None
        self.phase = "inter_trial"
        self.desired_label = REST
        self.next_phase_at_ns = self.get_clock().now().nanoseconds + int(
            self.inter_trial_pause_sec * 1e9
        )
        self._reset_rest_detection()
        self._publish_status("trial_completed")

    def _reset_trial_measurements(self) -> None:
        self.classification_errors = 0
        self.direction_errors = 0
        self.prediction_count = 0
        self.inference_latencies = []
        self.control_latencies = []
        self.seen_control_ids = set()
        self.provenance_errors = 0
        self.trial_segment_ids = set()
        self.trial_source_files = set()
        self.trial_source_ranges = {}
        self.trial_stream_ids = set()
        self.trial_replay_wrap_count = 0

    def _reset_rest_detection(self) -> None:
        self.rest_prediction_streak = 0
        self.first_rest_ns = 0
        self.first_rest_pose = None
        self.confirmation_ns = 0
        self.confirmation_pose = None
        self.selection_ready = False

    def _condition_summaries(self) -> list[dict]:
        summaries: list[dict] = []
        for distance in self.distances:
            for width in self.widths:
                rows = [
                    row
                    for row in self.rows
                    if math.isclose(float(row["distance_config_m"]), distance)
                    and math.isclose(float(row["target_width_m"]), width)
                ]
                metrics = condition_metrics(rows)
                summaries.append(
                    {
                        "condition_id": (
                            f"FORMAL_A{round(distance * 1000):03d}"
                            f"_W{round(width * 1000):03d}"
                        ),
                        "distance_m": distance,
                        "width_m": width,
                        "nominal_id_bits": nominal_id(distance, width),
                        **metrics,
                    }
                )
        return summaries

    def _sequence_summaries(self) -> list[dict]:
        summaries: list[dict] = []
        sequence_numbers = sorted(
            {int(row["sequence_number"]) for row in self.rows}
        )
        for sequence_number in sequence_numbers:
            rows = [
                row
                for row in self.rows
                if int(row["sequence_number"]) == sequence_number
            ]
            metrics = condition_metrics(rows)
            first = rows[0]
            summaries.append(
                {
                    "sequence_number": sequence_number,
                    "block_index": int(first["block_index"]),
                    "condition_order": int(first["condition_order"]),
                    "condition_id": str(first["condition_id"]),
                    "distance_m": float(first["distance_config_m"]),
                    "width_m": float(first["target_width_m"]),
                    "nominal_id_bits": float(first["index_of_difficulty_bits"]),
                    **metrics,
                }
            )
        return summaries

    def _summary(self) -> dict:
        condition_summaries = self._condition_summaries()
        sequence_summaries = self._sequence_summaries()
        hits = [row for row in self.rows if int(row["hit"]) == 1]
        misses = [row for row in self.rows if int(row["miss"]) == 1]
        timeouts = [row for row in self.rows if int(row["timeout"]) == 1]
        selections = [row for row in self.rows if int(row["selection_made"]) == 1]
        finite_conditions = [
            item
            for item in condition_summaries
            if math.isfinite(float(item["throughput_bits_per_s"]))
        ]
        regression_conditions = [
            item
            for item in condition_summaries
            if math.isfinite(float(item["mean_movement_time_s"]))
        ]
        regression = linear_regression(
            [item["nominal_id_bits"] for item in regression_conditions],
            [item["mean_movement_time_s"] for item in regression_conditions],
        )
        total_predictions = sum(int(row["prediction_count"]) for row in self.rows)
        total_classification_errors = sum(
            int(row["classification_errors"]) for row in self.rows
        )
        summary = {
            "subject_id": self.subject_id,
            "task_protocol": TASK_PROTOCOL,
            "iso_conformant": False,
            "iso_scope_note": (
                "Adapted multidirectional Fitts geometry; EMG gesture input is "
                "outside ISO 9241-411 physical-input-device scope."
            ),
            "input_mode": "heldout_replay_autopilot" if self.autopilot else "live_emg",
            "interpretation": (
                "robot_controller_fitts_style_benchmark"
                if self.autopilot
                else "live_hri_fitts_evaluation"
            ),
            "evaluation_mode": self.evaluation_mode,
            "evaluation_protocol": EVALUATION_PROTOCOL if self.evaluation_mode else "demo",
            "evaluation_split_id": (
                EVALUATION_SPLIT_ID if self.evaluation_mode else "demo"
            ),
            "timing_clock": "ros_time",
            "fold_index": self.fold_index,
            "held_out_condition": self.held_out_condition,
            "model_path": str(self.model_path),
            "model_sha256": self.model_sha256,
            "dataset_sha256": self.dataset_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "control_policy": CONTROL_POLICY,
            "control_policy_parameters": {
                "minimum_stop_radius_m": self.autopilot_stop_radius,
                "width_stop_fraction": self.width_stop_fraction,
                "maximum_stop_radius_m": self.max_stop_radius,
                "axis_hysteresis_fraction": self.axis_hysteresis_fraction,
                "near_threshold_m": self.speed_near_threshold,
                "far_threshold_m": self.speed_far_threshold,
                "near_speed_scale": self.speed_scale_near,
                "mid_speed_scale": self.speed_scale_mid,
                "far_speed_scale": self.speed_scale_far,
            },
            "target_count": self.target_count,
            "target_order": self.order,
            "distances_m": self.distances,
            "widths_m": self.widths,
            "sequences_per_condition": self.sequences_per_condition,
            "practice_trials_completed": self.practice_completed,
            "practice_trials_total": self.total_practice_trials,
            "completed_trials": len(self.rows),
            "total_trials": self.total_formal_trials,
            "selection_count": len(selections),
            "hit_count": len(hits),
            "miss_count": len(misses),
            "timeout_count": len(timeouts),
            "success_count": len(hits),
            "success_rate": len(hits) / len(self.rows) if self.rows else 0.0,
            "error_rate": (
                (len(misses) + len(timeouts)) / len(self.rows)
                if self.rows
                else 0.0
            ),
            "selection_error_rate": (
                len(misses) / len(selections) if selections else 0.0
            ),
            "total_failure_rate": (
                (len(misses) + len(timeouts)) / len(self.rows)
                if self.rows
                else 0.0
            ),
            "mean_movement_time_s": _mean(
                [float(row["movement_time_s"]) for row in selections]
            ),
            "mean_wall_movement_time_s": _mean(
                [float(row["wall_movement_time_s"]) for row in self.rows]
            ),
            "mean_observed_real_time_factor": _mean(
                [float(row["observed_real_time_factor"]) for row in self.rows]
            ),
            "mean_throughput_bits_per_s": _mean(
                [
                    float(item["throughput_bits_per_s"])
                    for item in finite_conditions
                ]
            ),
            "throughput_method": (
                "condition level: We=4.133*SDx, "
                "IDe=log2(Ae/We+1), TP=IDe/mean(MT)"
            ),
            "classification_errors": total_classification_errors,
            "prediction_count": total_predictions,
            "classification_accuracy": (
                1.0 - total_classification_errors / total_predictions
                if total_predictions
                else float("nan")
            ),
            "direction_errors": sum(
                int(row["direction_errors"]) for row in self.rows
            ),
            "provenance_valid_trials": sum(
                int(row["provenance_valid"]) for row in self.rows
            ),
            "provenance_error_count": sum(
                int(row["provenance_errors"]) for row in self.rows
            ),
            "mean_inference_ms": _mean(
                [
                    float(row["mean_inference_ms"])
                    for row in self.rows
                    if math.isfinite(float(row["mean_inference_ms"]))
                ]
            ),
            "mean_control_latency_ms": _mean(
                [
                    float(row["mean_control_latency_ms"])
                    for row in self.rows
                    if math.isfinite(float(row["mean_control_latency_ms"]))
                ]
            ),
            "mean_path_length_m": _mean(
                [float(row["path_length_m"]) for row in self.rows]
            ),
            "total_direction_switches": sum(
                int(row["direction_switches"]) for row in self.rows
            ),
            "fitts_regression_nominal_id": regression,
            "per_condition": condition_summaries,
            "per_sequence": sequence_summaries,
            "csv_path": str(self.csv_path),
            "condition_summary_csv_path": str(self.condition_csv_path),
        }
        return _json_safe(summary)

    def _finish_experiment(self) -> None:
        self.finished = True
        self.phase = "finished"
        summary = self._summary()
        self._write_condition_csv(summary["per_condition"])
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        msg = String()
        msg.data = json.dumps(summary, ensure_ascii=False)
        self.summary_pub.publish(msg)
        self._hide_gazebo_targets()
        throughput = summary["mean_throughput_bits_per_s"]
        throughput_text = (
            f"{throughput:.3f}" if isinstance(throughput, (int, float)) else "undefined"
        )
        self.get_logger().info(
            f"Experiment finished: hits={summary['hit_count']}/"
            f"{summary['total_trials']}, misses={summary['miss_count']}, "
            f"timeouts={summary['timeout_count']}, "
            f"condition-level TP={throughput_text} bit/s"
        )

    def _write_csv(self) -> None:
        with self.csv_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=self.CSV_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(self.rows)

    def _write_condition_csv(self, rows: list[dict]) -> None:
        if not rows:
            return
        safe_rows = [_json_safe(row) for row in rows]
        with self.condition_csv_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(safe_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(safe_rows)

    def _publish_desired(self, label: int) -> None:
        self.desired_label = int(label)
        msg = Int32()
        msg.data = self.desired_label
        self.desired_pub.publish(msg)

    def _publish_speed_scale(self, scale: float) -> None:
        msg = Float64()
        msg.data = float(scale)
        self.speed_scale_pub.publish(msg)

    def _publish_status(self, event: str) -> None:
        spec = self.active_spec or self.pending_spec
        payload = {
            "event": event,
            "phase": self.phase,
            "task_protocol": TASK_PROTOCOL,
            "input_mode": "heldout_replay_autopilot" if self.autopilot else "live_emg",
            "evaluation_mode": self.evaluation_mode,
            "fold_index": self.fold_index,
            "held_out_condition": self.held_out_condition,
            "trial_index": self.formal_trial_index,
            "total_trials": self.total_formal_trials,
            "practice_completed": self.practice_completed,
            "practice_total": self.total_practice_trials,
            "sequence_number": spec.sequence_number if spec else 0,
            "sequence_trial_index": spec.sequence_trial_index if spec else 0,
            "condition_id": spec.condition_id if spec else "",
            "distance_m": spec.distance_m if spec else None,
            "width_m": spec.width_m if spec else None,
            "effective_stop_radius_m": (
                width_aware_stop_radius(
                    self.autopilot_stop_radius,
                    spec.width_m,
                    self.width_stop_fraction,
                    self.max_stop_radius,
                )
                if spec
                else None
            ),
            "desired_label": self.desired_label,
            "last_outcome": self.last_trial_outcome,
            "summary": self._summary(),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def _publish_markers(self, pose: tuple[float, float, float]) -> None:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        plane = Marker()
        plane.header.frame_id = self.base_frame
        plane.header.stamp = stamp
        plane.ns = "fitts"
        plane.id = 0
        plane.type = Marker.CUBE
        plane.action = Marker.ADD
        plane.pose.position.x = self.home_xy[0] if self.home_xy else pose[0]
        plane.pose.position.y = (
            self.home_xy[1] - self.workspace_radius / 2.0
            if self.home_xy
            else pose[1]
        )
        plane.pose.position.z = self.plane_z - 0.012
        plane.pose.orientation.w = 1.0
        plane.scale.x = self.workspace_radius * 2.2
        plane.scale.y = self.workspace_radius * 2.2
        plane.scale.z = 0.004
        plane.color.r = 0.15
        plane.color.g = 0.35
        plane.color.b = 0.65
        plane.color.a = 0.22
        plane.lifetime.nanosec = 200_000_000
        markers.markers.append(plane)

        cursor = Marker()
        cursor.header = plane.header
        cursor.ns = "fitts"
        cursor.id = 1
        cursor.type = Marker.SPHERE
        cursor.action = Marker.ADD
        cursor.pose.position.x = pose[0]
        cursor.pose.position.y = pose[1]
        cursor.pose.position.z = pose[2]
        cursor.pose.orientation.w = 1.0
        cursor.scale.x = cursor.scale.y = cursor.scale.z = 0.025
        cursor.color.r = 1.0
        cursor.color.g = 0.78
        cursor.color.b = 0.02
        cursor.color.a = 1.0
        cursor.lifetime.nanosec = 200_000_000
        markers.markers.append(cursor)

        spec = self.active_spec or self.pending_spec
        if spec is not None and self.ring_points:
            for index, point in enumerate(self.ring_points):
                target = Marker()
                target.header = plane.header
                target.ns = "fitts_targets"
                target.id = 100 + index
                target.type = Marker.SPHERE
                target.action = Marker.ADD
                target.pose.position.x = point[0]
                target.pose.position.y = point[1]
                target.pose.position.z = self.plane_z
                target.pose.orientation.w = 1.0
                target.scale.x = target.scale.y = target.scale.z = spec.width_m
                if index == self.current_target_index:
                    target.color.r = 1.0
                    target.color.g = 0.02
                    target.color.b = 0.01
                    target.color.a = 1.0
                elif index == self.previous_target_index:
                    target.color.r = 0.05
                    target.color.g = 0.95
                    target.color.b = 0.20
                    target.color.a = 0.90
                else:
                    target.color.r = 0.68
                    target.color.g = 0.70
                    target.color.b = 0.74
                    target.color.a = 0.55
                target.lifetime.nanosec = 200_000_000
                markers.markers.append(target)

            halo = Marker()
            halo.header = plane.header
            halo.ns = "fitts"
            halo.id = 2
            halo.type = Marker.CYLINDER
            halo.action = Marker.ADD
            halo.pose.position.x = self.target_xy[0]
            halo.pose.position.y = self.target_xy[1]
            halo.pose.position.z = self.plane_z - spec.width_m / 2.0
            halo.pose.orientation.w = 1.0
            halo.scale.x = spec.width_m * 1.45
            halo.scale.y = spec.width_m * 1.45
            halo.scale.z = 0.008
            halo.color.r = 1.0
            halo.color.g = 0.05
            halo.color.b = 0.02
            halo.color.a = 0.30
            halo.lifetime.nanosec = 200_000_000
            markers.markers.append(halo)

            direction = Marker()
            direction.header = plane.header
            direction.ns = "fitts"
            direction.id = 3
            direction.type = Marker.LINE_STRIP
            direction.action = Marker.ADD
            direction.scale.x = 0.004
            direction.color.r = 0.05
            direction.color.g = 0.95
            direction.color.b = 1.0
            direction.color.a = 0.85
            direction.points = [
                Point(x=pose[0], y=pose[1], z=pose[2]),
                Point(x=self.target_xy[0], y=self.target_xy[1], z=self.plane_z),
            ]
            direction.lifetime.nanosec = 200_000_000
            markers.markers.append(direction)

            text = Marker()
            text.header = plane.header
            text.ns = "fitts"
            text.id = 4
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = self.target_xy[0]
            text.pose.position.y = self.target_xy[1]
            text.pose.position.z = self.plane_z + 0.075
            text.pose.orientation.w = 1.0
            text.scale.z = 0.030
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            if self.phase == "homing":
                phase_text = "HOME (not measured)"
            elif spec.practice:
                phase_text = (
                    f"PRACTICE {self.practice_completed + 1}/"
                    f"{self.total_practice_trials}"
                )
            else:
                phase_text = (
                    f"MEASURED {self.formal_trial_index}/"
                    f"{self.total_formal_trials}"
                )
            text.text = (
                f"{phase_text}  TARGET {self.current_target_index}  "
                f"A={spec.distance_m:.3f}m  W={spec.width_m:.3f}m  "
                f"LAST={self.last_trial_outcome or '-'}"
            )
            text.lifetime.nanosec = 200_000_000
            markers.markers.append(text)
        self.marker_pub.publish(markers)

    def _set_entity_pose(self, name: str, x: float, y: float, z: float) -> None:
        if not self.pose_client.service_is_ready():
            return
        request = SetEntityPose.Request()
        request.entity.name = name
        request.entity.type = Entity.MODEL
        request.pose = Pose()
        request.pose.position.x = x
        request.pose.position.y = y
        request.pose.position.z = z
        request.pose.orientation.w = 1.0
        self.pose_client.call_async(request)

    def _set_gazebo_targets(
        self,
        target_xy: tuple[float, float],
        previous_xy: tuple[float, float] | None,
        plane_z: float,
        width: float,
    ) -> None:
        selected = min(self.target_models, key=lambda value: abs(value - width))
        for model_width, name in self.target_models.items():
            z = self.base_world_z + plane_z if model_width == selected else -1.0
            self._set_entity_pose(name, target_xy[0], target_xy[1], z)
        for model_width, name in self.previous_models.items():
            if model_width == selected and previous_xy is not None:
                self._set_entity_pose(
                    name,
                    previous_xy[0],
                    previous_xy[1],
                    self.base_world_z + plane_z,
                )
            else:
                self._set_entity_pose(name, 0.0, 0.0, -1.0)

    def _hide_gazebo_targets(self) -> None:
        for name in [*self.target_models.values(), *self.previous_models.values()]:
            self._set_entity_pose(name, 0.0, 0.0, -1.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FittsTaskNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32, MultiArrayDimension, String

from .constants import (
    CLASS_NAMES,
    EMG_CHANNELS,
    EMG_POINTS,
    HELD_OUT_CONDITION,
    IMU_CHANNELS,
    IMU_POINTS,
    INPUT_FEATURES,
    REST,
    STEP_SECONDS,
)


FEATURE_VALUE_COUNT = (
    INPUT_FEATURES
    + EMG_CHANNELS * EMG_POINTS
    + IMU_CHANNELS * IMU_POINTS
)
MESSAGE_VALUE_COUNT = FEATURE_VALUE_COUNT + 2


class SjReplayNode(Node):
    """Replay only held-out dynamic SJ windows as a real-time ROS stream."""

    def __init__(self) -> None:
        super().__init__("sj_emg_replay")
        self.declare_parameter("project_root", "")
        self.declare_parameter("dataset_path", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("split_manifest_path", "")
        self.declare_parameter("condition", HELD_OUT_CONDITION)
        self.declare_parameter("publish_period_sec", STEP_SECONDS)
        self.declare_parameter("random_seed", 42)
        self.declare_parameter("subject_id", 1)
        self.declare_parameter("fold_index", 1)
        self.declare_parameter("evaluation_mode", True)

        project_root = Path(str(self.get_parameter("project_root").value))
        configured_dataset = str(self.get_parameter("dataset_path").value).strip()
        self.dataset_path = (
            Path(configured_dataset)
            if configured_dataset
            else project_root
            / "artifacts"
            / "prepared"
            / "window_dataset.npz"
        )
        configured_manifest = str(
            self.get_parameter("split_manifest_path").value
        ).strip()
        self.split_manifest_path = Path(configured_manifest)
        self.condition = str(self.get_parameter("condition").value)
        self.subject_id = int(self.get_parameter("subject_id").value)
        self.fold_index = int(self.get_parameter("fold_index").value)
        self.evaluation_mode = bool(
            self.get_parameter("evaluation_mode").value
        )
        if self.condition != HELD_OUT_CONDITION:
            raise ValueError(
                f"SJ evaluation must replay {HELD_OUT_CONDITION!r}, got {self.condition!r}"
            )
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"Prepared SJ dataset not found: {self.dataset_path}")
        if not self.split_manifest_path.is_file():
            raise FileNotFoundError(
                f"SJ split manifest not found: {self.split_manifest_path}"
            )
        self.split_manifest = json.loads(
            self.split_manifest_path.read_text(encoding="utf-8")
        )
        training_conditions = set(
            map(str, self.split_manifest.get("final_training_conditions", []))
        )
        if self.condition in training_conditions:
            raise ValueError(
                f"Held-out condition {self.condition!r} appears in training conditions"
            )
        if (
            str(self.split_manifest.get("held_out_test_condition", ""))
            != self.condition
        ):
            raise ValueError(
                "Split manifest held-out condition does not match replay condition"
            )

        dataset = np.load(self.dataset_path, allow_pickle=False)
        self.basic = np.asarray(dataset["X_features"], dtype=np.float32)
        self.emg = np.asarray(dataset["X_emg"], dtype=np.float32)
        self.imu = np.asarray(dataset["X_imu"], dtype=np.float32)
        self.labels = np.asarray(dataset["y"], dtype=np.int64)
        self.conditions = np.asarray(dataset["conditions"]).astype(str)
        self.source_files = np.asarray(dataset["source_files"]).astype(str)
        self.start_times = np.asarray(dataset["start_times_sec"], dtype=np.float32)
        self._validate_dataset()

        self.indices_by_label: dict[int, np.ndarray] = {}
        held_out = self.conditions == self.condition
        for label in range(len(CLASS_NAMES)):
            indices = np.flatnonzero(held_out & (self.labels == label))
            if len(indices) == 0:
                raise ValueError(
                    f"No {self.condition} windows for label {label}:{CLASS_NAMES[label]}"
                )
            order = np.lexsort((self.start_times[indices], self.source_files[indices]))
            self.indices_by_label[label] = indices[order]

        self.rng = np.random.default_rng(
            int(self.get_parameter("random_seed").value)
        )
        self.requested_label = REST
        self.cursor = 0
        self.stream_instance_id = 0
        self.loop_count = 0

        model_path = Path(str(self.get_parameter("model_path").value))
        self.model_sha256 = self._sha256(model_path) if model_path.is_file() else ""
        self.dataset_sha256 = self._sha256(self.dataset_path)
        self.split_manifest_sha256 = self._sha256(self.split_manifest_path)

        self.window_pub = self.create_publisher(
            Float32MultiArray, "/sj/raw_window", 10
        )
        self.truth_pub = self.create_publisher(Int32, "/sj/ground_truth", 10)
        self.provenance_pub = self.create_publisher(
            String, "/sj/replay_provenance", 20
        )
        self.create_subscription(
            Int32,
            "/fitts/demo_requested_label",
            self._requested_cb,
            10,
        )
        period = float(self.get_parameter("publish_period_sec").value)
        if period <= 0.0:
            raise ValueError("publish_period_sec must be positive")
        self.timer = self.create_timer(period, self._timer_cb)
        self.get_logger().info(
            f"Loaded {sum(map(len, self.indices_by_label.values()))} held-out "
            f"{self.condition} windows from {self.dataset_path}; "
            f"{len(self.indices_by_label[REST])} windows per gesture expected"
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _validate_dataset(self) -> None:
        count = len(self.labels)
        if self.basic.shape != (count, INPUT_FEATURES):
            raise ValueError(f"Unexpected basic feature shape: {self.basic.shape}")
        if self.emg.shape != (count, EMG_CHANNELS, EMG_POINTS):
            raise ValueError(f"Unexpected EMG shape: {self.emg.shape}")
        if self.imu.shape != (count, IMU_CHANNELS, IMU_POINTS):
            raise ValueError(f"Unexpected IMU shape: {self.imu.shape}")
        if not (
            len(self.conditions)
            == len(self.source_files)
            == len(self.start_times)
            == count
        ):
            raise ValueError("Prepared dataset arrays have inconsistent lengths")

    def _requested_cb(self, msg: Int32) -> None:
        label = int(msg.data)
        if label not in self.indices_by_label:
            self.get_logger().error(f"Unsupported requested label: {label}")
            return
        if label != self.requested_label:
            self.requested_label = label
            self.stream_instance_id += 1
            self.loop_count = 0
            # Each change selects a new starting point in the independent
            # sustained-action recording. Causal smoothing is reset by the
            # classifier on the matching ground-truth change.
            self.cursor = int(self.rng.integers(0, len(self.indices_by_label[label])))

    def _timer_cb(self) -> None:
        indices = self.indices_by_label[self.requested_label]
        if self.cursor >= len(indices):
            self.cursor = 0
            self.loop_count += 1
            self.stream_instance_id += 1
        index = int(indices[self.cursor])
        self.cursor += 1

        features = np.concatenate(
            [
                self.basic[index].reshape(-1),
                self.emg[index].reshape(-1),
                self.imu[index].reshape(-1),
            ]
        ).astype(np.float32, copy=False)
        if features.size != FEATURE_VALUE_COUNT:
            raise RuntimeError(
                f"Flattened window has {features.size} feature values"
            )
        # Keep the ground-truth label and replay stream ID in the same DDS
        # sample as the signal. Separate topics can be delivered in a
        # different order during gesture transitions and would corrupt both
        # the reported accuracy and the causal-history reset boundary.
        flat = np.concatenate(
            [
                np.asarray(
                    [self.requested_label, self.stream_instance_id],
                    dtype=np.float32,
                ),
                features,
            ]
        )

        message = Float32MultiArray()
        message.layout.dim = [
            MultiArrayDimension(
                label="sj_truth_stream_features",
                size=MESSAGE_VALUE_COUNT,
                stride=MESSAGE_VALUE_COUNT,
            )
        ]
        message.data = flat.tolist()
        self.window_pub.publish(message)

        truth = Int32()
        truth.data = self.requested_label
        self.truth_pub.publish(truth)

        provenance = String()
        provenance.data = json.dumps(
            {
                "evaluation_mode": self.evaluation_mode,
                "subject_id": self.subject_id,
                "fold_index": self.fold_index,
                "condition": self.condition,
                "label": self.requested_label,
                "name": CLASS_NAMES[self.requested_label],
                "segment_id": self.source_files[index],
                "source_file": self.source_files[index],
                "source_start_sec": float(self.start_times[index]),
                "source_end_sec": float(self.start_times[index] + 0.5),
                "source_start_index": index,
                "source_end_index": index,
                "stream_instance_id": self.stream_instance_id,
                "loop_count": self.loop_count,
                "model_sha256": self.model_sha256,
                "dataset_sha256": self.dataset_sha256,
                "split_manifest_path": str(self.split_manifest_path),
                "split_manifest_sha256": self.split_manifest_sha256,
                "held_out": True,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.provenance_pub.publish(provenance)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SjReplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

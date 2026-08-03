from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32, String

from .constants import (
    CLASS_NAMES,
    EMG_CHANNELS,
    EMG_POINTS,
    HELD_OUT_CONDITION,
    IMU_CHANNELS,
    IMU_POINTS,
    INPUT_FEATURES,
    LABEL_NAMES,
)
from .model_adapter import OptimizedSjModel


FEATURE_VALUE_COUNT = (
    INPUT_FEATURES
    + EMG_CHANNELS * EMG_POINTS
    + IMU_CHANNELS * IMU_POINTS
)
MESSAGE_VALUE_COUNT = FEATURE_VALUE_COUNT + 2


class GestureClassifierNode(Node):
    """Held-out SJ window -> optimized features -> Extra Trees -> smoothing."""

    def __init__(self) -> None:
        super().__init__("sj_gesture_classifier")
        self.declare_parameter("project_root", "")
        self.declare_parameter("model_path", "")
        self.declare_parameter("subject_id", 1)
        self.declare_parameter("expected_condition", HELD_OUT_CONDITION)

        project_root = Path(str(self.get_parameter("project_root").value))
        configured_model = str(self.get_parameter("model_path").value).strip()
        model_path = (
            Path(configured_model)
            if configured_model
            else project_root
            / "artifacts"
            / "ml"
            / "optimized"
            / "improved_strict_smoothed_model.joblib"
        )
        expected_condition = str(
            self.get_parameter("expected_condition").value
        )
        if expected_condition != HELD_OUT_CONDITION:
            raise ValueError(
                f"SJ evaluation expects {HELD_OUT_CONDITION!r}, got "
                f"{expected_condition!r}"
        )
        self.subject_id = int(self.get_parameter("subject_id").value)
        self.model = OptimizedSjModel(model_path)
        self.stream_instance_id: int | None = None
        self.prediction_count = 0

        self.prediction_pub = self.create_publisher(
            String, "/sj/prediction", 10
        )
        self.label_pub = self.create_publisher(
            Int32, "/sj/predicted_label", 10
        )
        self.create_subscription(
            Float32MultiArray,
            "/sj/raw_window",
            self._window_cb,
            10,
        )
        self.get_logger().info(
            f"Loaded optimized SJ model {model_path}; "
            f"feature_set={self.model.feature_set}, "
            f"history={self.model.history_windows} windows, "
            f"sha256={self.model.sha256[:12]}..."
        )

    @staticmethod
    def _decode(
        message: Float32MultiArray,
    ) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
        flat = np.asarray(message.data, dtype=np.float32)
        if flat.size != MESSAGE_VALUE_COUNT:
            raise ValueError(
                f"Expected {MESSAGE_VALUE_COUNT} message values, got {flat.size}"
            )
        ground_truth = int(flat[0])
        stream_instance_id = int(flat[1])
        if not 0 <= ground_truth < len(CLASS_NAMES):
            raise ValueError(f"Invalid embedded ground-truth label: {ground_truth}")
        if stream_instance_id < 0:
            raise ValueError(
                f"Invalid embedded stream instance: {stream_instance_id}"
            )
        feature_start = 2
        feature_end = feature_start + INPUT_FEATURES
        emg_end = feature_end + EMG_CHANNELS * EMG_POINTS
        basic = flat[feature_start:feature_end]
        emg = flat[feature_end:emg_end].reshape(EMG_CHANNELS, EMG_POINTS)
        imu = flat[emg_end:].reshape(IMU_CHANNELS, IMU_POINTS)
        return ground_truth, stream_instance_id, basic, emg, imu

    def _window_cb(self, message: Float32MultiArray) -> None:
        received_at_ns = self.get_clock().now().nanoseconds
        try:
            ground_truth, stream_instance_id, basic, emg, imu = self._decode(
                message
            )
            if stream_instance_id != self.stream_instance_id:
                # The replay switched to another independent sustained-action
                # segment (or wrapped the recording), so the causal history
                # must not leak across that boundary.
                self.model.reset()
                self.stream_instance_id = stream_instance_id
            prediction = self.model.predict(basic, emg, imu)
            self.prediction_count += 1
            published_at_ns = self.get_clock().now().nanoseconds
            payload = {
                "subject_id": self.subject_id,
                "evaluation_mode": True,
                "fold_index": 1,
                "held_out_condition": HELD_OUT_CONDITION,
                "model_sha256": self.model.sha256,
                "label": prediction.label,
                "name": LABEL_NAMES[prediction.label],
                "raw_label": prediction.raw_label,
                "raw_name": LABEL_NAMES[prediction.raw_label],
                "confidence": prediction.confidence,
                "raw_confidence": prediction.raw_confidence,
                "probabilities": prediction.probabilities,
                "raw_probabilities": prediction.raw_probabilities,
                "history_count": prediction.history_count,
                "history_windows": self.model.history_windows,
                "ground_truth": ground_truth,
                "ground_truth_name": CLASS_NAMES[ground_truth],
                "stream_instance_id": stream_instance_id,
                "correct": prediction.label == ground_truth,
                "feature_ms": prediction.feature_ms,
                "model_ms": prediction.model_ms,
                "inference_ms": prediction.inference_ms,
                "received_at_ns": received_at_ns,
                "published_at_ns": published_at_ns,
                "prediction_count": self.prediction_count,
            }
            text = String()
            text.data = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            )
            self.prediction_pub.publish(text)
            label = Int32()
            label.data = prediction.label
            self.label_pub.publish(label)
        except Exception as exc:
            self.get_logger().error(
                f"SJ classification failed: {type(exc).__name__}: {exc}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GestureClassifierNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

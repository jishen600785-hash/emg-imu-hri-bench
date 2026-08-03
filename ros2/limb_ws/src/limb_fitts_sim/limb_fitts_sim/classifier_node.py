from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Int32, String

from .constants import INPUT_CHANNELS, LABEL_NAMES, STRICT_FOLD_ARTIFACT_ROLE
from .model_adapter import PersonalizedLimbModel


class GestureClassifierNode(Node):
    """Raw samples -> trained window -> exact features -> fitted pipeline."""

    def __init__(self) -> None:
        super().__init__("gesture_classifier")
        self.declare_parameter("project_root", "")
        self.declare_parameter("subject_id", 1)
        self.declare_parameter("input_channels", INPUT_CHANNELS)
        self.declare_parameter("reset_on_ground_truth_change", True)
        self.declare_parameter("model_path", "")
        self.declare_parameter("evaluation_mode", False)
        self.declare_parameter("fold_index", 0)
        self.declare_parameter("expected_condition", "")

        project_root = Path(str(self.get_parameter("project_root").value))
        subject_id = int(self.get_parameter("subject_id").value)
        self.input_channels = int(self.get_parameter("input_channels").value)
        self.reset_on_truth_change = bool(self.get_parameter("reset_on_ground_truth_change").value)
        self.evaluation_mode = bool(self.get_parameter("evaluation_mode").value)
        self.fold_index = int(self.get_parameter("fold_index").value)
        self.expected_condition = str(self.get_parameter("expected_condition").value)
        configured_model_path = str(self.get_parameter("model_path").value).strip()
        model_path = Path(configured_model_path) if configured_model_path else (
            project_root / "models" / "limb_personalized" / f"limb_subject{subject_id:02d}_deployment.joblib"
        )
        self.model = PersonalizedLimbModel(model_path)
        if self.model.subject_id != subject_id:
            raise RuntimeError(
                f"Model subject {self.model.subject_id} does not match requested Subject {subject_id}"
            )
        if self.evaluation_mode:
            self._validate_strict_artifact()

        self.buffer = np.empty((0, self.input_channels), dtype=np.float32)
        self.total_samples = 0
        self.last_prediction_sample = 0
        self.ground_truth = -1

        self.prediction_pub = self.create_publisher(String, "/limb/prediction", 10)
        self.label_pub = self.create_publisher(Int32, "/limb/predicted_label", 10)
        self.create_subscription(Float32MultiArray, "/limb/raw_window_chunks", self._samples_cb, 10)
        self.create_subscription(Int32, "/limb/ground_truth", self._truth_cb, 10)

        meta = self.model.metadata
        self.get_logger().info(
            f"Subject {self.model.subject_id} model loaded: {meta['model_type']}, "
            f"{meta['modality']}, window={self.model.window_samples}, "
            f"step={self.model.step_samples}, features={self.model.feature_dim}, "
            f"fold={meta.get('fold_index', 0)}, held-out={meta.get('held_out_condition', 'none')}"
        )

    def _validate_strict_artifact(self) -> None:
        meta = self.model.metadata
        if meta.get("artifact_role") != STRICT_FOLD_ARTIFACT_ROLE:
            raise RuntimeError(
                "Evaluation mode refuses a full-calibration model; run build_heldout_models.sh first"
            )
        if int(meta.get("fold_index", -1)) != self.fold_index:
            raise RuntimeError(
                f"Model fold {meta.get('fold_index')} does not match requested fold {self.fold_index}"
            )
        if str(meta.get("held_out_condition", "")) != self.expected_condition:
            raise RuntimeError(
                f"Model held-out condition {meta.get('held_out_condition')} does not match "
                f"requested {self.expected_condition}"
            )
        training_ids = set(str(value) for value in meta.get("training_segment_ids", []))
        test_ids = set(str(value) for value in meta.get("test_segment_ids", []))
        overlap = training_ids & test_ids
        if overlap:
            raise RuntimeError(f"Model metadata contains train/test leakage: {sorted(overlap)}")
        if len(training_ids) != 35 or len(test_ids) != 5:
            raise RuntimeError(
                f"Expected 35 train+validation and 5 test segments, got "
                f"{len(training_ids)} and {len(test_ids)}"
            )

    def _truth_cb(self, msg: Int32) -> None:
        value = int(msg.data)
        if self.reset_on_truth_change and self.ground_truth >= 0 and value != self.ground_truth:
            self.buffer = np.empty((0, self.input_channels), dtype=np.float32)
            self.total_samples = 0
            self.last_prediction_sample = 0
        self.ground_truth = value

    def _decode(self, msg: Float32MultiArray) -> np.ndarray:
        if len(msg.layout.dim) >= 2:
            rows = int(msg.layout.dim[0].size)
            cols = int(msg.layout.dim[1].size)
        else:
            cols = self.input_channels
            rows = len(msg.data) // cols
        if rows <= 0 or cols <= 0 or rows * cols != len(msg.data):
            raise ValueError(f"Invalid Float32MultiArray shape {rows} x {cols} for {len(msg.data)} values")
        if cols != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} channels, received {cols}")
        return np.asarray(msg.data, dtype=np.float32).reshape(rows, cols)

    def _samples_cb(self, msg: Float32MultiArray) -> None:
        received_at_ns = self.get_clock().now().nanoseconds
        try:
            chunk = self._decode(msg)
            self.buffer = np.concatenate((self.buffer, chunk), axis=0)
            self.total_samples += len(chunk)
            keep = self.model.window_samples + self.model.step_samples
            if len(self.buffer) > keep:
                self.buffer = self.buffer[-keep:, :]

            first_ready = self.last_prediction_sample == 0 and self.total_samples >= self.model.window_samples
            stepped = self.total_samples - self.last_prediction_sample >= self.model.step_samples
            if not first_ready and not (self.last_prediction_sample > 0 and stepped):
                return

            prediction = self.model.predict(self.buffer[-self.model.window_samples :, :])
            self.last_prediction_sample = self.total_samples
            published_at_ns = self.get_clock().now().nanoseconds
            payload = {
                "subject_id": self.model.subject_id,
                "evaluation_mode": self.evaluation_mode,
                "fold_index": int(self.model.metadata.get("fold_index", 0)),
                "held_out_condition": str(self.model.metadata.get("held_out_condition", "")),
                "model_sha256": self.model.sha256,
                "split_manifest_sha256": str(
                    self.model.metadata.get("split_manifest_sha256", "")
                ),
                "label": prediction.label,
                "name": LABEL_NAMES[prediction.label],
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
                "ground_truth": self.ground_truth,
                "correct": self.ground_truth < 0 or prediction.label == self.ground_truth,
                "window_samples": self.model.window_samples,
                "step_samples": self.model.step_samples,
                "feature_ms": prediction.feature_ms,
                "model_ms": prediction.model_ms,
                "inference_ms": prediction.inference_ms,
                "received_at_ns": received_at_ns,
                "published_at_ns": published_at_ns,
            }
            text = String()
            text.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            self.prediction_pub.publish(text)
            label = Int32()
            label.data = prediction.label
            self.label_pub.publish(label)
        except Exception as exc:
            self.get_logger().error(f"Classification failed: {type(exc).__name__}: {exc}")


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

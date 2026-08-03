from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import rclpy
from rclpy.node import Node
import scipy.io as sio
from std_msgs.msg import Float32MultiArray, Int32, MultiArrayDimension, String

from .constants import FS, INPUT_CHANNELS, REST, STRICT_FOLD_ARTIFACT_ROLE


class EmgReplayNode(Node):
    """Streams real Limb Position segments selected by the demo controller."""

    def __init__(self) -> None:
        super().__init__("limb_emg_replay")
        self.declare_parameter("project_root", "")
        self.declare_parameter("subject_id", 1)
        self.declare_parameter("condition", "StaticP1")
        self.declare_parameter("chunk_samples", 63)
        self.declare_parameter("trim_fraction", 0.10)
        self.declare_parameter("model_path", "")
        self.declare_parameter("evaluation_mode", False)
        self.declare_parameter("fold_index", 0)
        self.declare_parameter("random_seed", 42)
        self.declare_parameter("stream_reserve_sec", 7.5)

        self.project_root = Path(str(self.get_parameter("project_root").value))
        self.subject_id = int(self.get_parameter("subject_id").value)
        self.condition = str(self.get_parameter("condition").value)
        self.chunk_samples = int(self.get_parameter("chunk_samples").value)
        self.trim_fraction = float(self.get_parameter("trim_fraction").value)
        self.evaluation_mode = bool(self.get_parameter("evaluation_mode").value)
        self.fold_index = int(self.get_parameter("fold_index").value)
        self.stream_reserve_samples = int(
            float(self.get_parameter("stream_reserve_sec").value) * FS
        )
        self.rng = np.random.default_rng(
            int(self.get_parameter("random_seed").value) + 1000 * self.subject_id + self.fold_index
        )
        if not self.project_root.is_dir():
            raise FileNotFoundError(f"project_root does not exist: {self.project_root}")

        self.model_metadata: dict = {}
        self.model_sha256 = ""
        self.model_window_samples = self.chunk_samples
        configured_model_path = str(self.get_parameter("model_path").value).strip()
        if self.evaluation_mode:
            if not configured_model_path:
                raise RuntimeError("Evaluation mode requires model_path")
            model_path = Path(configured_model_path)
            payload = joblib.load(model_path)
            self.model_metadata = dict(payload["metadata"])
            digest = hashlib.sha256()
            with model_path.open("rb") as handle:
                for digest_chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(digest_chunk)
            self.model_sha256 = digest.hexdigest()
            self.model_window_samples = int(self.model_metadata["window_length_samples"])

        self.segment_rows: dict[int, dict[str, str]] = {}
        self.segments = self._load_segments()
        if self.evaluation_mode:
            self._validate_evaluation_source()
        self.requested_label = REST
        self.stream_instance_id = 0
        self.loop_count = 0
        self.cursor = self._initial_cursor(REST)

        self.samples_pub = self.create_publisher(Float32MultiArray, "/limb/raw_window_chunks", 10)
        self.truth_pub = self.create_publisher(Int32, "/limb/ground_truth", 10)
        self.provenance_pub = self.create_publisher(String, "/limb/replay_provenance", 20)
        self.create_subscription(Int32, "/fitts/demo_requested_label", self._request_cb, 10)
        self.create_timer(self.chunk_samples / FS, self._timer_cb)

        self.get_logger().info(
            f"Loaded Subject {self.subject_id}, {self.condition}: "
            f"5 held-out real segments, fold={self.fold_index}, chunk={self.chunk_samples} samples"
        )

    def _load_segments(self) -> dict[int, np.ndarray]:
        manifest_path = self.project_root / "protocols" / "Limb" / "LIMB_RECORDING_MANIFEST.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if int(row["subject_id"]) == self.subject_id
                and row["condition"] == self.condition
                and int(row["action_label"]) in range(1, 6)
                and row["usable_status"].startswith("usable")
            ]
        if len(rows) != 5:
            raise RuntimeError(
                f"Expected 5 usable rows for Subject {self.subject_id}/{self.condition}, got {len(rows)}"
            )

        source_rel = rows[0]["source_file"].replace("\\", "/")
        source_path = self.project_root / source_rel
        if not source_path.is_file():
            raise FileNotFoundError(f"MAT file not found: {source_path}")
        limb_data = sio.loadmat(str(source_path), simplify_cells=True)["limbEMG_Data"]
        condition_array = np.asarray(limb_data[self.condition], dtype=np.float32)
        if condition_array.ndim != 2 or condition_array.shape[1] < INPUT_CHANNELS:
            raise ValueError(f"Unexpected Limb array shape: {condition_array.shape}")

        segments: dict[int, np.ndarray] = {}
        for row in rows:
            label = int(row["action_label"]) - 1
            start = int(row["start_index"]) - 1
            end_exclusive = int(row["end_index"])
            segments[label] = np.ascontiguousarray(
                condition_array[start:end_exclusive, :INPUT_CHANNELS], dtype=np.float32
            )
            self.segment_rows[label] = dict(row)
        return segments

    def _validate_evaluation_source(self) -> None:
        meta = self.model_metadata
        if meta.get("artifact_role") != STRICT_FOLD_ARTIFACT_ROLE:
            raise RuntimeError("Evaluation replay refuses a non-held-out model")
        if int(meta.get("subject_id", -1)) != self.subject_id:
            raise RuntimeError("Replay subject does not match model subject")
        if int(meta.get("fold_index", -1)) != self.fold_index:
            raise RuntimeError("Replay fold does not match model fold")
        if str(meta.get("held_out_condition", "")) != self.condition:
            raise RuntimeError(
                f"Replay condition {self.condition} is not model held-out condition "
                f"{meta.get('held_out_condition')}"
            )
        replay_ids = {str(row["recording_id"]) for row in self.segment_rows.values()}
        training_ids = {str(value) for value in meta.get("training_segment_ids", [])}
        test_ids = {str(value) for value in meta.get("test_segment_ids", [])}
        overlap = replay_ids & training_ids
        if overlap:
            raise RuntimeError(f"DATA LEAKAGE: replay segments were used for training: {sorted(overlap)}")
        if replay_ids != test_ids:
            raise RuntimeError(
                f"Replay/test manifest mismatch: replay={sorted(replay_ids)}, test={sorted(test_ids)}"
            )

    def _initial_cursor(self, label: int) -> int:
        segment = self.segments[label]
        low = min(int(len(segment) * self.trim_fraction), max(len(segment) - self.chunk_samples, 0))
        if not self.evaluation_mode:
            return low
        trimmed_end = max(low, len(segment) - int(len(segment) * self.trim_fraction))
        reserve = max(self.model_window_samples, self.stream_reserve_samples)
        high = max(low, trimmed_end - reserve)
        return int(self.rng.integers(low, high + 1)) if high > low else low

    def _request_cb(self, msg: Int32) -> None:
        label = int(msg.data)
        if label not in self.segments:
            self.get_logger().warning(f"Ignoring invalid requested label {label}")
            return
        if label != self.requested_label:
            self.requested_label = label
            self.stream_instance_id += 1
            self.loop_count = 0
            self.cursor = self._initial_cursor(label)

    def _timer_cb(self) -> None:
        segment = self.segments[self.requested_label]
        if self.cursor + self.chunk_samples > len(segment):
            self.stream_instance_id += 1
            self.loop_count += 1
            self.cursor = self._initial_cursor(self.requested_label)
        local_start = self.cursor
        chunk = segment[self.cursor : self.cursor + self.chunk_samples]
        self.cursor += self.chunk_samples

        truth = Int32()
        truth.data = self.requested_label
        self.truth_pub.publish(truth)

        msg = Float32MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(
                label="samples",
                size=int(chunk.shape[0]),
                stride=int(chunk.shape[0] * chunk.shape[1]),
            ),
            MultiArrayDimension(
                label="channels",
                size=int(chunk.shape[1]),
                stride=int(chunk.shape[1]),
            ),
        ]
        msg.data = chunk.reshape(-1).tolist()
        self.samples_pub.publish(msg)

        row = self.segment_rows[self.requested_label]
        source_start = int(row["start_index"]) + local_start
        provenance = String()
        provenance.data = json.dumps(
            {
                "evaluation_mode": self.evaluation_mode,
                "subject_id": self.subject_id,
                "fold_index": self.fold_index,
                "condition": self.condition,
                "label": self.requested_label,
                "segment_id": str(row["recording_id"]),
                "source_file": str(row["source_file"]),
                "source_start_index": source_start,
                "source_end_index": source_start + len(chunk) - 1,
                "stream_instance_id": self.stream_instance_id,
                "loop_count": self.loop_count,
                "model_sha256": self.model_sha256,
                "split_manifest_sha256": str(
                    self.model_metadata.get("split_manifest_sha256", "")
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.provenance_pub.publish(provenance)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EmgReplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import signal
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class VideoRecorderNode(Node):
    """Record the Gazebo task camera to an annotated MP4 file."""

    def __init__(self) -> None:
        super().__init__("fitts_video_recorder")
        self.declare_parameter("video_dir", "./videos")
        self.declare_parameter("subject_id", 1)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("post_roll_sec", 2.0)
        self.declare_parameter("camera_topic", "/fitts/camera/image")
        self.declare_parameter("evaluation_mode", False)
        self.declare_parameter("fold_index", 0)
        self.declare_parameter("held_out_condition", "")

        self.video_dir = Path(str(self.get_parameter("video_dir").value))
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.subject_id = int(self.get_parameter("subject_id").value)
        self.evaluation_mode = bool(self.get_parameter("evaluation_mode").value)
        self.fold_index = int(self.get_parameter("fold_index").value)
        self.held_out_condition = str(self.get_parameter("held_out_condition").value)
        self.fps = float(self.get_parameter("fps").value)
        self.post_roll_frames = max(0, int(float(self.get_parameter("post_roll_sec").value) * self.fps))
        camera_topic = str(self.get_parameter("camera_topic").value)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fold_tag = (
            f"_fold{self.fold_index:02d}_{self.held_out_condition}"
            if self.evaluation_mode
            else "_demo"
        )
        stem = f"fitts_ur5e_subject{self.subject_id:02d}{fold_tag}_{stamp}"
        self.final_path = self.video_dir / f"{stem}.mp4"
        self.temporary_path = self.video_dir / f"{stem}.recording.mp4"
        self.metadata_path = self.video_dir / f"{stem}.json"

        self.writer: cv2.VideoWriter | None = None
        self.width = 0
        self.height = 0
        self.frame_count = 0
        self.started_wall_ns = 0
        self.close_after_frames: int | None = None
        self.closed = False
        self.latest_gesture = "Waiting for prediction"
        self.latest_confidence = 0.0
        self.latest_trial = "Waiting for trial"
        self.latest_condition = "A/W pending"
        self.latest_event = "initializing"

        self.status_pub = self.create_publisher(String, "/fitts/video_status", 10)
        self.create_subscription(Image, camera_topic, self._image_cb, qos_profile_sensor_data)
        self.create_subscription(String, "/limb/prediction", self._prediction_cb, 10)
        self.create_subscription(String, "/fitts/status", self._task_status_cb, 10)
        self.create_subscription(String, "/fitts/summary", self._summary_cb, 10)
        self.get_logger().info(f"Waiting for Gazebo camera frames on {camera_topic}")

    def _prediction_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.latest_gesture = str(data.get("name", "Unknown"))
            self.latest_confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    def _task_status_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            trial = int(data.get("trial_index", 0))
            total = int(data.get("total_trials", 0))
            self.latest_trial = f"Trial {trial}/{total}"
            self.latest_event = str(data.get("event", "task"))
            distance = data.get("distance_m")
            width = data.get("width_m")
            if distance is not None and width is not None:
                self.latest_condition = (
                    f"A={100.0 * float(distance):.0f} cm  "
                    f"W={100.0 * float(width):.1f} cm"
                )
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    def _summary_cb(self, _msg: String) -> None:
        if self.close_after_frames is None:
            self.close_after_frames = self.frame_count + self.post_roll_frames
            self.latest_trial = "Experiment complete"
            self.get_logger().info(
                f"Experiment complete; recording {self.post_roll_frames} post-roll frames"
            )

    def _to_bgr(self, msg: Image) -> np.ndarray:
        encoding = msg.encoding.lower()
        channels_by_encoding = {
            "rgb8": 3,
            "bgr8": 3,
            "rgba8": 4,
            "bgra8": 4,
            "mono8": 1,
        }
        if encoding not in channels_by_encoding:
            raise ValueError(f"Unsupported camera encoding: {msg.encoding}")
        channels = channels_by_encoding[encoding]
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        expected_rows = msg.height * msg.step
        if raw.size < expected_rows:
            raise ValueError(f"Image buffer has {raw.size} bytes; expected at least {expected_rows}")
        rows = raw[:expected_rows].reshape(msg.height, msg.step)
        packed = rows[:, : msg.width * channels]
        if channels == 1:
            image = packed.reshape(msg.height, msg.width)
        else:
            image = packed.reshape(msg.height, msg.width, channels)

        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if encoding == "mono8":
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return np.ascontiguousarray(image)

    def _open_writer(self, frame: np.ndarray) -> None:
        self.height, self.width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.temporary_path),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        if not self.writer.isOpened():
            self.writer = None
            raise RuntimeError(f"OpenCV could not create MP4: {self.temporary_path}")
        self.started_wall_ns = time.perf_counter_ns()
        self.get_logger().info(
            f"Video recording started: {self.width}x{self.height} @ {self.fps:.1f} fps -> {self.final_path}"
        )
        self._publish_status("recording_started")

    def _annotate(self, frame: np.ndarray) -> None:
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (12, 12),
            (min(self.width - 12, 760), 140),
            (20, 20, 20),
            -1,
        )
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, frame)
        cv2.circle(frame, (32, 35), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (48, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(
            frame,
            (
                f"S{self.subject_id:02d} Fold {self.fold_index}/8 {self.held_out_condition} | "
                f"{self.latest_trial}"
                if self.evaluation_mode
                else f"Subject {self.subject_id:02d} DEMO | {self.latest_trial}"
            ),
            (22, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"{self.latest_condition}  |  {self.latest_event}",
            (22, 99),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (120, 255, 150),
            2,
        )
        cv2.putText(
            frame,
            f"Gesture: {self.latest_gesture}  confidence={self.latest_confidence:.3f}",
            (22, 126),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 230, 255),
            2,
        )

    def _image_cb(self, msg: Image) -> None:
        if self.closed:
            return
        try:
            frame = self._to_bgr(msg)
            if self.writer is None:
                self._open_writer(frame)
            self._annotate(frame)
            assert self.writer is not None
            self.writer.write(frame)
            self.frame_count += 1
            if self.close_after_frames is not None and self.frame_count >= self.close_after_frames:
                self.close("experiment_complete")
        except Exception as exc:
            self.get_logger().error(f"Video frame failed: {type(exc).__name__}: {exc}")

    def _publish_status(self, event: str) -> None:
        if not rclpy.ok():
            return
        msg = String()
        msg.data = json.dumps(
            {
                "event": event,
                "path": str(self.final_path),
                "frames": self.frame_count,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
            },
            ensure_ascii=False,
        )
        self.status_pub.publish(msg)

    def close(self, reason: str = "shutdown") -> None:
        if self.closed:
            return
        self.closed = True
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.frame_count > 0 and self.temporary_path.is_file():
            self.temporary_path.replace(self.final_path)
        if self.frame_count > 0 and self.final_path.is_file():
            elapsed = (time.perf_counter_ns() - self.started_wall_ns) / 1e9
            metadata = {
                "video_path": str(self.final_path),
                "subject_id": self.subject_id,
                "evaluation_mode": self.evaluation_mode,
                "fold_index": self.fold_index,
                "held_out_condition": self.held_out_condition,
                "frame_count": self.frame_count,
                "fps": self.fps,
                "resolution": [self.width, self.height],
                "nominal_video_duration_sec": self.frame_count / self.fps,
                "recording_wall_duration_sec": elapsed,
                "close_reason": reason,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            self.metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.get_logger().info(
                f"Video saved: {self.final_path} ({self.frame_count} frames, {self.frame_count / self.fps:.2f} s)"
            )
            self._publish_status("recording_saved")
        else:
            self.get_logger().warning("No camera frames were received; no video was created")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VideoRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # ros2 launch and an outer timeout may both forward SIGINT.  Ignore a
        # second signal while OpenCV flushes the MP4 index and metadata file.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.close("shutdown")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

from geometry_msgs.msg import TwistStamped
from moveit_msgs.srv import ServoCommandType
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Int32, String
from tf2_ros import Buffer, TransformException, TransformListener

from .constants import LABEL_NAMES, LABEL_TO_XY, REST


class GestureServoNode(Node):
    """Maps filtered gesture predictions to planar MoveIt Servo twists."""

    def __init__(self) -> None:
        super().__init__("gesture_servo")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("linear_speed_mps", 0.055)
        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("confidence_threshold", 0.50)
        self.declare_parameter("stable_predictions", 1)
        self.declare_parameter("prediction_timeout_sec", 1.2)
        self.declare_parameter("workspace_x", [-0.85, 0.85])
        self.declare_parameter("workspace_y", [-0.85, 0.85])

        self.base_frame = str(self.get_parameter("base_frame").value)
        self.ee_frame = str(self.get_parameter("ee_frame").value)
        self.speed = float(self.get_parameter("linear_speed_mps").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.stable_predictions = int(self.get_parameter("stable_predictions").value)
        self.timeout_ns = int(float(self.get_parameter("prediction_timeout_sec").value) * 1e9)
        self.workspace_x = [float(x) for x in self.get_parameter("workspace_x").value]
        self.workspace_y = [float(x) for x in self.get_parameter("workspace_y").value]

        callback_group = ReentrantCallbackGroup()
        self.twist_pub = self.create_publisher(
            TwistStamped,
            "/servo_node/delta_twist_cmds",
            10,
            callback_group=callback_group,
        )
        self.command_type_client = self.create_client(
            ServoCommandType,
            "/servo_node/switch_command_type",
            callback_group=callback_group,
        )
        self.twist_mode_ready = False
        self.twist_mode_request_pending = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.candidate_label = REST
        self.candidate_count = 0
        self.applied_label = REST
        self.latest_prediction: dict = {}
        self.last_prediction_ns = 0

        self.applied_pub = self.create_publisher(Int32, "/limb/applied_label", 10)
        self.event_pub = self.create_publisher(String, "/fitts/control_event", 10)
        self.create_subscription(
            String,
            "/limb/prediction",
            self._prediction_cb,
            10,
            callback_group=callback_group,
        )
        rate = float(self.get_parameter("control_rate_hz").value)
        self.create_timer(1.0 / rate, self._control_cb, callback_group=callback_group)

        self.get_logger().info(
            "Gesture mapping: HandOpen=+X, Lateral=-X, Pinch=+Y, Power=-Y, Rest=stop/confirm"
        )

    def _ensure_twist_mode(self) -> None:
        if self.twist_mode_ready or self.twist_mode_request_pending:
            return
        if not self.command_type_client.service_is_ready():
            return
        request = ServoCommandType.Request()
        request.command_type = ServoCommandType.Request.TWIST
        self.twist_mode_request_pending = True
        future = self.command_type_client.call_async(request)

        def done_callback(done_future) -> None:
            self.twist_mode_request_pending = False
            try:
                self.twist_mode_ready = bool(done_future.result().success)
                if self.twist_mode_ready:
                    self.get_logger().info("MoveIt Servo command type switched to TWIST")
                else:
                    self.get_logger().error("MoveIt Servo rejected TWIST command mode")
            except Exception as exc:
                self.get_logger().warning(f"Could not switch Servo command type: {exc}")

        future.add_done_callback(done_callback)

    def _prediction_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            raw_label = int(data["label"])
            confidence = float(data["confidence"])
            candidate = raw_label if confidence >= self.confidence_threshold else REST
            if candidate == self.candidate_label:
                self.candidate_count += 1
            else:
                self.candidate_label = candidate
                self.candidate_count = 1
            if self.candidate_count >= self.stable_predictions:
                self.applied_label = candidate
            self.latest_prediction = data
            self.last_prediction_ns = self.get_clock().now().nanoseconds
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Bad prediction message: {exc}")

    def _inside_workspace(self, dx: float, dy: float) -> bool:
        try:
            transform = self.tf_buffer.lookup_transform(self.base_frame, self.ee_frame, rclpy.time.Time())
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            margin = 0.01
            if dx > 0 and x >= self.workspace_x[1] - margin:
                return False
            if dx < 0 and x <= self.workspace_x[0] + margin:
                return False
            if dy > 0 and y >= self.workspace_y[1] - margin:
                return False
            if dy < 0 and y <= self.workspace_y[0] + margin:
                return False
        except TransformException:
            pass
        return True

    def _control_cb(self) -> None:
        self._ensure_twist_mode()
        now_ns = self.get_clock().now().nanoseconds
        stale = self.last_prediction_ns == 0 or now_ns - self.last_prediction_ns > self.timeout_ns
        label = REST if stale else self.applied_label
        dx, dy = LABEL_TO_XY[label]
        if not self._inside_workspace(dx, dy):
            dx = dy = 0.0
            label = REST

        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = self.base_frame
        twist.twist.linear.x = dx * self.speed
        twist.twist.linear.y = dy * self.speed
        self.twist_pub.publish(twist)

        applied = Int32()
        applied.data = label
        self.applied_pub.publish(applied)

        prediction_stamp = int(self.latest_prediction.get("published_at_ns", 0))
        event = {
            "label": label,
            "name": LABEL_NAMES[label],
            "stale_prediction": stale,
            "prediction_id": prediction_stamp,
            "control_at_ns": now_ns,
            "control_latency_ms": (now_ns - prediction_stamp) / 1e6 if prediction_stamp else None,
            "vx": dx * self.speed,
            "vy": dy * self.speed,
        }
        text = String()
        text.data = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        self.event_pub.publish(text)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GestureServoNode()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

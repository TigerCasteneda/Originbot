#!/usr/bin/env python3
import json

import rclpy
from geometry_msgs.msg import Point, Twist
from rclpy.node import Node
from std_msgs.msg import String


class LaneDetection(Node):
    def __init__(self):
        super().__init__("lanedetection")
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.debug_pub = self.create_publisher(String, "/debug/lane_state", 10)
        self.point_sub = self.create_subscription(
            Point, "/line_point", self.point_callback, 10
        )
        self.point = None
        self.last_point_time = None

        # Experiment 8 baseline: direct proportional control from /line_point to /cmd_vel.
        self.declare_parameter("target_x", 1130.0)
        self.declare_parameter("base_speed", 0.15)
        self.declare_parameter("turn_gain", 0.0020)
        self.declare_parameter("max_angular_speed", 0.35)
        self.declare_parameter("point_timeout", 0.5)

        self.target_x = float(self.get_parameter("target_x").value)
        self.base_speed = float(self.get_parameter("base_speed").value)
        self.turn_gain = float(self.get_parameter("turn_gain").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.min_valid_x = 0.0
        self.max_valid_x = 1920.0
        self.min_valid_y = 0.0
        self.max_valid_y = 1080.0
        self.point_timeout = float(self.get_parameter("point_timeout").value)

        self.create_timer(0.1, self.timer_callback)
        self.get_logger().info(
            "lanedetection baseline node started "
            f"(target_x={self.target_x:.1f}, base_speed={self.base_speed:.3f}, "
            f"turn_gain={self.turn_gain:.4f}, max_w={self.max_angular_speed:.3f})"
        )

    def point_callback(self, msg: Point):
        self.point = msg
        self.last_point_time = self.get_clock().now()
        self.publish_cmd()

    def timer_callback(self):
        if self.last_point_time is not None:
            age = (self.get_clock().now() - self.last_point_time).nanoseconds / 1e9
            if age > self.point_timeout:
                self.point = None
        self.publish_cmd()

    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def valid_point(self):
        if self.point is None:
            return False
        return (
            self.min_valid_x <= self.point.x <= self.max_valid_x
            and self.min_valid_y <= self.point.y <= self.max_valid_y
        )

    def publish_cmd(self):
        twist = Twist()
        state = {
            "mode": "TRACK",
            "point_valid": False,
            "point_x": None,
            "point_y": None,
            "target_x": self.target_x,
            "speed": 0.0,
            "angular_z": 0.0,
            "reason": "no_point",
        }

        if self.valid_point():
            error_x = float(self.point.x - self.target_x)
            angular_z = self.clamp(
                -self.turn_gain * error_x,
                -self.max_angular_speed,
                self.max_angular_speed,
            )
            twist.linear.x = self.base_speed
            twist.angular.z = angular_z
            state.update(
                {
                    "point_valid": True,
                    "point_x": float(self.point.x),
                    "point_y": float(self.point.y),
                    "error_x": error_x,
                    "speed": float(twist.linear.x),
                    "angular_z": float(twist.angular.z),
                    "reason": "tracking",
                }
            )

        self.cmd_vel_pub.publish(twist)
        debug_msg = String()
        debug_msg.data = json.dumps(state, ensure_ascii=False)
        self.debug_pub.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


class ObstacleDetection(Node):
    def __init__(self):
        super().__init__("obstacle_detection")
        self.obstacle_pub = self.create_publisher(Bool, "/has_obs", 10)
        self.info_pub = self.create_publisher(String, "/obstacle_info", 10)
        self.debug_pub = self.create_publisher(String, "/debug/obstacle_state", 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos_profile_sensor_data,
        )

        self.declare_parameter("min_range", 0.22)
        self.declare_parameter("trigger_range", 0.50)
        self.declare_parameter("clear_range", 0.65)
        self.declare_parameter("slow_range", 0.75)
        self.declare_parameter("side_block_range", 0.35)
        self.declare_parameter("front_angle", 32.0)
        self.declare_parameter("side_angle_min", 20.0)
        self.declare_parameter("side_angle_max", 85.0)
        self.declare_parameter("front_hit_threshold", 8)
        self.declare_parameter("detect_confirm_scans", 2)
        self.declare_parameter("clear_confirm_scans", 3)
        self.declare_parameter("escape_check_range", 0.75)

        self.detect_count = 0
        self.clear_count = 0
        self.has_obs_latched = False
        self.last_state = None
        self.get_logger().info("obstacle detection node started")

    def _params(self):
        return {
            "min_range": float(self.get_parameter("min_range").value),
            "trigger_range": float(self.get_parameter("trigger_range").value),
            "clear_range": float(self.get_parameter("clear_range").value),
            "slow_range": float(self.get_parameter("slow_range").value),
            "side_block_range": float(self.get_parameter("side_block_range").value),
            "front_angle": math.radians(float(self.get_parameter("front_angle").value)),
            "side_angle_min": math.radians(float(self.get_parameter("side_angle_min").value)),
            "side_angle_max": math.radians(float(self.get_parameter("side_angle_max").value)),
            "front_hit_threshold": int(self.get_parameter("front_hit_threshold").value),
            "detect_confirm_scans": int(self.get_parameter("detect_confirm_scans").value),
            "clear_confirm_scans": int(self.get_parameter("clear_confirm_scans").value),
            "escape_check_range": float(self.get_parameter("escape_check_range").value),
        }

    def scan_callback(self, scan: LaserScan):
        p = self._params()
        nearest_range = None
        nearest_angle_deg = None
        front_min_dist = None
        left_min_dist = None
        right_min_dist = None
        front_hits = 0
        left_hits = 0
        right_hits = 0
        obs_left_edge = None
        obs_right_edge = None
        left_escape_blocked = False
        right_escape_blocked = False
        angle = scan.angle_min

        for distance in scan.ranges:
            if math.isfinite(distance) and distance >= p["min_range"]:
                angle_deg = math.degrees(angle)

                if nearest_range is None or distance < nearest_range:
                    nearest_range = distance
                    nearest_angle_deg = angle_deg

                if abs(angle) <= p["front_angle"]:
                    if front_min_dist is None or distance < front_min_dist:
                        front_min_dist = distance
                    if distance < p["trigger_range"]:
                        front_hits += 1

                if abs(angle_deg) <= 90.0 and distance < p["trigger_range"]:
                    if obs_left_edge is None or angle_deg < obs_left_edge:
                        obs_left_edge = angle_deg
                    if obs_right_edge is None or angle_deg > obs_right_edge:
                        obs_right_edge = angle_deg

                if p["side_angle_min"] <= angle <= p["side_angle_max"]:
                    if left_min_dist is None or distance < left_min_dist:
                        left_min_dist = distance
                    if distance < p["side_block_range"]:
                        left_hits += 1
                    if distance < p["escape_check_range"]:
                        left_escape_blocked = True

                if -p["side_angle_max"] <= angle <= -p["side_angle_min"]:
                    if right_min_dist is None or distance < right_min_dist:
                        right_min_dist = distance
                    if distance < p["side_block_range"]:
                        right_hits += 1
                    if distance < p["escape_check_range"]:
                        right_escape_blocked = True
            angle += scan.angle_increment

        front_blocked = (
            front_min_dist is not None
            and front_min_dist < p["trigger_range"]
            and front_hits >= p["front_hit_threshold"]
        )
        front_cleared = (
            front_min_dist is None
            or front_min_dist > p["clear_range"]
            or front_hits == 0
        )

        if front_blocked:
            self.detect_count += 1
            self.clear_count = 0
        elif front_cleared:
            self.clear_count += 1
            self.detect_count = 0
        else:
            self.detect_count = 0
            self.clear_count = 0

        if not self.has_obs_latched and self.detect_count >= p["detect_confirm_scans"]:
            self.has_obs_latched = True
        elif self.has_obs_latched and self.clear_count >= p["clear_confirm_scans"]:
            self.has_obs_latched = False

        has_obs = self.has_obs_latched
        left_clearance = left_min_dist if left_min_dist is not None else 99.0
        right_clearance = right_min_dist if right_min_dist is not None else 99.0

        left_blocked = left_min_dist is not None and left_min_dist < p["side_block_range"]
        right_blocked = right_min_dist is not None and right_min_dist < p["side_block_range"]

        if has_obs:
            if left_blocked and not right_blocked:
                recommended_side = "right"
            elif right_blocked and not left_blocked:
                recommended_side = "left"
            else:
                recommended_side = "left" if left_clearance >= right_clearance else "right"
        else:
            recommended_side = "none"

        escape_clearance_ok = False
        if recommended_side == "left":
            escape_clearance_ok = not left_escape_blocked
        elif recommended_side == "right":
            escape_clearance_ok = not right_escape_blocked

        obs_angular_span = None
        if obs_left_edge is not None and obs_right_edge is not None:
            obs_angular_span = obs_right_edge - obs_left_edge

        msg = Bool()
        msg.data = has_obs
        self.obstacle_pub.publish(msg)

        info = {
            "has_obs": has_obs,
            "nearest_range": nearest_range,
            "nearest_angle_deg": nearest_angle_deg,
            "front_min_dist": front_min_dist,
            "left_min_dist": left_min_dist,
            "right_min_dist": right_min_dist,
            "front_hits": front_hits,
            "left_hits": left_hits,
            "right_hits": right_hits,
            "front_blocked": front_blocked,
            "left_blocked": left_blocked,
            "right_blocked": right_blocked,
            "recommended_side": recommended_side,
            "obs_left_edge_deg": obs_left_edge,
            "obs_right_edge_deg": obs_right_edge,
            "obs_angular_span_deg": obs_angular_span,
            "escape_clearance_ok": escape_clearance_ok,
            "min_range": p["min_range"],
            "trigger_range": p["trigger_range"],
            "clear_range": p["clear_range"],
            "slow_range": p["slow_range"],
            "side_block_range": p["side_block_range"],
            "front_angle_deg": math.degrees(p["front_angle"]),
        }
        info_msg = String()
        info_msg.data = json.dumps(info, ensure_ascii=False)
        self.info_pub.publish(info_msg)

        debug = String()
        debug.data = info_msg.data
        self.debug_pub.publish(debug)

        if has_obs != self.last_state:
            state = "detected" if has_obs else "cleared"
            self.get_logger().info(f"obstacle {state}")
            self.last_state = has_obs


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

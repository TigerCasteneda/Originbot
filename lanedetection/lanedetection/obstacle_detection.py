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

        self.min_range = 0.12
        self.trigger_range = 0.30
        self.clear_range = 0.38
        self.side_block_range = 0.40
        self.front_angle = math.radians(18.0)
        self.side_angle_min = math.radians(35.0)
        self.side_angle_max = math.radians(85.0)
        self.front_hit_threshold = 5
        self.detect_confirm_scans = 2
        self.clear_confirm_scans = 2
        self.detect_count = 0
        self.clear_count = 0
        self.has_obs_latched = False
        self.last_state = None
        self.get_logger().info("obstacle detection node started")

    def scan_callback(self, scan: LaserScan):
        nearest_range = None
        nearest_angle_deg = None
        front_min_dist = None
        left_min_dist = None
        right_min_dist = None
        front_hits = 0
        left_hits = 0
        right_hits = 0
        angle = scan.angle_min

        for distance in scan.ranges:
            if math.isfinite(distance) and distance >= self.min_range:
                if nearest_range is None or distance < nearest_range:
                    nearest_range = distance
                    nearest_angle_deg = math.degrees(angle)

                if abs(angle) <= self.front_angle:
                    if front_min_dist is None or distance < front_min_dist:
                        front_min_dist = distance
                    if distance < self.trigger_range:
                        front_hits += 1

                if self.side_angle_min <= angle <= self.side_angle_max:
                    if left_min_dist is None or distance < left_min_dist:
                        left_min_dist = distance
                    if distance < self.side_block_range:
                        left_hits += 1

                if -self.side_angle_max <= angle <= -self.side_angle_min:
                    if right_min_dist is None or distance < right_min_dist:
                        right_min_dist = distance
                    if distance < self.side_block_range:
                        right_hits += 1
            angle += scan.angle_increment

        front_blocked = (
            front_min_dist is not None
            and front_min_dist < self.trigger_range
            and front_hits >= self.front_hit_threshold
        )
        front_cleared = (
            front_min_dist is None
            or front_min_dist > self.clear_range
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

        if not self.has_obs_latched and self.detect_count >= self.detect_confirm_scans:
            self.has_obs_latched = True
        elif self.has_obs_latched and self.clear_count >= self.clear_confirm_scans:
            self.has_obs_latched = False

        has_obs = self.has_obs_latched
        left_blocked = left_min_dist is not None and left_min_dist < self.side_block_range
        right_blocked = (
            right_min_dist is not None and right_min_dist < self.side_block_range
        )
        left_clearance = left_min_dist if left_min_dist is not None else 99.0
        right_clearance = right_min_dist if right_min_dist is not None else 99.0
        if has_obs:
            if left_blocked and not right_blocked:
                recommended_side = "right"
            elif right_blocked and not left_blocked:
                recommended_side = "left"
            else:
                recommended_side = (
                    "left" if left_clearance >= right_clearance else "right"
                )
        else:
            recommended_side = "none"

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
            "min_range": self.min_range,
            "trigger_range": self.trigger_range,
            "clear_range": self.clear_range,
            "side_block_range": self.side_block_range,
            "front_angle_deg": math.degrees(self.front_angle),
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

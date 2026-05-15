#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ViewLidar(Node):
    def __init__(self):
        super().__init__("view_lidar")
        self.min_range = 0.12
        self.max_range = 0.30
        self.front_angle = math.radians(30.0)
        self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data
        )
        self.get_logger().info("LiDAR viewer started")

    def scan_callback(self, scan: LaserScan):
        angle = scan.angle_min
        hits = []
        for index, distance in enumerate(scan.ranges):
            if (
                math.isfinite(distance)
                and abs(angle) <= self.front_angle
                and self.min_range < distance < self.max_range
            ):
                hits.append(
                    f"ranges[{index}]={distance:.3f} angle={math.degrees(angle):.1f}deg"
                )
            angle += scan.angle_increment

        if hits:
            print("\n".join(hits[:12]))


def main(args=None):
    rclpy.init(args=args)
    node = ViewLidar()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

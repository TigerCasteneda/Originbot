#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


class ObstacleDetection(Node):
    def __init__(self):
        super().__init__('obstacle_detection')
        self.get_logger().info('Start obstacle detection.')

        self.min_distance = 0.05
        self.trigger_forward_distance = 0.55
        self.release_forward_distance = 0.72
        self.slow_forward_distance = 0.82
        self.corridor_half_width = 0.22
        self.front_angle_deg = 42.0
        self.max_considered_angle_deg = 160.0
        self.side_angle_min_deg = 20.0
        self.side_angle_max_deg = 90.0
        self.side_block_distance = 0.32
        self.escape_check_distance = 0.65
        self.obs_track_max_distance = 0.45
        self.trigger_frames_required = 1
        self.clear_frames_required = 3
        self.front_hit_threshold = 1

        self.has_obs = False
        self.last_has_obs = None
        self.last_recommended_side = 'none'
        self.trigger_count = 0
        self.clear_count = 0

        self.obs_pub = self.create_publisher(Bool, '/has_obs', 10)
        self.info_pub = self.create_publisher(String, '/obstacle_info', 10)
        self.debug_pub = self.create_publisher(String, '/debug/obstacle_info', 10)
        self.laser_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            qos_profile_sensor_data,
        )

    def laser_callback(self, scan_data):
        nearest_distance = float('inf')
        nearest_angle_deg = 0.0
        front_min_dist = None
        left_min_dist = None
        right_min_dist = None
        front_hits = 0
        left_hits = 0
        right_hits = 0
        obs_left_edge_deg = None
        obs_right_edge_deg = None
        left_escape_blocked = False
        right_escape_blocked = False

        for index, distance in enumerate(scan_data.ranges):
            if math.isnan(distance) or math.isinf(distance) or distance <= 0.0:
                continue

            angle = scan_data.angle_min + index * scan_data.angle_increment
            angle_deg = math.degrees(angle)
            if abs(angle_deg) > self.max_considered_angle_deg:
                continue

            forward = distance * math.cos(angle)
            lateral = distance * math.sin(angle)
            if forward <= 0.0 or distance < self.min_distance:
                continue

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_angle_deg = angle_deg

            if abs(angle_deg) <= self.front_angle_deg and abs(lateral) <= self.corridor_half_width:
                if front_min_dist is None or forward < front_min_dist:
                    front_min_dist = forward
                if forward < self.trigger_forward_distance:
                    front_hits += 1

            if abs(angle_deg) <= 90.0 and abs(lateral) <= (self.corridor_half_width * 2.2):
                if forward < self.release_forward_distance:
                    if obs_left_edge_deg is None or angle_deg < obs_left_edge_deg:
                        obs_left_edge_deg = angle_deg
                    if obs_right_edge_deg is None or angle_deg > obs_right_edge_deg:
                        obs_right_edge_deg = angle_deg

            if self.side_angle_min_deg <= angle_deg <= self.side_angle_max_deg:
                if left_min_dist is None or distance < left_min_dist:
                    left_min_dist = distance
                if distance < self.side_block_distance:
                    left_hits += 1
                if distance < self.escape_check_distance:
                    left_escape_blocked = True

            if -self.side_angle_max_deg <= angle_deg <= -self.side_angle_min_deg:
                if right_min_dist is None or distance < right_min_dist:
                    right_min_dist = distance
                if distance < self.side_block_distance:
                    right_hits += 1
                if distance < self.escape_check_distance:
                    right_escape_blocked = True

        nearest_range = None if math.isinf(nearest_distance) else nearest_distance
        front_blocked = (
            front_min_dist is not None
            and front_min_dist < self.trigger_forward_distance
            and front_hits >= self.front_hit_threshold
        )
        front_cleared = (
            front_min_dist is None
            or front_min_dist > self.release_forward_distance
            or front_hits == 0
        )

        if self.has_obs:
            if front_cleared:
                self.clear_count += 1
            else:
                self.clear_count = 0
            if self.clear_count >= self.clear_frames_required:
                self.has_obs = False
                self.clear_count = 0
                self.trigger_count = 0
        else:
            if front_blocked:
                self.trigger_count += 1
            else:
                self.trigger_count = 0
            if self.trigger_count >= self.trigger_frames_required:
                self.has_obs = True
                self.trigger_count = 0
                self.clear_count = 0

        left_blocked = left_min_dist is not None and left_min_dist < self.side_block_distance
        right_blocked = right_min_dist is not None and right_min_dist < self.side_block_distance

        left_clearance = left_min_dist if left_min_dist is not None else 99.0
        right_clearance = right_min_dist if right_min_dist is not None else 99.0

        if self.has_obs:
            if left_blocked and not right_blocked:
                recommended_side = 'right'
            elif right_blocked and not left_blocked:
                recommended_side = 'left'
            else:
                recommended_side = 'left' if left_clearance >= right_clearance else 'right'
        else:
            recommended_side = 'none'

        escape_clearance_ok = False
        if recommended_side == 'left':
            escape_clearance_ok = not left_escape_blocked
        elif recommended_side == 'right':
            escape_clearance_ok = not right_escape_blocked

        obs_angular_span_deg = None
        if obs_left_edge_deg is not None and obs_right_edge_deg is not None:
            obs_angular_span_deg = obs_right_edge_deg - obs_left_edge_deg

        obs_track_valid = nearest_range is not None and nearest_range <= self.obs_track_max_distance
        obs_bearing_deg = None if nearest_range is None else nearest_angle_deg
        obs_bearing = None if obs_bearing_deg is None else math.radians(obs_bearing_deg)
        obs_range = nearest_range

        bool_msg = Bool()
        bool_msg.data = self.has_obs
        self.obs_pub.publish(bool_msg)

        info = {
            'has_obs': self.has_obs,
            'nearest_range': nearest_range,
            'nearest_angle_deg': None if nearest_range is None else nearest_angle_deg,
            'front_min_dist': front_min_dist,
            'left_min_dist': left_min_dist,
            'right_min_dist': right_min_dist,
            'front_hits': front_hits,
            'left_hits': left_hits,
            'right_hits': right_hits,
            'front_blocked': front_blocked,
            'left_blocked': left_blocked,
            'right_blocked': right_blocked,
            'recommended_side': recommended_side,
            'escape_clearance_ok': escape_clearance_ok,
            'obs_left_edge_deg': obs_left_edge_deg,
            'obs_right_edge_deg': obs_right_edge_deg,
            'obs_angular_span_deg': obs_angular_span_deg,
            'obs_bearing': obs_bearing,
            'obs_bearing_deg': obs_bearing_deg,
            'obs_range': obs_range,
            'obs_track_valid': obs_track_valid,
            'obs_cluster_id': 1 if obs_track_valid else None,
            'slow_range': self.slow_forward_distance,
            'trigger_range': self.trigger_forward_distance,
            'clear_range': self.release_forward_distance,
            'side_block_range': self.side_block_distance,
        }

        info_msg = String()
        info_msg.data = json.dumps(info, ensure_ascii=False)
        self.info_pub.publish(info_msg)
        self.debug_pub.publish(info_msg)

        if self.has_obs != self.last_has_obs:
            if self.has_obs:
                self.get_logger().warn(
                    'Obstacle detected: '
                    f'front={front_min_dist if front_min_dist is not None else -1.0:.3f}m '
                    f'bearing={obs_bearing_deg if obs_bearing_deg is not None else 0.0:.1f}deg '
                    f'side={recommended_side}'
                )
            else:
                self.get_logger().info('Obstacle cleared.')
            self.last_has_obs = self.has_obs

        if self.has_obs and recommended_side != self.last_recommended_side:
            self.get_logger().info(
                f'Obstacle recommendation update: side={recommended_side} '
                f'left={left_min_dist if left_min_dist is not None else -1.0:.3f}m '
                f'right={right_min_dist if right_min_dist is not None else -1.0:.3f}m'
            )
            self.last_recommended_side = recommended_side
        elif not self.has_obs:
            self.last_recommended_side = 'none'


def main(args=None):
    rclpy.init(args=args)
    obstacle_detection = ObstacleDetection()
    try:
        rclpy.spin(obstacle_detection)
    except KeyboardInterrupt:
        pass
    finally:
        obstacle_detection.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

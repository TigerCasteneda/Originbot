#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
from collections import deque

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool, String


class LaneDetection(Node):
    MODE_TRACK = 'track'
    MODE_AVOID_APPROACH = 'avoid_approach'
    MODE_AVOID_TURN_LEFT = 'avoid_turn_left'
    MODE_AVOID_FORWARD_1 = 'avoid_forward_1'
    MODE_AVOID_TURN_RIGHT = 'avoid_turn_right'
    MODE_AVOID_ORBIT_SEARCH = 'avoid_orbit_search'
    MODE_AVOID_HANDOFF = 'avoid_handoff'
    MODE_AVOID_FAILSAFE = 'avoid_failsafe'

    def __init__(self):
        super().__init__('lanedetection')
        self.get_logger().info('Start lane keeping.')

        self.image_width = 1920.0
        self.center_x = self.image_width / 2.0
        self.deadband_pixels = 55.0
        self.kp = 0.0015
        self.ki = 0.0015
        self.kd = 0.0005
        self.integral_limit = 600.0
        self.max_angular_z = 2.3
        self.min_linear_x = 0.25
        self.max_linear_x = 0.5
        self.cmd_timeout = Duration(seconds=0.5)
        self.control_period = 0.05
        self.error_filter_alpha = 0.30
        self.derivative_filter_alpha = 0.20
        self.linear_blend = 0.24
        self.angular_blend = 0.22
        self.max_linear_accel = 0.35
        self.max_linear_decel = 0.55
        self.max_angular_accel = 3.0

        self.avoid_max_linear_accel = 0.16
        self.avoid_max_linear_decel = 0.20
        self.avoid_max_angular_accel = 1.20
        self.avoid_turn_angle = math.pi / 2.0
        self.avoid_side_distance = 0.05
        self.avoid_forward_linear_x = 0.14
        self.avoid_yaw_tolerance = 0.08
        self.avoid_yaw_kp = 2.6
        self.avoid_yaw_min_angular_z = 0.45
        self.avoid_yaw_max_angular_z = 1.35
        self.avoid_turn_timeout = 3.0
        self.avoid_forward_timeout = 4.0
        self.avoid_orbit_timeout = 6.0
        self.avoid_handoff_timeout = 5.0
        self.avoid_handoff_duration = 0.70
        self.avoid_handoff_max_linear_x = 0.18
        self.approach_trigger_margin = 0.10
        self.approach_min_linear_x = 0.06
        self.approach_max_linear_x = 0.15

        self.orbit_search_linear_x = 0.11
        self.orbit_close_linear_x = 0.08
        self.orbit_target_bearing_abs = math.radians(135.0)
        self.orbit_target_range = 0.30
        self.orbit_omega_base_min = 0.10
        self.orbit_omega_base_max = 0.62
        self.orbit_k_bearing_min = 0.65
        self.orbit_k_bearing_max = 1.55
        self.orbit_k_range_min = 0.35
        self.orbit_k_range_max = 1.20
        self.orbit_k_damp_min = 0.28
        self.orbit_k_damp_max = 0.95
        self.orbit_bearing_error_limit = math.radians(85.0)
        self.orbit_range_error_limit = 0.18
        self.orbit_max_angular_z_min = 0.35
        self.orbit_max_angular_z_max = 1.35
        self.orbit_front_turn_boost_angle = math.radians(25.0)
        self.orbit_front_turn_boost_dist = 0.24
        self.orbit_front_turn_boost = 0.22
        self.orbit_obs_bearing_stable_threshold = math.radians(8.0)
        self.orbit_obs_range_stable_threshold = 0.06
        self.orbit_required_consistent_frames = 3
        self.orbit_required_obs_stable_frames = 3
        self.orbit_safe_front_margin = 0.04

        self.track_resume_duration = 1.00
        self.track_resume_speed_scale = 0.60
        self.track_resume_omega_limit = 1.00
        self.obs_cooldown_duration = 1.20

        self.point_trend_min_y = 250.0
        self.point_capture_min_y = 310.0
        self.point_trend_window = 4
        self.point_capture_window = 3
        self.point_min_signed_delta = 5.0
        self.point_jitter_limit = 130.0
        self.capture_threshold = 300.0
        self.capture_omega_threshold = 1.15
        self.handoff_threshold = 240.0
        self.handoff_omega_threshold = 0.95
        self.handoff_alpha_ramp_frames = 6
        self.handoff_obstacle_safe_bearing = math.radians(100.0)

        self.has_point = False
        self.has_obs = False
        self.has_odom = False
        self.mode = self.MODE_TRACK
        self.mode_start_time = self.get_clock().now()
        self.mode_start_x = 0.0
        self.mode_start_y = 0.0
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.avoid_origin_yaw = 0.0
        self.orbit_direction = -1.0
        self.orbit_target_bearing = -self.orbit_target_bearing_abs
        self.track_resume_until = None
        self.obs_cooldown_until = None
        self.last_point_time = self.get_clock().now()
        self.last_control_time = None
        self.last_motion_time = None
        self.last_state_debug_time = 0.0
        self.last_state_debug_signature = None
        self.latest_point_x = self.center_x
        self.latest_point_y = 0.0
        self.e_last = 0.0
        self.ie = 0.0
        self.filtered_error = 0.0
        self.filtered_derivative = 0.0

        self.obs_info = None
        self.prev_obs_bearing = None
        self.prev_obs_range = None
        self.obs_track_valid_frames = 0
        self.obs_bearing_stable_frames = 0
        self.obs_range_stable_frames = 0
        self.point_fresh_frames = 0
        self.point_consistent_frames = 0
        self.handoff_stable_frames = 0
        self.point_history = deque(maxlen=8)

        self.point_sub = self.create_subscription(
            PointStamped,
            '/line_track_center_detection',
            self.point_callback,
            10,
        )
        self.obs_sub = self.create_subscription(
            Bool,
            '/has_obs',
            self.obs_bool_callback,
            10,
        )
        self.obs_info_sub = self.create_subscription(
            String,
            '/obstacle_info',
            self.obs_info_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10,
        )
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.debug_pub = self.create_publisher(String, '/debug/lane_state', 10)
        self.watchdog_timer = self.create_timer(self.control_period, self.watchdog_callback)

        self.twist = Twist()

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def reset_pid_state(self):
        self.e_last = 0.0
        self.ie = 0.0
        self.filtered_error = 0.0
        self.filtered_derivative = 0.0
        self.last_control_time = None

    def reset_reacquire_state(self):
        self.point_history.clear()
        self.prev_obs_bearing = None
        self.prev_obs_range = None
        self.obs_track_valid_frames = 0
        self.obs_bearing_stable_frames = 0
        self.obs_range_stable_frames = 0
        self.point_fresh_frames = 0
        self.point_consistent_frames = 0
        self.handoff_stable_frames = 0

    def on_shutdown(self):
        self.reset_pid_state()
        self.stop_vehicle(immediate=True)

    def stop_vehicle(self, immediate=False):
        if immediate:
            self.twist.linear.x = 0.0
            self.twist.linear.y = 0.0
            self.twist.linear.z = 0.0
            self.twist.angular.x = 0.0
            self.twist.angular.y = 0.0
            self.twist.angular.z = 0.0
            self.last_motion_time = None
            self.cmd_vel_pub.publish(self.twist)
            return
        self.publish_motion(0.0, 0.0)

    def clamp(self, value, lower, upper):
        return max(lower, min(upper, value))

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def angle_error(self, target, current):
        return self.normalize_angle(target - current)

    def safe_obs_value(self, key, default=None):
        if self.obs_info is None:
            return default
        value = self.obs_info.get(key, default)
        return value

    def is_track_resume_active(self):
        return self.track_resume_until is not None and self.now_seconds() < self.track_resume_until

    def is_obs_cooldown_active(self):
        return self.obs_cooldown_until is not None and self.now_seconds() < self.obs_cooldown_until

    def start_track_resume(self):
        self.track_resume_until = self.now_seconds() + self.track_resume_duration

    def start_obs_cooldown(self):
        self.obs_cooldown_until = self.now_seconds() + self.obs_cooldown_duration

    def set_mode(self, mode, reason):
        prev_mode = self.mode
        self.mode = mode
        self.mode_start_time = self.get_clock().now()
        self.mode_start_x = self.odom_x
        self.mode_start_y = self.odom_y
        if mode in (self.MODE_AVOID_ORBIT_SEARCH, self.MODE_AVOID_HANDOFF):
            self.reset_reacquire_state()
        self.get_logger().info(f'State: {prev_mode} -> {mode} ({reason})')

    def emit_debug(self, force=False, **state):
        payload = {
            'mode': self.mode,
            'point_x': self.latest_point_x if self.has_point else None,
            'point_y': self.latest_point_y if self.has_point else None,
            'front_min_dist': self.safe_obs_value('front_min_dist'),
            'obs_bearing_deg': self.safe_obs_value('obs_bearing_deg'),
            'obs_range': self.safe_obs_value('obs_range'),
        }
        payload.update(state)

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.debug_pub.publish(msg)

        now_s = self.now_seconds()
        signature = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        if force or signature != self.last_state_debug_signature or now_s - self.last_state_debug_time >= 0.50:
            extra_payload = {
                key: value
                for key, value in payload.items()
                if key not in ('mode', 'point_x', 'point_y', 'front_min_dist', 'obs_bearing_deg', 'obs_range')
            }
            self.get_logger().info(
                f"[{self.mode}] "
                f"front={payload.get('front_min_dist')} "
                f"bearing_deg={payload.get('obs_bearing_deg')} "
                f"range={payload.get('obs_range')} "
                f"px={payload.get('point_x')} "
                f"py={payload.get('point_y')} "
                f"extra={extra_payload}"
            )
            self.last_state_debug_time = now_s
            self.last_state_debug_signature = signature

    def obs_bool_callback(self, msg):
        self.has_obs = bool(msg.data)

    def obs_info_callback(self, msg):
        try:
            self.obs_info = json.loads(msg.data)
            self.has_obs = bool(self.obs_info.get('has_obs', self.has_obs))
        except json.JSONDecodeError:
            self.get_logger().warn('Failed to parse /obstacle_info JSON.')
            self.obs_info = None

        if self.mode == self.MODE_TRACK and not self.is_obs_cooldown_active():
            front_min_dist = self.safe_obs_value('front_min_dist')
            slow_range = self.safe_obs_value('slow_range')
            if (
                self.has_obs
                and front_min_dist is not None
                and slow_range is not None
                and front_min_dist < slow_range
            ):
                self.start_avoid_approach('front obstacle reached slow_range')

    def odom_callback(self, msg):
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)

        orientation = msg.pose.pose.orientation
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.has_odom = True

    def point_callback(self, msg):
        now = self.get_clock().now()
        point_x = float(msg.point.x)
        point_y = float(msg.point.y)
        point_x = max(0.0, min(point_x, self.image_width))
        self.latest_point_x = point_x
        self.latest_point_y = point_y
        self.last_point_time = now
        self.has_point = True
        self.point_history.append((self.now_seconds(), point_x, point_y))

        if self.mode != self.MODE_TRACK:
            return

        target_linear_x, target_angular_z = self.compute_track_targets(point_x, now)

        if self.is_track_resume_active():
            target_linear_x *= self.track_resume_speed_scale
            target_angular_z = self.clamp(
                target_angular_z,
                -self.track_resume_omega_limit,
                self.track_resume_omega_limit,
            )

        self.twist.linear.x = (
            (1.0 - self.linear_blend) * self.twist.linear.x
            + self.linear_blend * target_linear_x
        )
        self.twist.angular.z = (
            (1.0 - self.angular_blend) * self.twist.angular.z
            + self.angular_blend * target_angular_z
        )
        self.cmd_vel_pub.publish(self.twist)
        self.last_motion_time = now

    def point_fresh(self):
        if not self.has_point:
            return False
        return (self.get_clock().now() - self.last_point_time) <= self.cmd_timeout

    def phase_distance(self):
        if not self.has_odom:
            return 0.0
        return math.hypot(self.odom_x - self.mode_start_x, self.odom_y - self.mode_start_y)

    def left_turn_heading(self):
        return self.normalize_angle(self.avoid_origin_yaw + self.avoid_turn_angle)

    def yaw_to_angular_z(self, target_yaw):
        yaw_error = self.angle_error(target_yaw, self.odom_yaw)
        if abs(yaw_error) < self.avoid_yaw_tolerance:
            return 0.0

        target_angular_z = self.avoid_yaw_kp * yaw_error
        signed_min = math.copysign(self.avoid_yaw_min_angular_z, target_angular_z)
        if abs(target_angular_z) < self.avoid_yaw_min_angular_z:
            target_angular_z = signed_min

        return self.clamp(
            target_angular_z,
            -self.avoid_yaw_max_angular_z,
            self.avoid_yaw_max_angular_z,
        )

    def hold_heading_angular_z(self, target_heading):
        return self.yaw_to_angular_z(target_heading)

    def publish_motion(
        self,
        linear_x,
        angular_z,
        max_linear_accel=None,
        max_linear_decel=None,
        max_angular_accel=None,
    ):
        dt = self.control_period if self.last_motion_time is None else (
            (self.get_clock().now() - self.last_motion_time).nanoseconds / 1e9
        )
        if dt <= 0.0:
            dt = self.control_period

        if max_linear_accel is None:
            max_linear_accel = self.max_linear_accel
        if max_linear_decel is None:
            max_linear_decel = self.max_linear_decel
        if max_angular_accel is None:
            max_angular_accel = self.max_angular_accel

        linear_delta = linear_x - self.twist.linear.x
        max_linear_delta = (
            max_linear_accel * dt
            if abs(linear_x) > abs(self.twist.linear.x)
            else max_linear_decel * dt
        )
        angular_delta = angular_z - self.twist.angular.z
        max_angular_delta = max_angular_accel * dt

        self.twist.linear.x += self.clamp(linear_delta, -max_linear_delta, max_linear_delta)
        self.twist.linear.y = 0.0
        self.twist.linear.z = 0.0
        self.twist.angular.x = 0.0
        self.twist.angular.y = 0.0
        self.twist.angular.z += self.clamp(angular_delta, -max_angular_delta, max_angular_delta)
        self.last_motion_time = self.get_clock().now()
        self.cmd_vel_pub.publish(self.twist)

    def publish_avoid_motion(self, linear_x, angular_z):
        self.publish_motion(
            linear_x,
            angular_z,
            max_linear_accel=self.avoid_max_linear_accel,
            max_linear_decel=self.avoid_max_linear_decel,
            max_angular_accel=self.avoid_max_angular_accel,
        )

    def compute_track_targets(self, point_x, now):
        if self.last_control_time is None:
            dt = self.control_period
        else:
            dt = (now - self.last_control_time).nanoseconds / 1e9
            if dt <= 0.0 or dt > 0.25:
                dt = self.control_period

        e_now = self.center_x - point_x
        if abs(e_now) < self.deadband_pixels:
            e_now = 0.0

        if self.has_point:
            self.filtered_error = (
                (1.0 - self.error_filter_alpha) * self.filtered_error
                + self.error_filter_alpha * e_now
            )
        else:
            self.filtered_error = e_now

        self.ie += self.filtered_error * dt
        self.ie = self.clamp(self.ie, -self.integral_limit, self.integral_limit)

        raw_de = (self.filtered_error - self.e_last) / dt
        self.filtered_derivative = (
            (1.0 - self.derivative_filter_alpha) * self.filtered_derivative
            + self.derivative_filter_alpha * raw_de
        )

        target_angular_z = (
            self.kp * self.filtered_error
            + self.ki * self.ie
            + self.kd * self.filtered_derivative
        )
        target_angular_z = self.clamp(target_angular_z, -self.max_angular_z, self.max_angular_z)

        normalized_error = min(abs(self.filtered_error) / self.center_x, 1.0)
        speed_scale = 1.0 - normalized_error
        target_linear_x = self.min_linear_x + (self.max_linear_x - self.min_linear_x) * speed_scale

        self.e_last = self.filtered_error
        self.last_control_time = now
        return target_linear_x, target_angular_z

    def start_avoid_approach(self, reason):
        if self.mode != self.MODE_TRACK:
            return
        if not self.has_odom:
            self.enter_failsafe('No /odom received before avoidance start.')
            return
        self.avoid_origin_yaw = self.odom_yaw
        self.set_mode(self.MODE_AVOID_APPROACH, reason)
        self.emit_debug(force=True, reason=reason)

    def start_template_turn_left(self, reason):
        if not self.has_odom:
            self.enter_failsafe('No /odom received, template avoidance unavailable.')
            return
        self.reset_pid_state()
        self.set_mode(self.MODE_AVOID_TURN_LEFT, reason)
        self.emit_debug(force=True, reason=reason, avoid_origin_yaw=self.avoid_origin_yaw)

    def start_orbit_search(self, reason):
        obs_bearing = self.safe_obs_value('obs_bearing')
        if obs_bearing is None:
            self.orbit_direction = -1.0
        else:
            self.orbit_direction = 1.0 if obs_bearing >= 0.0 else -1.0
        self.orbit_target_bearing = self.orbit_direction * self.orbit_target_bearing_abs
        self.set_mode(self.MODE_AVOID_ORBIT_SEARCH, reason)
        self.emit_debug(
            force=True,
            reason=reason,
            orbit_direction=self.orbit_direction,
            orbit_target_bearing_deg=math.degrees(self.orbit_target_bearing),
        )

    def start_handoff(self, reason):
        self.reset_pid_state()
        self.set_mode(self.MODE_AVOID_HANDOFF, reason)
        self.emit_debug(force=True, reason=reason)

    def enter_failsafe(self, reason):
        self.reset_pid_state()
        self.stop_vehicle()
        self.set_mode(self.MODE_AVOID_FAILSAFE, reason)
        self.get_logger().error(f'Avoidance failsafe: {reason}')
        self.emit_debug(force=True, reason=reason)

    def finish_handoff(self, reason):
        self.set_mode(self.MODE_TRACK, reason)
        self.start_track_resume()
        self.start_obs_cooldown()
        self.get_logger().info('Tracking handoff complete, back to tracking.')
        self.emit_debug(force=True, reason=reason)

    def obs_front_distance(self):
        return self.safe_obs_value('front_min_dist')

    def obs_trigger_range(self):
        return self.safe_obs_value('trigger_range', 0.46)

    def obs_slow_range(self):
        return self.safe_obs_value('slow_range', 0.66)

    def point_consistent(self):
        if not self.point_fresh():
            return False
        if len(self.point_history) < self.point_trend_window:
            return False
        recent = list(self.point_history)[-self.point_trend_window:]
        xs = [item[1] for item in recent]
        ys = [item[2] for item in recent]
        if ys[-1] < self.point_trend_min_y:
            return False

        expected_sign = 1.0 if self.orbit_direction > 0.0 else -1.0
        signed_deltas = []
        for idx in range(1, len(xs)):
            signed_deltas.append((xs[idx] - xs[idx - 1]) * expected_sign)

        if not signed_deltas:
            return False
        positive_steps = sum(delta > self.point_min_signed_delta for delta in signed_deltas)
        avg_signed_delta = sum(signed_deltas) / len(signed_deltas)
        if positive_steps < max(1, len(signed_deltas) - 1):
            return False
        if avg_signed_delta < self.point_min_signed_delta:
            return False
        return True

    def track_point_valid(self):
        if not self.point_consistent():
            return False
        if self.latest_point_y < self.point_capture_min_y:
            return False
        if abs(self.center_x - self.latest_point_x) > self.capture_threshold:
            return False
        recent = list(self.point_history)[-self.point_capture_window:]
        xs = [item[1] for item in recent]
        if xs and (max(xs) - min(xs) > self.point_jitter_limit * 0.6):
            return False
        if abs(self.twist.angular.z) > self.capture_omega_threshold:
            return False
        if self.safe_obs_value('front_blocked', False):
            return False

        obs_bearing = self.safe_obs_value('obs_bearing')
        if obs_bearing is not None and abs(obs_bearing) < self.handoff_obstacle_safe_bearing:
            return False
        return True

    def update_reacquire_counters(self):
        if self.safe_obs_value('obs_track_valid', False):
            self.obs_track_valid_frames += 1
        else:
            self.obs_track_valid_frames = 0

        obs_bearing = self.safe_obs_value('obs_bearing')
        if obs_bearing is not None and self.prev_obs_bearing is not None:
            if abs(obs_bearing - self.prev_obs_bearing) < self.orbit_obs_bearing_stable_threshold:
                self.obs_bearing_stable_frames += 1
            else:
                self.obs_bearing_stable_frames = 0
        else:
            self.obs_bearing_stable_frames = 0
        self.prev_obs_bearing = obs_bearing

        obs_range = self.safe_obs_value('obs_range')
        if obs_range is not None and self.prev_obs_range is not None:
            if abs(obs_range - self.prev_obs_range) < self.orbit_obs_range_stable_threshold:
                self.obs_range_stable_frames += 1
            else:
                self.obs_range_stable_frames = 0
        else:
            self.obs_range_stable_frames = 0
        self.prev_obs_range = obs_range

        if self.point_fresh():
            self.point_fresh_frames += 1
        else:
            self.point_fresh_frames = 0

        if self.point_consistent():
            self.point_consistent_frames += 1
        else:
            self.point_consistent_frames = 0

    def compute_orbit_omega(self):
        obs_bearing = self.safe_obs_value('obs_bearing')
        obs_range = self.safe_obs_value('obs_range')
        front_min_dist = self.obs_front_distance()
        slow_range = self.obs_slow_range()
        trigger_range = self.obs_trigger_range()

        if (
            front_min_dist is not None
            and slow_range is not None
            and trigger_range is not None
        ):
            front_band = max(slow_range - trigger_range, 0.05)
            front_urgency = self.clamp(
                (slow_range - front_min_dist) / front_band,
                0.0,
                1.0,
            )
        else:
            front_urgency = 0.0

        if obs_bearing is not None:
            e_bearing = self.orbit_target_bearing - obs_bearing
            e_bearing = self.clamp(
                e_bearing,
                -self.orbit_bearing_error_limit,
                self.orbit_bearing_error_limit,
            )
            bearing_urgency = self.clamp(
                abs(e_bearing) / self.orbit_bearing_error_limit,
                0.0,
                1.0,
            )
        else:
            e_bearing = None
            bearing_urgency = 0.0

        if obs_range is not None:
            e_range = self.orbit_target_range - obs_range
            e_range = self.clamp(
                e_range,
                -self.orbit_range_error_limit,
                self.orbit_range_error_limit,
            )
            range_urgency = self.clamp(
                abs(e_range) / self.orbit_range_error_limit,
                0.0,
                1.0,
            )
        else:
            e_range = None
            range_urgency = 0.0

        orbit_urgency = max(front_urgency, bearing_urgency, range_urgency)
        dynamic_base = self.orbit_omega_base_min + (
            self.orbit_omega_base_max - self.orbit_omega_base_min
        ) * orbit_urgency
        dynamic_k_bearing = self.orbit_k_bearing_min + (
            self.orbit_k_bearing_max - self.orbit_k_bearing_min
        ) * orbit_urgency
        dynamic_k_range = self.orbit_k_range_min + (
            self.orbit_k_range_max - self.orbit_k_range_min
        ) * orbit_urgency
        dynamic_k_damp = self.orbit_k_damp_max - (
            self.orbit_k_damp_max - self.orbit_k_damp_min
        ) * orbit_urgency
        dynamic_omega_limit = self.orbit_max_angular_z_min + (
            self.orbit_max_angular_z_max - self.orbit_max_angular_z_min
        ) * orbit_urgency

        omega_cmd = self.orbit_direction * dynamic_base
        if e_bearing is not None:
            omega_cmd += dynamic_k_bearing * e_bearing
        if e_range is not None:
            omega_cmd += dynamic_k_range * e_range

        if (
            obs_bearing is not None
            and front_min_dist is not None
            and abs(obs_bearing) < self.orbit_front_turn_boost_angle
            and front_min_dist < self.orbit_front_turn_boost_dist
        ):
            omega_cmd += self.orbit_direction * self.orbit_front_turn_boost

        omega_cmd -= dynamic_k_damp * self.twist.angular.z
        omega_cmd = self.clamp(
            omega_cmd,
            -dynamic_omega_limit,
            dynamic_omega_limit,
        )
        return (
            omega_cmd,
            e_bearing,
            e_range,
            orbit_urgency,
            dynamic_base,
            dynamic_k_bearing,
            dynamic_k_range,
            dynamic_k_damp,
            dynamic_omega_limit,
        )

    def compute_orbit_linear_x(self):
        front_min_dist = self.obs_front_distance()
        if front_min_dist is not None and front_min_dist < 0.20:
            return self.orbit_close_linear_x
        return self.orbit_search_linear_x

    def handle_approach_mode(self, elapsed):
        front_min_dist = self.obs_front_distance()
        slow_range = self.obs_slow_range()
        trigger_range = self.obs_trigger_range()
        approach_commit_distance = trigger_range + self.approach_trigger_margin

        if front_min_dist is None or not self.has_obs or front_min_dist > slow_range:
            self.set_mode(self.MODE_TRACK, 'obstacle cleared during approach')
            self.emit_debug(force=True, reason='obstacle cleared during approach')
            return

        if front_min_dist < approach_commit_distance:
            self.start_template_turn_left(
                f'approach commit: front={front_min_dist:.3f}m threshold={approach_commit_distance:.3f}m'
            )
            return

        now = self.get_clock().now()
        base_linear_x, base_angular_z = self.compute_track_targets(self.latest_point_x, now)
        denom = max(slow_range - approach_commit_distance, 0.05)
        closeness = self.clamp((slow_range - front_min_dist) / denom, 0.0, 1.0)
        linear_scale = 0.42 - 0.26 * closeness
        linear_x = self.clamp(
            base_linear_x * linear_scale,
            self.approach_min_linear_x,
            self.approach_max_linear_x,
        )

        recommended_side = self.safe_obs_value('recommended_side', 'left')
        direction = 1.0
        avoid_steer = direction * (0.24 + 0.42 * closeness * closeness)
        angular_z = self.clamp(base_angular_z + avoid_steer, -self.max_angular_z, self.max_angular_z)

        self.publish_avoid_motion(linear_x, angular_z)
        self.emit_debug(
            front_min_dist=front_min_dist,
            approach_commit_distance=round(approach_commit_distance, 3),
            closeness=round(closeness, 3),
            linear_x=round(linear_x, 3),
            angular_z=round(angular_z, 3),
            recommended_side=recommended_side,
        )

    def handle_turn_left_mode(self, elapsed):
        target_yaw = self.left_turn_heading()
        angular_z = self.yaw_to_angular_z(target_yaw)
        self.publish_avoid_motion(0.0, angular_z)
        yaw_error = self.angle_error(target_yaw, self.odom_yaw)
        self.emit_debug(target_yaw=round(target_yaw, 3), yaw_error=round(yaw_error, 3))

        if abs(yaw_error) < self.avoid_yaw_tolerance:
            self.set_mode(self.MODE_AVOID_FORWARD_1, 'left turn complete')
            return
        if elapsed >= self.avoid_turn_timeout:
            self.enter_failsafe(f'turn_left timeout yaw_err={yaw_error:.3f}rad')

    def handle_forward_1_mode(self, elapsed):
        target_yaw = self.left_turn_heading()
        angular_z = self.hold_heading_angular_z(target_yaw)
        distance = self.phase_distance()
        self.publish_avoid_motion(self.avoid_forward_linear_x, angular_z)
        self.emit_debug(
            phase_distance=round(distance, 3),
            target_distance=self.avoid_side_distance,
            angular_z=round(angular_z, 3),
        )

        if distance >= self.avoid_side_distance:
            self.set_mode(self.MODE_AVOID_TURN_RIGHT, 'forward_1 complete')
            return
        if elapsed >= self.avoid_forward_timeout:
            self.enter_failsafe(f'forward_1 timeout dist={distance:.3f}m')

    def handle_turn_right_mode(self, elapsed):
        target_yaw = self.avoid_origin_yaw
        angular_z = self.yaw_to_angular_z(target_yaw)
        self.publish_avoid_motion(0.0, angular_z)
        yaw_error = self.angle_error(target_yaw, self.odom_yaw)
        self.emit_debug(target_yaw=round(target_yaw, 3), yaw_error=round(yaw_error, 3))

        if abs(yaw_error) < self.avoid_yaw_tolerance:
            self.start_orbit_search('turn_right complete')
            return
        if elapsed >= self.avoid_turn_timeout:
            self.enter_failsafe(f'turn_right timeout yaw_err={yaw_error:.3f}rad')

    def handle_orbit_search_mode(self, elapsed):
        self.update_reacquire_counters()
        (
            omega_cmd,
            e_bearing,
            e_range,
            orbit_urgency,
            dynamic_base,
            dynamic_k_bearing,
            dynamic_k_range,
            dynamic_k_damp,
            dynamic_omega_limit,
        ) = self.compute_orbit_omega()
        linear_x = self.compute_orbit_linear_x()
        self.publish_avoid_motion(linear_x, omega_cmd)

        front_min_dist = self.obs_front_distance()
        obstacle_stable = (
            self.obs_track_valid_frames >= 1
            and self.obs_bearing_stable_frames >= self.orbit_required_obs_stable_frames
            and self.obs_range_stable_frames >= self.orbit_required_obs_stable_frames
        )
        point_ready = self.point_consistent_frames >= self.orbit_required_consistent_frames
        front_safe = (
            front_min_dist is None
            or front_min_dist > (self.obs_trigger_range() + self.orbit_safe_front_margin)
        )

        self.emit_debug(
            front_min_dist=front_min_dist,
            linear_x=round(linear_x, 3),
            omega_cmd=round(omega_cmd, 3),
            e_bearing=None if e_bearing is None else round(e_bearing, 3),
            e_range=None if e_range is None else round(e_range, 3),
            orbit_urgency=round(orbit_urgency, 3),
            dynamic_base=round(dynamic_base, 3),
            dynamic_k_bearing=round(dynamic_k_bearing, 3),
            dynamic_k_range=round(dynamic_k_range, 3),
            dynamic_k_damp=round(dynamic_k_damp, 3),
            dynamic_omega_limit=round(dynamic_omega_limit, 3),
            obs_track_valid_frames=self.obs_track_valid_frames,
            obs_bearing_stable_frames=self.obs_bearing_stable_frames,
            obs_range_stable_frames=self.obs_range_stable_frames,
            point_consistent_frames=self.point_consistent_frames,
            front_safe=front_safe,
        )

        if point_ready and obstacle_stable and front_safe:
            self.start_handoff(
                'point_consistent and obstacle geometry stable'
            )
            return

        if elapsed >= self.avoid_orbit_timeout:
            self.enter_failsafe(
                'orbit_search timeout '
                f'point_consistent_frames={self.point_consistent_frames} '
                f'obs_stable={obstacle_stable}'
            )

    def handle_handoff_mode(self, elapsed):
        self.update_reacquire_counters()
        (
            omega_radar,
            e_bearing,
            e_range,
            orbit_urgency,
            dynamic_base,
            dynamic_k_bearing,
            dynamic_k_range,
            dynamic_k_damp,
            dynamic_omega_limit,
        ) = self.compute_orbit_omega()
        orbit_linear_x = self.compute_orbit_linear_x()

        if self.point_fresh():
            now = self.get_clock().now()
            lane_linear_x, omega_lane = self.compute_track_targets(self.latest_point_x, now)
            lane_linear_x = min(lane_linear_x, self.avoid_handoff_max_linear_x)
        else:
            lane_linear_x = 0.0
            omega_lane = 0.0

        if (
            self.point_consistent()
            and self.latest_point_y >= self.point_capture_min_y
            and abs(self.center_x - self.latest_point_x) < self.capture_threshold
        ):
            self.handoff_stable_frames += 1
        else:
            self.handoff_stable_frames = max(0, self.handoff_stable_frames - 1)

        alpha = self.clamp(
            self.handoff_stable_frames / float(self.handoff_alpha_ramp_frames),
            0.0,
            1.0,
        )
        linear_x = (1.0 - alpha) * orbit_linear_x + alpha * lane_linear_x
        angular_z = (1.0 - alpha) * omega_radar + alpha * omega_lane
        angular_limit = max(
            self.handoff_omega_threshold,
            self.avoid_yaw_max_angular_z * (1.0 - 0.55 * alpha),
        )
        angular_z = self.clamp(angular_z, -angular_limit, angular_limit)
        self.publish_avoid_motion(linear_x, angular_z)

        track_valid = self.track_point_valid()
        obstacle_safe = not self.safe_obs_value('front_blocked', False)
        if track_valid and obstacle_safe:
            handoff_ready = (
                elapsed >= self.avoid_handoff_duration
                and abs(self.center_x - self.latest_point_x) < self.handoff_threshold
                and abs(self.twist.angular.z) < self.handoff_omega_threshold
            )
        else:
            handoff_ready = False

        self.emit_debug(
            alpha=round(alpha, 3),
            linear_x=round(linear_x, 3),
            angular_z=round(angular_z, 3),
            omega_radar=round(omega_radar, 3),
            omega_lane=round(omega_lane, 3),
            e_bearing=None if e_bearing is None else round(e_bearing, 3),
            e_range=None if e_range is None else round(e_range, 3),
            orbit_urgency=round(orbit_urgency, 3),
            dynamic_base=round(dynamic_base, 3),
            dynamic_k_bearing=round(dynamic_k_bearing, 3),
            dynamic_k_range=round(dynamic_k_range, 3),
            dynamic_k_damp=round(dynamic_k_damp, 3),
            dynamic_omega_limit=round(dynamic_omega_limit, 3),
            handoff_stable_frames=self.handoff_stable_frames,
            track_valid=track_valid,
            obstacle_safe=obstacle_safe,
        )

        if handoff_ready:
            self.finish_handoff('track point valid and handoff window satisfied')
            return

        if self.safe_obs_value('front_blocked', False):
            self.enter_failsafe(
                'handoff front blocked again '
                f'front={self.obs_front_distance():.3f}m'
            )
            return

        if elapsed >= self.avoid_handoff_timeout:
            self.enter_failsafe(
                'handoff timeout '
                f'alpha={alpha:.2f} track_valid={track_valid}'
            )

    def run_avoidance_state(self):
        elapsed = (self.get_clock().now() - self.mode_start_time).nanoseconds / 1e9

        if not self.has_odom:
            self.enter_failsafe('lost odom during avoidance')
            return

        if self.mode == self.MODE_AVOID_APPROACH:
            self.handle_approach_mode(elapsed)
        elif self.mode == self.MODE_AVOID_TURN_LEFT:
            self.handle_turn_left_mode(elapsed)
        elif self.mode == self.MODE_AVOID_FORWARD_1:
            self.handle_forward_1_mode(elapsed)
        elif self.mode == self.MODE_AVOID_TURN_RIGHT:
            self.handle_turn_right_mode(elapsed)
        elif self.mode == self.MODE_AVOID_ORBIT_SEARCH:
            self.handle_orbit_search_mode(elapsed)
        elif self.mode == self.MODE_AVOID_HANDOFF:
            self.handle_handoff_mode(elapsed)
        elif self.mode == self.MODE_AVOID_FAILSAFE:
            self.stop_vehicle()

    def watchdog_callback(self):
        if self.mode != self.MODE_TRACK:
            self.run_avoidance_state()
            return

        if not self.has_point:
            return

        if self.get_clock().now() - self.last_point_time > self.cmd_timeout:
            self.reset_pid_state()
            self.stop_vehicle()
            self.has_point = False
            self.get_logger().warn('Track watchdog timeout: lost line point, stop vehicle.')


def main(args=None):
    rclpy.init(args=args)
    lanedetection = LaneDetection()
    try:
        rclpy.spin(lanedetection)
    except KeyboardInterrupt:
        pass
    finally:
        lanedetection.on_shutdown()
        lanedetection.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

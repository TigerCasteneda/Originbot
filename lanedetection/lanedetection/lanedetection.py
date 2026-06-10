#!/usr/bin/env python3
import json
import math

import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String

try:
    from ai_msgs.msg import PerceptionTargets
except ImportError:
    PerceptionTargets = None

try:
    from .finish_detection import (
        CheckerboardConfig,
        decode_compressed_image,
        detect_checkerboard_finish,
    )
    from .recovery_logic import (
        avoid_ready_for_return,
        clamp,
        compute_return_command,
        compute_return_target_yaw,
        is_reacquired_line_candidate,
        obstacle_cleared_for_return,
        should_enter_search,
    )
except ImportError:
    from finish_detection import (
        CheckerboardConfig,
        decode_compressed_image,
        detect_checkerboard_finish,
    )
    from recovery_logic import (
        avoid_ready_for_return,
        clamp,
        compute_return_command,
        compute_return_target_yaw,
        is_reacquired_line_candidate,
        obstacle_cleared_for_return,
        should_enter_search,
    )


class LaneDetection(Node):
    def __init__(self):
        super().__init__("lanedetection")
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.debug_pub = self.create_publisher(String, "/debug/lane_state", 10)
        self.finish_pub = self.create_publisher(Bool, "/finish_detected", 10)
        self.finish_debug_pub = self.create_publisher(String, "/debug/finish_state", 10)
        self.point_sub = self.create_subscription(
            Point, "/line_point", self.point_callback, 10
        )
        self.obs_sub = self.create_subscription(
            String, "/obstacle_info", self.obs_callback, 10
        )
        self.image_sub = self.create_subscription(
            CompressedImage,
            "/image",
            self.image_callback,
            qos_profile_sensor_data,
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        if PerceptionTargets is not None:
            self.traffic_sub = self.create_subscription(
                PerceptionTargets,
                "/traffice_sign",
                self.traffic_callback,
                10,
            )
        else:
            self.traffic_sub = None

        # ---- lane PID ----
        self.declare_parameter("target_x", 900.0)
        self.declare_parameter("base_speed", 0.30)
        self.declare_parameter("kp", 0.0022)
        self.declare_parameter("ki", 0.0020)
        self.declare_parameter("kd", 0.0008)
        self.declare_parameter("integral_limit", 600.0)
        self.declare_parameter("deadband_pixels", 8.0)
        self.declare_parameter("error_filter_alpha", 0.60)
        self.declare_parameter("derivative_filter_alpha", 0.40)
        self.declare_parameter("linear_blend", 0.40)
        self.declare_parameter("angular_blend", 0.40)
        self.declare_parameter("min_linear_x", 0.15)
        self.declare_parameter("max_linear_x", 0.50)
        self.declare_parameter("max_angular_speed", 1.00)
        self.declare_parameter("max_linear_accel", 0.35)
        self.declare_parameter("max_linear_decel", 0.55)
        self.declare_parameter("max_angular_accel", 3.0)
        self.declare_parameter("point_timeout", 0.5)

        # ---- obstacle detection thresholds ----
        self.declare_parameter("slow_range", 0.75)
        self.declare_parameter("trigger_range", 0.50)
        self.declare_parameter("obs_cooldown", 5.0)

        # ---- avoid (heading PID) ----
        self.declare_parameter("avoid_speed", 0.12)
        self.declare_parameter("avoid_turn_angle", 1.30)       # rad (~75 deg)
        self.declare_parameter("avoid_yaw_kp", 1.2)
        self.declare_parameter("avoid_yaw_tolerance", 0.12)
        self.declare_parameter("avoid_yaw_max_angular", 1.5)
        self.declare_parameter("avoid_timeout", 5.0)
        self.declare_parameter("min_avoid_time", 1.2)
        self.declare_parameter("min_avoid_distance", 0.25)
        self.declare_parameter("reavoid_grace_period", 2.5)    # sec before re-trigger allowed
        self.declare_parameter("return_front_clear_range", 0.55)
        self.declare_parameter("return_side_release_range", 0.60)

        # ---- return ----
        self.declare_parameter("return_speed", 0.06)
        self.declare_parameter("return_yaw_kp", 2.0)
        self.declare_parameter("return_yaw_tolerance", 0.06)
        self.declare_parameter("return_timeout", 10.0)
        self.declare_parameter("min_return_time", 2.0)

        # ---- search ----
        self.declare_parameter("search_speed", 0.04)
        self.declare_parameter("search_omega", 0.40)
        self.declare_parameter("search_timeout", 8.0)
        self.declare_parameter("line_reacquire_error_px", 800.0)
        self.declare_parameter("line_reacquire_min_y", 200.0)
        self.declare_parameter("line_reacquire_frames", 3)

        # ---- stop / misc ----
        self.declare_parameter("stop_cooldown", 3.0)
        self.declare_parameter("recover_timeout", 10.0)

        # ---- finish line ----
        self.declare_parameter("finish_detect_enabled", True)
        self.declare_parameter("finish_confirm_frames", 2)
        self.declare_parameter("finish_check_interval", 0.15)
        self.declare_parameter("finish_ignore_after_start", 1.5)
        self.declare_parameter("finish_roi_top_ratio", 0.35)
        self.declare_parameter("finish_roi_bottom_ratio", 0.95)
        self.declare_parameter("finish_roi_left_ratio", 0.05)
        self.declare_parameter("finish_roi_right_ratio", 0.95)
        self.declare_parameter("finish_grid_cols", 10)
        self.declare_parameter("finish_grid_rows", 6)
        self.declare_parameter("finish_min_contrast", 55.0)
        self.declare_parameter("finish_min_white_ratio", 0.25)
        self.declare_parameter("finish_max_white_ratio", 0.75)
        self.declare_parameter("finish_min_transition_ratio", 0.50)
        self.declare_parameter("finish_min_strong_cells", 16)

        # ---- traffic light ----
        self.declare_parameter("traffic_detect_enabled", True)
        self.declare_parameter("traffic_confirm_frames", 2)
        self.declare_parameter("traffic_confidence_threshold", 0.3)
        self.declare_parameter("traffic_cooldown", 2.0)

        # ---- state variables ----
        self.point = None
        self.last_point_time = None
        self.obs_info = None
        self.mode = "TRACK"
        self.control_period = 0.1

        self.min_valid_x = 0.0
        self.max_valid_x = 1920.0
        self.min_valid_y = 0.0
        self.max_valid_y = 1080.0
        self.point_timeout = float(self.get_parameter("point_timeout").value)

        # odometry
        self.odom_yaw = 0.0
        self.has_odom = False

        # avoidance state
        self.arc_direction = 0.0
        self.avoid_speed_val = 0.08
        self.avoid_origin_yaw = 0.0
        self.avoid_start = None
        self.avoid_distance_accum = 0.0
        self.return_clear_streak = 0
        self.search_line_streak = 0
        self.return_start = None
        self.search_start = None
        self.track_resume_cycles = 0
        self.stop_start = None
        self.obs_cooldown_until = None

        # finish line state
        self.finish_seen_streak = 0
        self.finish_detected = False
        self.finish_metrics = {}
        self.finish_frame_seen = False
        self.last_finish_check_time = None
        self.node_start_time = self.get_clock().now()

        # traffic light state
        self.traffic_state = "none"
        self.traffic_red_streak = 0
        self.traffic_green_streak = 0
        self.traffic_stop_reason = False
        self.traffic_cooldown_until = 0.0

        # PID state
        self.e_last = 0.0
        self.ie = 0.0
        self.filtered_error = 0.0
        self.filtered_derivative = 0.0
        self.last_twist_linear_x = 0.0
        self.last_twist_angular_z = 0.0
        self.last_control_time_ns = None
        self.last_motion_time = None

        self.create_timer(self.control_period, self.timer_callback)
        self.get_logger().info("lanedetection PID+avoid+traffic+finish started")

    def param(self, name):
        return self.get_parameter(name).value

    # ==================================================================
    #  callbacks
    # ==================================================================
    def point_callback(self, msg: Point):
        self.point = msg
        self.last_point_time = self.get_clock().now()

    def obs_callback(self, msg: String):
        try:
            self.obs_info = json.loads(msg.data)
        except json.JSONDecodeError:
            self.obs_info = None

    def odom_callback(self, msg: Odometry):
        orientation = msg.pose.pose.orientation
        siny = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.odom_yaw = math.atan2(siny, cosy)
        self.has_odom = True

    def traffic_callback(self, msg):
        if not bool(self.param("traffic_detect_enabled")):
            return
        threshold = float(self.param("traffic_confidence_threshold"))
        confirm = int(self.param("traffic_confirm_frames"))

        has_red = False
        has_green = False
        for box in msg.targets:
            if box.rois[0].confidence < threshold:
                continue
            cls_type = box.rois[0].type.strip()
            if cls_type == "red_light":
                has_red = True
            elif cls_type == "green_light":
                has_green = True

        if has_red:
            self.traffic_red_streak += 1
            self.traffic_green_streak = 0
        elif has_green:
            self.traffic_green_streak += 1
            self.traffic_red_streak = 0
        else:
            self.traffic_red_streak = max(0, self.traffic_red_streak - 1)
            self.traffic_green_streak = max(0, self.traffic_green_streak - 1)

        if self.traffic_red_streak >= confirm:
            prev = self.traffic_state
            self.traffic_state = "red"
            if prev != "red":
                self.get_logger().warn("traffic light: RED detected")
        elif self.traffic_green_streak >= confirm:
            prev = self.traffic_state
            self.traffic_state = "green"
            if prev != "green":
                self.get_logger().info("traffic light: GREEN detected")
                self.traffic_stop_reason = False
        elif self.traffic_red_streak == 0 and self.traffic_green_streak == 0:
            self.traffic_state = "none"

    def image_callback(self, msg: CompressedImage):
        if not bool(self.param("finish_detect_enabled")) or self.finish_detected:
            return

        now = self.get_clock().now()
        if self.elapsed_seconds(self.node_start_time) < float(self.param("finish_ignore_after_start")):
            return
        if self.last_finish_check_time is not None:
            elapsed = (now - self.last_finish_check_time).nanoseconds / 1e9
            if elapsed < float(self.param("finish_check_interval")):
                return
        self.last_finish_check_time = now

        image = decode_compressed_image(msg.data)
        config = CheckerboardConfig(
            roi_top_ratio=float(self.param("finish_roi_top_ratio")),
            roi_bottom_ratio=float(self.param("finish_roi_bottom_ratio")),
            roi_left_ratio=float(self.param("finish_roi_left_ratio")),
            roi_right_ratio=float(self.param("finish_roi_right_ratio")),
            grid_cols=int(self.param("finish_grid_cols")),
            grid_rows=int(self.param("finish_grid_rows")),
            min_contrast=float(self.param("finish_min_contrast")),
            min_white_ratio=float(self.param("finish_min_white_ratio")),
            max_white_ratio=float(self.param("finish_max_white_ratio")),
            min_transition_ratio=float(self.param("finish_min_transition_ratio")),
            min_strong_cells=int(self.param("finish_min_strong_cells")),
        )
        detected, metrics = detect_checkerboard_finish(image, config)
        self.finish_metrics = metrics
        self.finish_frame_seen = True

        if detected:
            self.finish_seen_streak += 1
        else:
            self.finish_seen_streak = 0

        if self.finish_seen_streak >= int(self.param("finish_confirm_frames")):
            self.finish_detected = True
            self.mode = "FINISH"
            self.publish_stop("finish_detected")
            self.get_logger().warn(f"finish checkerboard detected, stopping: {metrics}")

        self.publish_finish_state(detected)

    # ==================================================================
    #  helpers
    # ==================================================================
    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def angle_error(target, current):
        return LaneDetection.normalize_angle(target - current)

    def publish_finish_state(self, detected_this_frame=False):
        msg = Bool()
        msg.data = bool(self.finish_detected)
        self.finish_pub.publish(msg)
        state = {
            "detected": bool(self.finish_detected),
            "detected_this_frame": bool(detected_this_frame),
            "seen_streak": int(self.finish_seen_streak),
            "confirm_frames": int(self.param("finish_confirm_frames")),
            "frame_seen": bool(self.finish_frame_seen),
            "metrics": self.finish_metrics,
        }
        self.finish_debug_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))

    def elapsed_seconds(self, start_time):
        if start_time is None:
            return 0.0
        return (self.get_clock().now() - start_time).nanoseconds / 1e9

    def valid_point(self):
        if self.point is None:
            return False
        return (self.min_valid_x <= self.point.x <= self.max_valid_x
                and self.min_valid_y <= self.point.y <= self.max_valid_y)

    def point_age(self):
        if self.last_point_time is None:
            return None
        return (self.get_clock().now() - self.last_point_time).nanoseconds / 1e9

    def _set_obs_cooldown(self):
        cooldown = float(self.param("obs_cooldown"))
        if cooldown > 0.0:
            self.obs_cooldown_until = self.get_clock().now().nanoseconds / 1e9 + cooldown

    def _reset_track_state(self):
        self.e_last = 0.0
        self.ie = 0.0
        self.filtered_error = 0.0
        self.filtered_derivative = 0.0
        self.last_control_time_ns = None
        self.last_twist_linear_x = 0.0
        self.last_twist_angular_z = 0.0

    def obstacle_active(self, obs):
        if obs is None:
            return False
        return bool(obs.get("has_obs", False) or obs.get("front_blocked", False))

    def line_reacquired(self, ignore_obs=False):
        if not self.valid_point():
            return False
        point = {"x": float(self.point.x), "y": float(self.point.y)}
        return is_reacquired_line_candidate(
            point=point,
            point_age=self.point_age(),
            target_x=float(self.param("target_x")),
            line_reacquire_error_px=float(self.param("line_reacquire_error_px")),
            line_reacquire_min_y=float(self.param("line_reacquire_min_y")),
            point_timeout=self.point_timeout,
            obs=None if ignore_obs else self.obs_info,
        )

    def publish_stop(self, reason="stop"):
        twist = Twist()
        self.cmd_vel_pub.publish(twist)
        self.last_twist_linear_x = 0.0
        self.last_twist_angular_z = 0.0
        self.last_motion_time = None
        self.publish_finish_state(self.finish_detected)
        state = {
            "mode": self.mode,
            "reason": reason,
            "speed": 0.0,
            "angular_z": 0.0,
            "finish_detected": self.finish_detected,
            "finish_seen_streak": self.finish_seen_streak,
            "finish_metrics": self.finish_metrics,
            "traffic_state": self.traffic_state,
        }
        self.debug_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))

    def timer_callback(self):
        if self.last_point_time is not None:
            age = (self.get_clock().now() - self.last_point_time).nanoseconds / 1e9
            if age > self.point_timeout:
                self.point = None
                if self.mode == "TRACK":
                    self._reset_track_state()
        self.publish_cmd()

    # ==================================================================
    #  unified motion output (blend + accel limits)
    # ==================================================================
    def publish_motion(self, linear_x, angular_z):
        if self.last_motion_time is None:
            dt = self.control_period
        else:
            dt = (self.get_clock().now() - self.last_motion_time).nanoseconds / 1e9
        if dt <= 0.0 or dt > 0.25:
            dt = self.control_period

        max_linear_accel = float(self.param("max_linear_accel"))
        max_linear_decel = float(self.param("max_linear_decel"))
        max_angular_accel = float(self.param("max_angular_accel"))
        linear_blend = float(self.param("linear_blend"))
        angular_blend = float(self.param("angular_blend"))

        # blend
        target_lx = (1.0 - linear_blend) * self.last_twist_linear_x + linear_blend * linear_x
        target_az = (1.0 - angular_blend) * self.last_twist_angular_z + angular_blend * angular_z

        # acceleration limits
        linear_delta = target_lx - self.last_twist_linear_x
        max_ld = max_linear_accel * dt if abs(target_lx) > abs(self.last_twist_linear_x) else max_linear_decel * dt
        target_lx = self.last_twist_linear_x + clamp(linear_delta, -max_ld, max_ld)

        angular_delta = target_az - self.last_twist_angular_z
        target_az = self.last_twist_angular_z + clamp(angular_delta, -max_angular_accel * dt, max_angular_accel * dt)

        self.last_twist_linear_x = target_lx
        self.last_twist_angular_z = target_az
        self.last_motion_time = self.get_clock().now()

        twist = Twist()
        twist.linear.x = target_lx
        twist.angular.z = target_az
        self.cmd_vel_pub.publish(twist)

    # ==================================================================
    #  lane PID computation (shared by TRACK / APPROACH / SEARCH)
    # ==================================================================
    def compute_lane_pid(self, point_x):
        """Returns (target_linear_x, target_angular_z) from visual PID."""
        target_x = float(self.param("target_x"))
        kp = float(self.param("kp"))
        ki = float(self.param("ki"))
        kd = float(self.param("kd"))
        integral_limit = float(self.param("integral_limit"))
        max_angular_speed = float(self.param("max_angular_speed"))
        error_filter_alpha = float(self.param("error_filter_alpha"))
        deriv_filter_alpha = float(self.param("derivative_filter_alpha"))
        deadband = float(self.param("deadband_pixels"))
        min_speed = float(self.param("min_linear_x"))
        max_speed = float(self.param("max_linear_x"))

        now_ns = self.get_clock().now().nanoseconds
        if self.last_control_time_ns is None:
            dt = self.control_period
        else:
            dt = (now_ns - self.last_control_time_ns) / 1e9
            if dt <= 0.0 or dt > 0.25:
                dt = self.control_period

        e_now = target_x - float(point_x)
        if abs(e_now) < deadband:
            e_now = 0.0

        self.filtered_error = ((1.0 - error_filter_alpha) * self.filtered_error
                               + error_filter_alpha * e_now)
        self.ie += self.filtered_error * dt
        self.ie = clamp(self.ie, -integral_limit, integral_limit)

        raw_de = (self.filtered_error - self.e_last) / dt
        self.filtered_derivative = ((1.0 - deriv_filter_alpha) * self.filtered_derivative
                                    + deriv_filter_alpha * raw_de)

        angular_z = kp * self.filtered_error + ki * self.ie + kd * self.filtered_derivative
        angular_z = clamp(angular_z, -max_angular_speed, max_angular_speed)

        normalized_error = min(abs(self.filtered_error) / max(target_x, 1.0), 1.0)
        linear_x = min_speed + (max_speed - min_speed) * (1.0 - normalized_error)

        self.e_last = self.filtered_error
        self.last_control_time_ns = now_ns
        return linear_x, angular_z

    # ==================================================================
    #  heading PID (shared by AVOID / RETURN)
    # ==================================================================
    def heading_to_angular(self, target_yaw, kp_yaw, tolerance, max_angular):
        """Returns angular_z to reach target_yaw via P control."""
        if not self.has_odom:
            return 0.0
        err = self.angle_error(target_yaw, self.odom_yaw)
        if abs(err) < tolerance:
            return 0.0
        return clamp(kp_yaw * err, -max_angular, max_angular)

    # ==================================================================
    #  TRACK
    # ==================================================================
    def track_normal(self):
        if self.valid_point():
            target_lx, target_az = self.compute_lane_pid(self.point.x)
        else:
            self.ie = 0.0
            target_lx, target_az = 0.0, 0.0

        if self.track_resume_cycles > 0:
            target_lx *= 0.6
            target_az = clamp(target_az, -0.25, 0.25)
            self.track_resume_cycles -= 1

        self.publish_motion(target_lx, target_az)

        state = {
            "mode": "TRACK",
            "point_valid": self.valid_point(),
            "point_x": float(self.point.x) if self.valid_point() else None,
            "point_y": float(self.point.y) if self.valid_point() else None,
            "target_x": float(self.param("target_x")),
            "filtered_error": round(self.filtered_error, 1),
            "ie": round(self.ie, 1),
            "speed": round(self.last_twist_linear_x, 3),
            "angular_z": round(self.last_twist_angular_z, 3),
            "traffic_state": self.traffic_state,
            "odom_yaw": round(self.odom_yaw, 3) if self.has_odom else None,
        }
        self.debug_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))

    # ==================================================================
    #  APPROACH  (full PID lane tracking + reduced speed + escape bias)
    # ==================================================================
    def approach_slow(self):
        obs = self.obs_info
        trigger_range = float(self.param("trigger_range"))
        slow_range = float(self.param("slow_range"))

        # lane PID
        if self.valid_point():
            target_lx, target_az = self.compute_lane_pid(self.point.x)
        else:
            target_lx, target_az = 0.0, 0.0

        # reduce speed based on obstacle proximity
        if obs and obs.get("front_min_dist") is not None:
            front = obs["front_min_dist"]
            margin = max(slow_range - trigger_range, 0.01)
            t = clamp((slow_range - front) / margin, 0.0, 1.0)
        else:
            t = 0.0
        target_lx = target_lx * (1.0 - 0.65 * t)

        # feed-forward escape bias (steer away from obstacle)
        avoid_bias = 0.0
        if obs and obs.get("front_min_dist") is not None and obs.get("recommended_side", "none") != "none":
            direction = 1.0 if obs["recommended_side"] == "left" else -1.0
            avoid_bias = direction * t * t * 0.25
        target_az = clamp(target_az + avoid_bias, -float(self.param("max_angular_speed")),
                          float(self.param("max_angular_speed")))

        self.publish_motion(target_lx, target_az)

        state = {
            "mode": "APPROACH",
            "speed": round(self.last_twist_linear_x, 3),
            "angular_z": round(self.last_twist_angular_z, 3),
            "avoid_bias": round(avoid_bias, 3),
            "closeness": round(t, 3),
            "front_dist": obs.get("front_min_dist") if obs else None,
            "traffic_state": self.traffic_state,
        }
        self.debug_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))

    # ==================================================================
    #  AVOID  (heading PID: turn away from obstacle)
    # ==================================================================
    def plan_avoid(self):
        obs = self.obs_info
        if obs is None:
            return

        if obs.get("recommended_side") == "left":
            self.arc_direction = -1.0
        else:
            self.arc_direction = 1.0

        self.avoid_speed_val = float(self.param("avoid_speed"))
        self.avoid_origin_yaw = self.odom_yaw if self.has_odom else 0.0
        self.avoid_start = self.get_clock().now()
        self.avoid_distance_accum = 0.0
        self.return_clear_streak = 0
        self._reset_track_state()

        self.get_logger().info(
            f"avoid start: dir={'left' if self.arc_direction > 0 else 'right'}, "
            f"origin_yaw={math.degrees(self.avoid_origin_yaw):.1f}deg, "
            f"front={obs.get('front_min_dist', '?')}"
        )

    def execute_avoid(self):
        turn_angle = float(self.param("avoid_turn_angle"))
        kp_yaw = float(self.param("avoid_yaw_kp"))
        tolerance = float(self.param("avoid_yaw_tolerance"))
        max_ang = float(self.param("avoid_yaw_max_angular"))

        target_yaw = self.normalize_angle(self.avoid_origin_yaw + self.arc_direction * turn_angle)
        angular_z = self.heading_to_angular(target_yaw, kp_yaw, tolerance, max_ang)

        # LiDAR side-distance correction
        if self.obs_info:
            side_key = "left_min_dist" if self.arc_direction > 0 else "right_min_dist"
            side_dist = self.obs_info.get(side_key)
            if side_dist is not None:
                err = side_dist - float(self.param("trigger_range"))
                angular_z += clamp(-0.45 * err, -0.12, 0.12)

        angular_z = clamp(angular_z, -max_ang, max_ang)
        self.publish_motion(self.avoid_speed_val, angular_z)
        self.avoid_distance_accum += abs(self.last_twist_linear_x) * self.control_period

        # termination: heading reached + obstacle cleared
        yaw_ok = abs(self.angle_error(target_yaw, self.odom_yaw)) < tolerance if self.has_odom else False
        obstacle_clear = obstacle_cleared_for_return(
            self.obs_info, self.arc_direction,
            float(self.param("return_front_clear_range")),
            float(self.param("return_side_release_range")),
        )
        if obstacle_clear:
            self.return_clear_streak += 1
        else:
            self.return_clear_streak = 0

        avoid_elapsed = self.elapsed_seconds(self.avoid_start)
        avoid_timeout = avoid_elapsed > float(self.param("avoid_timeout"))
        ready = ((self.return_clear_streak >= 2
                  and (yaw_ok or not self.has_odom)
                  and self.avoid_distance_accum >= float(self.param("min_avoid_distance"))
                  and avoid_elapsed >= float(self.param("min_avoid_time")))
                 or avoid_timeout)  # safety: force RETURN after timeout regardless of odom

        state = {
            "mode": "AVOID",
            "speed": round(self.last_twist_linear_x, 3),
            "angular_z": round(self.last_twist_angular_z, 3),
            "target_yaw_deg": round(math.degrees(target_yaw), 1),
            "odom_yaw_deg": round(math.degrees(self.odom_yaw), 1) if self.has_odom else None,
            "yaw_ok": yaw_ok,
            "avoid_elapsed": round(avoid_elapsed, 2),
            "distance": round(self.avoid_distance_accum, 3),
            "return_clear_streak": self.return_clear_streak,
            "ready": ready,
            "front_dist": self.obs_info.get("front_min_dist") if self.obs_info else None,
            "traffic_state": self.traffic_state,
        }
        self.debug_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))
        return ready

    # ==================================================================
    #  RETURN  (heading PID: turn back toward original heading)
    # ==================================================================
    def start_return(self):
        self.mode = "RETURN"
        self.return_start = self.get_clock().now()
        self.search_start = None
        self.search_line_streak = 0
        self.get_logger().info(
            f"switching to RETURN, target yaw={math.degrees(self.avoid_origin_yaw):.1f}deg"
        )

    def execute_return(self):
        kp_yaw = float(self.param("return_yaw_kp"))
        tolerance = float(self.param("return_yaw_tolerance"))
        max_ang = float(self.param("avoid_yaw_max_angular"))
        return_speed = float(self.param("return_speed"))

        angular_z = self.heading_to_angular(self.avoid_origin_yaw, kp_yaw, tolerance, max_ang)
        angular_z = clamp(angular_z, -max_ang, max_ang)
        self.publish_motion(return_speed, angular_z)

        yaw_ok = abs(self.angle_error(self.avoid_origin_yaw, self.odom_yaw)) < tolerance if self.has_odom else False
        front_clear = (self.obs_info is None
                       or self.obs_info.get("front_min_dist") is None
                       or self.obs_info["front_min_dist"] > float(self.param("return_front_clear_range")))

        return_elapsed = self.elapsed_seconds(self.return_start)
        if return_elapsed < float(self.param("min_return_time")):
            # still in minimum return period — keep turning, don't exit yet
            pass
        elif yaw_ok and front_clear:
            self.return_clear_streak += 1
        else:
            self.return_clear_streak = 0

        if self.return_clear_streak >= 3:
            self.start_search()
            return

        if return_elapsed > float(self.param("return_timeout")):
            self.get_logger().warn("return timeout, going to SEARCH")
            self.start_search()

        state = {
            "mode": "RETURN",
            "speed": round(self.last_twist_linear_x, 3),
            "angular_z": round(self.last_twist_angular_z, 3),
            "target_yaw_deg": round(math.degrees(self.avoid_origin_yaw), 1),
            "odom_yaw_deg": round(math.degrees(self.odom_yaw), 1) if self.has_odom else None,
            "yaw_ok": yaw_ok,
            "front_clear": front_clear,
            "return_clear_streak": self.return_clear_streak,
            "elapsed": round(self.elapsed_seconds(self.return_start), 2),
            "traffic_state": self.traffic_state,
        }
        self.debug_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))

    # ==================================================================
    #  SEARCH  (visual PID when line visible, heading scan when not)
    # ==================================================================
    def start_search(self):
        self.mode = "SEARCH"
        self.search_start = self.get_clock().now()
        self.return_start = None
        self.search_line_streak = 0
        self._reset_track_state()
        self.get_logger().info("switching to SEARCH")

    def execute_search(self):
        elapsed = self.elapsed_seconds(self.search_start)
        search_timeout = float(self.param("search_timeout"))

        if self.valid_point():
            # visual PID – steer toward the line
            target_lx, target_az = self.compute_lane_pid(self.point.x)
            target_lx = min(target_lx, float(self.param("search_speed")) * 1.5)
            self.publish_motion(target_lx, target_az)

            if self.line_reacquired(ignore_obs=True):
                self.search_line_streak += 1
            else:
                self.search_line_streak = 0

            if self.search_line_streak >= int(self.param("line_reacquire_frames")):
                self.mode = "TRACK"
                self.track_resume_cycles = 1
                self._set_obs_cooldown()
                self.get_logger().info("line re-acquired via PID, back to TRACK")
                return
        else:
            # blind sweep with heading
            turn_dir = -self.arc_direction if self.arc_direction != 0.0 else -1.0
            if elapsed > search_timeout * 0.5:
                turn_dir = -turn_dir

            search_omega = float(self.param("search_omega"))
            if self.has_odom:
                # sweep relative to avoid_origin_yaw
                sweep_target = self.normalize_angle(self.avoid_origin_yaw + turn_dir * 1.2)
                angular_z = self.heading_to_angular(sweep_target, 1.5, 0.1, search_omega * 2.0)
            else:
                angular_z = turn_dir * search_omega

            self.publish_motion(float(self.param("search_speed")), angular_z)
            self.search_line_streak = 0

        if elapsed > search_timeout:
            self.get_logger().warn("search timeout, stopping")
            self.mode = "STOP"

        state = {
            "mode": "SEARCH",
            "speed": round(self.last_twist_linear_x, 3),
            "angular_z": round(self.last_twist_angular_z, 3),
            "search_elapsed": round(elapsed, 2),
            "reacquire_streak": self.search_line_streak,
            "has_line_candidate": self.valid_point(),
            "front_dist": self.obs_info.get("front_min_dist") if self.obs_info else None,
            "traffic_state": self.traffic_state,
        }
        self.debug_pub.publish(String(data=json.dumps(state, ensure_ascii=False)))

    # ==================================================================
    #  main state machine
    # ==================================================================
    def publish_cmd(self):
        # ---- finish line (highest priority) ----
        if self.finish_detected or self.mode == "FINISH":
            self.mode = "FINISH"
            self.publish_stop("finish_detected")
            return

        now_s = self.get_clock().now().nanoseconds / 1e9

        # ---- traffic light: red → force stop ----
        if bool(self.param("traffic_detect_enabled")) and self.traffic_state == "red":
            if self.mode not in ("STOP", "FINISH"):
                self.traffic_stop_reason = True
                self.mode = "STOP"
                self.stop_start = self.get_clock().now()
                self.traffic_cooldown_until = now_s + float(self.param("traffic_cooldown"))
                self._reset_track_state()
                self.get_logger().warn("traffic light RED, forcing STOP")
            self.publish_stop("traffic_red")
            return

        # ---- traffic light: green → resume ----
        if self.traffic_stop_reason and self.traffic_state == "green":
            self.traffic_stop_reason = False
            if self.mode == "STOP":
                self.mode = "TRACK"
                self._reset_track_state()
                self.stop_start = None
                self.traffic_cooldown_until = now_s + float(self.param("traffic_cooldown"))
                self.get_logger().info("traffic light GREEN, resuming to TRACK")

        # ---- master kill switch ----
        if float(self.param("base_speed")) <= 0.0:
            self.publish_stop("speed_zero")
            return

        obs = self.obs_info

        # ==========================
        #  TRACK
        # ==========================
        if self.mode == "TRACK":
            in_cooldown = (self.obs_cooldown_until is not None
                           and now_s < self.obs_cooldown_until)
            if (not in_cooldown
                    and self.obstacle_active(obs)
                    and obs.get("front_min_dist") is not None
                    and obs["front_min_dist"] < float(self.param("slow_range"))):
                self.mode = "APPROACH"
                self.get_logger().info(f"obstacle at {obs['front_min_dist']:.3f}m → APPROACH")
            else:
                self.track_normal()

        # ==========================
        #  APPROACH
        # ==========================
        elif self.mode == "APPROACH":
            if (obs is None or obs.get("front_min_dist") is None
                    or obs["front_min_dist"] > float(self.param("slow_range"))):
                self.mode = "TRACK"
                self._reset_track_state()
                self._set_obs_cooldown()
                self.get_logger().info("obstacle cleared → TRACK")
            elif obs["front_min_dist"] < float(self.param("trigger_range")):
                if not obs.get("escape_clearance_ok", False):
                    self.get_logger().warn("escape path partially blocked, AVOID anyway")
                self.plan_avoid()
                self.mode = "AVOID"
                self.get_logger().info("trigger reached → AVOID")
            else:
                self.approach_slow()

        # ==========================
        #  AVOID
        # ==========================
        elif self.mode == "AVOID":
            ready = self.execute_avoid()
            if ready:
                self.start_return()

        # ==========================
        #  RETURN
        # ==========================
        elif self.mode == "RETURN":
            self.execute_return()

        # ==========================
        #  SEARCH
        # ==========================
        elif self.mode == "SEARCH":
            self.execute_search()

        # ==========================
        #  STOP
        # ==========================
        elif self.mode == "STOP":
            if self.stop_start is None:
                self.stop_start = self.get_clock().now()
            if self.traffic_stop_reason:
                self.publish_stop("traffic_red_waiting")
            elif (self.valid_point()
                  and self.elapsed_seconds(self.stop_start) > float(self.param("stop_cooldown"))):
                self.mode = "TRACK"
                self._reset_track_state()
                self.stop_start = None
                self.track_resume_cycles = 2
                self._set_obs_cooldown()
                self.get_logger().info("auto-recovering STOP → TRACK")
            else:
                self.publish_stop("stopped")

        # ==========================
        #  FINISH
        # ==========================
        elif self.mode == "FINISH":
            self.publish_stop("finish_detected")


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetection()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

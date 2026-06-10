def clamp(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def compute_return_target_yaw(avoid_yaw_accum, return_yaw_ratio):
    return max(0.0, float(avoid_yaw_accum) * float(return_yaw_ratio))


def obstacle_cleared_for_return(obs, arc_direction, front_clear_range, side_release_range):
    if obs is None:
        return True

    front_dist = obs.get("front_min_dist")
    front_clear = front_dist is None or front_dist > float(front_clear_range)

    side_key = "left_min_dist" if arc_direction > 0 else "right_min_dist"
    side_dist = obs.get(side_key)
    side_clear = side_dist is None or side_dist > float(side_release_range)

    return front_clear and side_clear


def avoid_ready_for_return(
    obstacle_clear_stable,
    avoid_yaw_accum,
    avoid_distance_accum,
    avoid_elapsed,
    min_avoid_yaw,
    min_avoid_distance,
    min_avoid_time,
):
    return (
        bool(obstacle_clear_stable)
        and float(avoid_yaw_accum) >= float(min_avoid_yaw)
        and float(avoid_distance_accum) >= float(min_avoid_distance)
        and float(avoid_elapsed) >= float(min_avoid_time)
    )


def should_enter_search(return_yaw_accum, return_target_yaw, obs, front_clear_range):
    if float(return_yaw_accum) < float(return_target_yaw):
        return False

    if obs is None:
        return True

    front_dist = obs.get("front_min_dist")
    return front_dist is None or front_dist > float(front_clear_range)


def compute_return_command(
    obs,
    arc_direction,
    arc_omega,
    return_speed,
    return_omega_scale,
    correction_gain=0.25,
    max_correction=0.05,
):
    direction = 1.0 if arc_direction >= 0.0 else -1.0
    base_omega = -direction * abs(float(arc_omega)) * float(return_omega_scale)

    correction = 0.0
    if obs is not None:
        left_dist = obs.get("left_min_dist")
        right_dist = obs.get("right_min_dist")
        if left_dist is not None and right_dist is not None:
            correction = clamp(
                (float(left_dist) - float(right_dist)) * float(correction_gain),
                -float(max_correction),
                float(max_correction),
            )

    return float(return_speed), base_omega + correction, correction


def _point_xy(point):
    if point is None:
        return None, None
    if isinstance(point, dict):
        return point.get("x"), point.get("y")
    if isinstance(point, (tuple, list)) and len(point) >= 2:
        return point[0], point[1]
    return getattr(point, "x", None), getattr(point, "y", None)


def is_reacquired_line_candidate(
    point,
    point_age,
    target_x,
    line_reacquire_error_px,
    line_reacquire_min_y,
    point_timeout,
    obs,
):
    if point is None or point_age is None or float(point_age) > float(point_timeout):
        return False

    point_x, point_y = _point_xy(point)
    if point_x is None or point_y is None:
        return False

    if abs(float(point_x) - float(target_x)) > float(line_reacquire_error_px):
        return False

    if float(point_y) < float(line_reacquire_min_y):
        return False

    if obs:
        if obs.get("front_blocked", False) or obs.get("has_obs", False):
            return False

    return True

#!/usr/bin/env python3
from dataclasses import dataclass

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


@dataclass(frozen=True)
class CheckerboardConfig:
    roi_top_ratio: float = 0.45
    roi_bottom_ratio: float = 0.95
    roi_left_ratio: float = 0.12
    roi_right_ratio: float = 0.88
    grid_cols: int = 10
    grid_rows: int = 6
    min_contrast: float = 55.0
    min_white_ratio: float = 0.25
    max_white_ratio: float = 0.75
    min_transition_ratio: float = 0.58
    min_strong_cells: int = 18


def clamp_ratio(value, default):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(value, 0.0), 1.0)


def detect_checkerboard_finish(image_bgr, config=None):
    """Return (detected, metrics) for a black/white checkerboard finish marker."""
    if config is None:
        config = CheckerboardConfig()

    if cv2 is None or np is None:
        return False, {"reason": "opencv_unavailable"}
    if image_bgr is None or image_bgr.size == 0:
        return False, {"reason": "empty_image"}

    height, width = image_bgr.shape[:2]
    top = int(height * clamp_ratio(config.roi_top_ratio, 0.45))
    bottom = int(height * clamp_ratio(config.roi_bottom_ratio, 0.95))
    left = int(width * clamp_ratio(config.roi_left_ratio, 0.12))
    right = int(width * clamp_ratio(config.roi_right_ratio, 0.88))

    if bottom <= top or right <= left:
        return False, {"reason": "invalid_roi"}

    roi = image_bgr[top:bottom, left:right]
    if roi.size == 0:
        return False, {"reason": "empty_roi"}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(gray, (config.grid_cols * 12, config.grid_rows * 12))

    cell_means = np.zeros((config.grid_rows, config.grid_cols), dtype=np.float32)
    cell_h = resized.shape[0] // config.grid_rows
    cell_w = resized.shape[1] // config.grid_cols
    for row in range(config.grid_rows):
        for col in range(config.grid_cols):
            cell = resized[
                row * cell_h:(row + 1) * cell_h,
                col * cell_w:(col + 1) * cell_w,
            ]
            cell_means[row, col] = float(np.mean(cell))

    dark_level = float(np.percentile(cell_means, 20))
    light_level = float(np.percentile(cell_means, 80))
    contrast = light_level - dark_level
    threshold = (dark_level + light_level) * 0.5
    white_cells = cell_means >= threshold
    white_ratio = float(np.mean(white_cells))

    strong_dark = cell_means <= dark_level + max(contrast * 0.25, 8.0)
    strong_light = cell_means >= light_level - max(contrast * 0.25, 8.0)
    strong_cells = int(np.count_nonzero(strong_dark | strong_light))

    horizontal_changes = white_cells[:, 1:] != white_cells[:, :-1]
    vertical_changes = white_cells[1:, :] != white_cells[:-1, :]
    transition_count = int(np.count_nonzero(horizontal_changes) + np.count_nonzero(vertical_changes))
    transition_total = int(horizontal_changes.size + vertical_changes.size)
    transition_ratio = transition_count / max(transition_total, 1)

    detected = (
        contrast >= config.min_contrast
        and config.min_white_ratio <= white_ratio <= config.max_white_ratio
        and transition_ratio >= config.min_transition_ratio
        and strong_cells >= config.min_strong_cells
    )

    return detected, {
        "contrast": contrast,
        "white_ratio": white_ratio,
        "transition_ratio": transition_ratio,
        "transition_count": transition_count,
        "strong_cells": strong_cells,
        "roi": [left, top, right, bottom],
    }


def decode_compressed_image(data):
    if cv2 is None or np is None:
        return None
    image_array = np.frombuffer(data, dtype=np.uint8)
    if image_array.size == 0:
        return None
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

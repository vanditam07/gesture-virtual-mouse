from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ALL_ACTIONS, GestureConfig, save_config

_WINDOW = "Gesture Settings"
_WIDTH = 480
_ROW_H = 28
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_THICK = 1

_SLIDER_DEFS = [
    ("Cursor Speed", "cursor_speed", 1, 30, 10.0, "x"),
    ("Dead Zone", "dead_zone", 0, 100, 1, "px^2"),
    ("Mid Range", "mid_range", 100, 2500, 1, "px^2"),
    ("Debounce", "debounce_frames", 1, 10, 1, "frames"),
    ("Pinch Sens.", "pinch_threshold", 1, 10, 10.0, ""),
]

_ACTION_LABELS = {
    "move_cursor": "Move Cursor",
    "drag": "Drag",
    "left_click": "Left Click",
    "right_click": "Right Click",
    "double_click": "Double Click",
    "scroll": "Scroll",
    "volume_brightness": "Vol / Bright",
}


def _noop(_val: int) -> None:
    pass


def _draw_panel(config: GestureConfig) -> np.ndarray:
    """Render a settings summary image to display in the window."""
    section_gap = 12
    header_h = 30

    n_sliders = len(_SLIDER_DEFS)
    n_toggles = len(ALL_ACTIONS)
    total_h = header_h + n_sliders * _ROW_H + section_gap + header_h + n_toggles * _ROW_H + 16

    img = np.zeros((total_h, _WIDTH, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)

    y = 0

    # --- Sensitivity header ---
    cv2.rectangle(img, (0, y), (_WIDTH, y + header_h), (70, 50, 50), -1)
    cv2.putText(img, "SENSITIVITY", (10, y + 20), _FONT, 0.55, (180, 200, 255), 1, cv2.LINE_AA)
    y += header_h

    for label, attr, lo, hi, divisor, unit in _SLIDER_DEFS:
        raw = getattr(config, attr)
        display_val = f"{raw:.1f}" if isinstance(raw, float) else str(raw)
        if unit:
            display_val += f" {unit}"

        frac = 0.0
        if hi != lo:
            raw_int = int(raw * divisor) if isinstance(divisor, float) else int(raw)
            frac = max(0.0, min(1.0, (raw_int - lo) / (hi - lo)))

        cv2.putText(img, label, (10, y + 18), _FONT, _FONT_SCALE, (200, 200, 200), _THICK, cv2.LINE_AA)

        bar_x = 130
        bar_w = _WIDTH - 130 - 80
        bar_y = y + 8
        bar_h = 12
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill_w = int(bar_w * frac)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (100, 180, 255), -1)

        cv2.putText(img, display_val, (_WIDTH - 75, y + 18), _FONT, _FONT_SCALE, (220, 220, 220), _THICK, cv2.LINE_AA)
        y += _ROW_H

    y += section_gap

    # --- Toggles header ---
    cv2.rectangle(img, (0, y), (_WIDTH, y + header_h), (50, 70, 50), -1)
    cv2.putText(img, "GESTURE TOGGLES", (10, y + 20), _FONT, 0.55, (180, 255, 180), 1, cv2.LINE_AA)
    y += header_h

    for action in ALL_ACTIONS:
        enabled = config.enabled_actions.get(action, True)
        nice = _ACTION_LABELS.get(action, action)

        dot_color = (80, 220, 80) if enabled else (80, 80, 80)
        label_color = (220, 220, 220) if enabled else (120, 120, 120)
        status = "ON" if enabled else "OFF"
        status_color = (80, 220, 80) if enabled else (100, 100, 220)

        cv2.circle(img, (20, y + 14), 6, dot_color, -1)
        cv2.putText(img, nice, (34, y + 18), _FONT, _FONT_SCALE, label_color, _THICK, cv2.LINE_AA)
        cv2.putText(img, status, (_WIDTH - 50, y + 18), _FONT, _FONT_SCALE, status_color, _THICK, cv2.LINE_AA)
        y += _ROW_H

    return img


class TrackbarUI:
    """OpenCV window with trackbars + a rendered settings summary panel."""

    def __init__(self, config: GestureConfig, config_path: Path) -> None:
        self._config_path = config_path

        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(_WINDOW, _WIDTH, 520)

        for label, attr, lo, hi, divisor, _unit in _SLIDER_DEFS:
            raw = getattr(config, attr)
            initial = int(raw * divisor) if isinstance(divisor, float) else int(raw)
            initial = max(lo, min(hi, initial))
            cv2.createTrackbar(label, _WINDOW, initial, hi, _noop)

        for action in ALL_ACTIONS:
            nice = _ACTION_LABELS.get(action, action)
            cv2.createTrackbar(nice, _WINDOW, int(config.enabled_actions.get(action, True)), 1, _noop)

        cv2.imshow(_WINDOW, _draw_panel(config))

    def sync(self, config: GestureConfig) -> bool:
        """Read trackbar positions into *config*. Auto-saves and redraws on change."""
        changed = False

        for label, attr, lo, hi, divisor, _unit in _SLIDER_DEFS:
            raw_val = cv2.getTrackbarPos(label, _WINDOW)
            if raw_val < lo:
                raw_val = lo
            value = raw_val / divisor if isinstance(divisor, float) else raw_val
            if getattr(config, attr) != value:
                setattr(config, attr, value)
                changed = True

        for action in ALL_ACTIONS:
            nice = _ACTION_LABELS.get(action, action)
            val = bool(cv2.getTrackbarPos(nice, _WINDOW))
            if config.enabled_actions.get(action) != val:
                config.enabled_actions[action] = val
                changed = True

        if changed:
            save_config(config, self._config_path)
            cv2.imshow(_WINDOW, _draw_panel(config))

        return changed

    @staticmethod
    def destroy() -> None:
        try:
            cv2.destroyWindow(_WINDOW)
        except cv2.error:
            pass

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import ALL_ACTIONS, GestureConfig, save_config

_CTRL_WIN = "Gesture Controls"
_INFO_WIN = "Settings Guide"
_FONT = cv2.FONT_HERSHEY_SIMPLEX

_SLIDER_DEFS = [
    ("Speed", "cursor_speed", 1, 30, 10.0, "x"),
    ("DeadZn", "dead_zone", 0, 100, 1, "px\u00b2"),
    ("MidRng", "mid_range", 100, 2500, 1, "px\u00b2"),
    ("Dbnce", "debounce_frames", 1, 10, 1, "frames"),
    ("Pinch", "pinch_threshold", 1, 10, 10.0, ""),
]

_TOGGLE_KEYS = [
    ("MC", "move_cursor"),
    ("DR", "drag"),
    ("LC", "left_click"),
    ("RC", "right_click"),
    ("DC", "double_click"),
    ("SC", "scroll"),
    ("VB", "volume_brightness"),
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

_SLIDER_HELP = {
    "cursor_speed": ("Cursor Speed", "How fast the cursor tracks your hand"),
    "dead_zone": ("Dead Zone", "Ignores small hand jitter below this threshold"),
    "mid_range": ("Mid Range", "Smooth acceleration zone before full speed"),
    "debounce_frames": ("Debounce", "Frames a gesture must hold before it registers"),
    "pinch_threshold": ("Pinch Sens.", "Sensitivity for pinch scroll / volume / brightness"),
}

_TOGGLE_HELP = {
    "move_cursor": "V-gesture moves the mouse pointer",
    "drag": "Fist gesture holds left-click and drags",
    "left_click": "Middle finger triggers a left click",
    "right_click": "Index finger triggers a right click",
    "double_click": "Two fingers closed triggers a double click",
    "scroll": "Minor-hand pinch scrolls vertically / horizontally",
    "volume_brightness": "Major-hand pinch adjusts volume / brightness",
}


def _noop(_val: int) -> None:
    pass


def _draw_ctrl_strip() -> np.ndarray:
    """Tiny image for the trackbar window so OpenCV has something to display."""
    img = np.full((24, 300, 3), 50, dtype=np.uint8)
    cv2.putText(img, "Adjust sliders above  |  See 'Settings Guide'",
                (6, 16), _FONT, 0.36, (170, 170, 170), 1, cv2.LINE_AA)
    return img


def _get_info_size():
    """Get the info window's current image rect, or a default."""
    try:
        rect = cv2.getWindowImageRect(_INFO_WIN)
        if rect[2] > 100 and rect[3] > 100:
            return rect[2], rect[3]
    except Exception:
        pass
    return 520, 700


def _draw_info(config: GestureConfig, width: int, height: int) -> np.ndarray:
    """Render the full settings guide panel, scaled to fit the window."""
    n_sliders = len(_SLIDER_DEFS)
    n_toggles = len(_TOGGLE_KEYS)
    total_rows = 1 + n_sliders + 1 + n_toggles  # headers + items
    content_units = total_rows * 2 + 2  # 2 lines per row + padding
    row_h = max(20, height // content_units)

    scale = row_h / 26.0
    f_label = max(0.35, 0.44 * scale)
    f_desc = max(0.30, 0.36 * scale)
    f_header = max(0.40, 0.50 * scale)
    f_val = max(0.35, 0.42 * scale)
    pad = max(8, int(10 * scale))
    bar_h = max(6, int(10 * scale))
    dot_r = max(3, int(5 * scale))
    gap = max(4, int(8 * scale))

    total_h = max(height, (total_rows * row_h) + row_h)
    img = np.full((total_h, width, 3), 40, dtype=np.uint8)
    y = 0

    # ── SENSITIVITY header ──
    hdr_h = int(row_h * 0.7)
    cv2.rectangle(img, (0, y), (width, y + hdr_h), (70, 50, 50), -1)
    cv2.putText(img, "SENSITIVITY", (pad, y + hdr_h - int(6 * scale)),
                _FONT, f_header, (180, 200, 255), 1, cv2.LINE_AA)
    y += hdr_h + gap

    bar_start = int(width * 0.28)
    bar_end_x = int(width * 0.72)
    val_col = int(width * 0.75)

    for _tb_label, attr, lo, hi, divisor, unit in _SLIDER_DEFS:
        full_name, desc = _SLIDER_HELP[attr]
        raw = getattr(config, attr)
        display = f"{raw:.1f}" if isinstance(raw, float) else str(raw)
        if unit:
            display += f" {unit}"

        frac = 0.0
        if hi != lo:
            ri = int(raw * divisor) if isinstance(divisor, float) else int(raw)
            frac = max(0.0, min(1.0, (ri - lo) / (hi - lo)))

        line1_y = y + int(row_h * 0.45)
        cv2.putText(img, full_name, (pad, line1_y),
                    _FONT, f_label, (210, 210, 210), 1, cv2.LINE_AA)

        bw = bar_end_x - bar_start
        by = y + int(row_h * 0.2)
        cv2.rectangle(img, (bar_start, by), (bar_start + bw, by + bar_h), (80, 80, 80), -1)
        cv2.rectangle(img, (bar_start, by), (bar_start + int(bw * frac), by + bar_h), (100, 180, 255), -1)

        cv2.putText(img, display, (val_col, line1_y),
                    _FONT, f_val, (220, 220, 220), 1, cv2.LINE_AA)

        line2_y = y + int(row_h * 0.85)
        cv2.putText(img, desc, (pad + int(4 * scale), line2_y),
                    _FONT, f_desc, (130, 130, 130), 1, cv2.LINE_AA)
        y += row_h

    y += gap

    # ── GESTURE TOGGLES header ──
    cv2.rectangle(img, (0, y), (width, y + hdr_h), (50, 70, 50), -1)
    cv2.putText(img, "GESTURE TOGGLES", (pad, y + hdr_h - int(6 * scale)),
                _FONT, f_header, (180, 255, 180), 1, cv2.LINE_AA)
    y += hdr_h + gap

    status_col = int(width * 0.82)

    for _key, action in _TOGGLE_KEYS:
        enabled = config.enabled_actions.get(action, True)
        nice = _ACTION_LABELS.get(action, action)
        desc = _TOGGLE_HELP.get(action, "")

        dot_col = (80, 220, 80) if enabled else (80, 80, 80)
        txt_col = (220, 220, 220) if enabled else (120, 120, 120)
        status = "ON" if enabled else "OFF"
        st_col = (80, 220, 80) if enabled else (100, 100, 220)

        line1_y = y + int(row_h * 0.45)
        cx = pad + dot_r + 2
        cv2.circle(img, (cx, y + int(row_h * 0.35)), dot_r, dot_col, -1)
        name_x = cx + dot_r + int(8 * scale)
        cv2.putText(img, nice, (name_x, line1_y),
                    _FONT, f_label, txt_col, 1, cv2.LINE_AA)
        cv2.putText(img, status, (status_col, line1_y),
                    _FONT, f_val, st_col, 1, cv2.LINE_AA)

        line2_y = y + int(row_h * 0.85)
        cv2.putText(img, desc, (name_x, line2_y),
                    _FONT, f_desc, (130, 130, 130), 1, cv2.LINE_AA)
        y += row_h

    return img[:max(1, min(total_h, y + gap))]


class TrackbarUI:
    """Two-window UI: compact trackbar controls + a resizable info guide."""

    def __init__(self, config: GestureConfig, config_path: Path) -> None:
        self._config_path = config_path
        self._last_size = (0, 0)

        cv2.namedWindow(_CTRL_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_CTRL_WIN, 340, 620)

        for tb_label, attr, lo, hi, divisor, _unit in _SLIDER_DEFS:
            raw = getattr(config, attr)
            init = int(raw * divisor) if isinstance(divisor, float) else int(raw)
            cv2.createTrackbar(tb_label, _CTRL_WIN, max(lo, min(hi, init)), hi, _noop)

        for key, action in _TOGGLE_KEYS:
            cv2.createTrackbar(key, _CTRL_WIN, int(config.enabled_actions.get(action, True)), 1, _noop)

        cv2.imshow(_CTRL_WIN, _draw_ctrl_strip())

        cv2.namedWindow(_INFO_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_INFO_WIN, 520, 700)
        w, h = _get_info_size()
        self._last_size = (w, h)
        cv2.imshow(_INFO_WIN, _draw_info(config, w, h))

    def sync(self, config: GestureConfig) -> bool:
        """Read trackbar values into *config*. Auto-saves & redraws on change."""
        changed = False

        for tb_label, attr, lo, hi, divisor, _unit in _SLIDER_DEFS:
            raw_val = max(lo, cv2.getTrackbarPos(tb_label, _CTRL_WIN))
            value = raw_val / divisor if isinstance(divisor, float) else raw_val
            if getattr(config, attr) != value:
                setattr(config, attr, value)
                changed = True

        for key, action in _TOGGLE_KEYS:
            val = bool(cv2.getTrackbarPos(key, _CTRL_WIN))
            if config.enabled_actions.get(action) != val:
                config.enabled_actions[action] = val
                changed = True

        w, h = _get_info_size()
        resized = (w, h) != self._last_size

        if changed:
            save_config(config, self._config_path)

        if changed or resized:
            self._last_size = (w, h)
            cv2.imshow(_INFO_WIN, _draw_info(config, w, h))

        return changed

    @staticmethod
    def destroy() -> None:
        for win in (_CTRL_WIN, _INFO_WIN):
            try:
                cv2.destroyWindow(win)
            except cv2.error:
                pass

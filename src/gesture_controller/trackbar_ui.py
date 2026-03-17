from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import ALL_ACTIONS, GestureConfig, save_config

_WINDOW = "Gesture Settings"

_SLIDER_DEFS = {
    "Cursor Speed": ("cursor_speed", 1, 30, 10.0),
    "Dead Zone": ("dead_zone", 0, 100, 1),
    "Mid Range": ("mid_range", 100, 2500, 1),
    "Debounce Frames": ("debounce_frames", 1, 10, 1),
    "Pinch Threshold": ("pinch_threshold", 1, 10, 10.0),
}

_TOGGLE_PREFIX = "Enable "


def _noop(_val: int) -> None:
    pass


class TrackbarUI:
    """Creates an OpenCV window with trackbars bound to a GestureConfig."""

    def __init__(self, config: GestureConfig, config_path: Path) -> None:
        self._config_path = config_path
        self._prev_snapshot: Optional[dict] = None

        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WINDOW, 420, 50)

        for label, (attr, lo, hi, divisor) in _SLIDER_DEFS.items():
            raw = getattr(config, attr)
            initial = int(raw * divisor) if isinstance(divisor, float) else int(raw)
            initial = max(lo, min(hi, initial))
            cv2.createTrackbar(label, _WINDOW, initial, hi, _noop)

        for action in ALL_ACTIONS:
            label = _TOGGLE_PREFIX + action
            cv2.createTrackbar(label, _WINDOW, int(config.enabled_actions.get(action, True)), 1, _noop)

        blank = np.zeros((1, 420, 3), dtype=np.uint8)
        cv2.imshow(_WINDOW, blank)

    def sync(self, config: GestureConfig) -> bool:
        """
        Read trackbar positions into *config*.
        Returns True if anything changed (and auto-saves).
        """
        changed = False

        for label, (attr, lo, hi, divisor) in _SLIDER_DEFS.items():
            raw_val = cv2.getTrackbarPos(label, _WINDOW)
            if raw_val < lo:
                raw_val = lo
            value = raw_val / divisor if isinstance(divisor, float) else raw_val
            if getattr(config, attr) != value:
                setattr(config, attr, value)
                changed = True

        for action in ALL_ACTIONS:
            label = _TOGGLE_PREFIX + action
            val = bool(cv2.getTrackbarPos(label, _WINDOW))
            if config.enabled_actions.get(action) != val:
                config.enabled_actions[action] = val
                changed = True

        if changed:
            save_config(config, self._config_path)

        return changed

    @staticmethod
    def destroy() -> None:
        try:
            cv2.destroyWindow(_WINDOW)
        except cv2.error:
            pass

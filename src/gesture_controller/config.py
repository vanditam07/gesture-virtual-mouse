from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

from .enums import Gest

ALL_ACTIONS = (
    "move_cursor",
    "drag",
    "left_click",
    "right_click",
    "double_click",
    "scroll",
    "volume_brightness",
)

_DEFAULT_MAPPINGS: Dict[str, str] = {
    "move_cursor": "V_GEST",
    "drag": "FIST",
    "left_click": "MID",
    "right_click": "INDEX",
    "double_click": "TWO_FINGER_CLOSED",
    "scroll": "PINCH_MINOR",
    "volume_brightness": "PINCH_MAJOR",
}

_DEFAULT_ENABLED: Dict[str, bool] = {action: True for action in ALL_ACTIONS}


@dataclass
class GestureConfig:
    gesture_mappings: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_MAPPINGS))
    enabled_actions: Dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_ENABLED))
    cursor_speed: float = 1.0
    dead_zone: int = 25
    mid_range: int = 900
    move_duration: float = 0.1
    pinch_threshold: float = 0.3
    debounce_frames: int = 4

    @classmethod
    def default(cls) -> GestureConfig:
        return cls()

    def resolve_gesture(self, action_name: str) -> Optional[int]:
        """Return the Gest int value mapped to *action_name*, or None."""
        gest_name = self.gesture_mappings.get(action_name)
        if gest_name is None:
            return None
        try:
            return int(Gest[gest_name])
        except KeyError:
            return None

    def build_reverse_map(self) -> Dict[int, str]:
        """Return {gest_int: action_name} for fast lookup each frame."""
        result: Dict[int, str] = {}
        for action, gest_name in self.gesture_mappings.items():
            gest_val = self.resolve_gesture(action)
            if gest_val is not None:
                result[gest_val] = action
        return result

    def is_enabled(self, action_name: str) -> bool:
        return self.enabled_actions.get(action_name, False)


def load_config(path: Path) -> GestureConfig:
    if not path.exists():
        cfg = GestureConfig.default()
        save_config(cfg, path)
        return cfg

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    defaults = asdict(GestureConfig.default())
    for key in defaults:
        if key not in raw:
            raw[key] = defaults[key]

    return GestureConfig(
        gesture_mappings=raw["gesture_mappings"],
        enabled_actions=raw["enabled_actions"],
        cursor_speed=float(raw["cursor_speed"]),
        dead_zone=int(raw["dead_zone"]),
        mid_range=int(raw["mid_range"]),
        move_duration=float(raw["move_duration"]),
        pinch_threshold=float(raw["pinch_threshold"]),
        debounce_frames=int(raw["debounce_frames"]),
    )


def save_config(config: GestureConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(config), fh, indent=2)

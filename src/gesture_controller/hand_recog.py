from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

from .enums import Gest, HLabel

if TYPE_CHECKING:
    from .config import GestureConfig


class HandRecog:
    """
    Convert MediaPipe landmarks to a stable gesture signal.
    """

    def __init__(self, hand_label: HLabel):
        self.finger = 0
        self.ori_gesture: int = int(Gest.PALM)
        self.prev_gesture: int = int(Gest.PALM)
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label

    def update_hand_result(self, hand_result) -> None:
        self.hand_result = hand_result

    def get_signed_dist(self, point) -> float:
        sign = -1
        if self.hand_result.landmark[point[0]].y < self.hand_result.landmark[point[1]].y:
            sign = 1
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        return math.sqrt(dist) * sign

    def get_dist(self, point) -> float:
        dist = (self.hand_result.landmark[point[0]].x - self.hand_result.landmark[point[1]].x) ** 2
        dist += (self.hand_result.landmark[point[0]].y - self.hand_result.landmark[point[1]].y) ** 2
        return math.sqrt(dist)

    def get_dz(self, point) -> float:
        return abs(self.hand_result.landmark[point[0]].z - self.hand_result.landmark[point[1]].z)

    def set_finger_state(self) -> None:
        """
        Compute a 4-bit encoding for open/closed fingers (excluding thumb).
        """
        if self.hand_result is None:
            return

        points = [[8, 5, 0], [12, 9, 0], [16, 13, 0], [20, 17, 0]]
        self.finger = 0
        self.finger = self.finger | 0  # thumb

        for point in points:
            dist = self.get_signed_dist(point[:2])
            dist2 = self.get_signed_dist(point[1:])
            try:
                ratio = round(dist / dist2, 1)
            except Exception:
                ratio = round(dist / 0.01, 1)

            self.finger = self.finger << 1
            if ratio > 0.5:
                self.finger = self.finger | 1

    def get_gesture(self, cfg: Optional[GestureConfig] = None) -> int:
        """
        Return a stabilized gesture value (debounced across frames).
        Uses cfg.debounce_frames when provided.
        """
        if self.hand_result is None:
            return int(Gest.PALM)

        debounce = cfg.debounce_frames if cfg is not None else 4

        current_gesture: int = int(Gest.PALM)
        if self.finger in [Gest.LAST3, Gest.LAST4] and self.get_dist([8, 4]) < 0.05:
            current_gesture = int(Gest.PINCH_MINOR if self.hand_label == HLabel.MINOR else Gest.PINCH_MAJOR)
        elif Gest.FIRST2 == self.finger:
            point = [[8, 12], [5, 9]]
            dist1 = self.get_dist(point[0])
            dist2 = self.get_dist(point[1])
            ratio = dist1 / dist2
            if ratio > 1.7:
                current_gesture = int(Gest.V_GEST)
            else:
                current_gesture = int(Gest.TWO_FINGER_CLOSED if self.get_dz([8, 12]) < 0.1 else Gest.MID)
        else:
            current_gesture = int(self.finger)

        if current_gesture == self.prev_gesture:
            self.frame_count += 1
        else:
            self.frame_count = 0
        self.prev_gesture = current_gesture

        if self.frame_count > debounce:
            self.ori_gesture = current_gesture
        return self.ori_gesture


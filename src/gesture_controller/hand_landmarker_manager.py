from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True)
class HandDetection:
    left: Optional[object]
    right: Optional[object]


class _HandResultAdapter:
    def __init__(self, landmarks):
        self.landmark = landmarks


class HandLandmarkerManager:
    def __init__(
        self,
        model_path: Path,
        model_url: str = _DEFAULT_MODEL_URL,
        num_hands: int = 2,
    ) -> None:
        self._model_path = model_path
        self._model_url = model_url
        self._num_hands = num_hands
        self._landmarker = None

    def ensure_ready(self) -> None:
        self._ensure_model_file()
        if self._landmarker is None:
            base_options = mp_python.BaseOptions(model_asset_path=str(self._model_path))
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self._num_hands,
                running_mode=mp_vision.RunningMode.VIDEO,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def detect_bgr(self, frame_bgr, timestamp_ms: Optional[int] = None) -> HandDetection:
        self.ensure_ready()
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        by_label: Dict[str, object] = {}
        for idx, landmarks in enumerate(result.hand_landmarks or []):
            label = self._extract_label(result, idx) or f"hand_{idx}"
            by_label[label] = _HandResultAdapter(landmarks)

        # Normalize keys to "Left"/"Right" when possible.
        left = by_label.get("Left") or by_label.get("left")
        right = by_label.get("Right") or by_label.get("right")
        return HandDetection(left=left, right=right)

    def _extract_label(self, result, idx: int) -> Optional[str]:
        try:
            handedness_list = result.handedness[idx]
            if not handedness_list:
                return None
            cat = handedness_list[0]
            return getattr(cat, "category_name", None) or getattr(cat, "display_name", None)
        except Exception:
            return None

    def _ensure_model_file(self) -> None:
        if self._model_path.exists():
            return
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(self._model_url, self._model_path)


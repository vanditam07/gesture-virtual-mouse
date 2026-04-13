"""Personalisation Wizard: guided enrolment to record user-specific gesture prototypes.

Flow per gesture class:
    1. Display gesture name + visual prompt
    2. 3-second countdown
    3. Capture 3-5 seconds of frames where MediaPipe confidence > 0.7
    4. Extract 77-dim vector per frame -> frozen encoder -> 64-dim embedding
    5. Buffer embeddings

After all classes: compute mean prototype per class -> save user_prototypes.npy
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .encoder import EMBEDDING_DIM, ProtoEncoder
from .feature_extraction import extract_feature_vector
from .gesture_templates import CLASS_NAMES, NUM_GESTURE_CLASSES

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WINDOW = "Personalisation Wizard"

_GESTURE_INSTRUCTIONS = {
    "PALM": "Open your hand wide, all fingers extended",
    "V_GEST": "Extend index + middle finger in a V shape",
    "FIST": "Close all fingers into a tight fist",
    "MID": "Extend only your middle finger",
    "INDEX": "Extend only your index finger (pointing)",
    "TWO_FINGER_CLOSED": "Extend index + middle finger, held together",
    "PINCH_MINOR": "Pinch thumb + index, other fingers open",
    "PINCH_MAJOR": "Pinch thumb + index, other fingers curled",
}

_MIN_SAMPLES = 10
_COUNTDOWN_SECS = 3
_CAPTURE_SECS = 4


class PersonalisationWizard:
    """OpenCV-based enrolment UI that records gesture prototypes."""

    def __init__(
        self,
        encoder_path: Path,
        save_path: Path,
        landmarker_manager,
    ) -> None:
        self._encoder = ProtoEncoder.load_checkpoint(encoder_path)
        self._save_path = save_path
        self._landmarker = landmarker_manager

    def run(self, cap: cv2.VideoCapture) -> bool:
        """Execute the full enrolment flow.

        Returns True if prototypes were saved successfully.
        """
        cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WINDOW, 720, 540)

        all_prototypes = np.zeros((NUM_GESTURE_CLASSES, EMBEDDING_DIM), dtype=np.float32)

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            embeddings = self._record_class(cap, cls_idx, cls_name)
            if embeddings is None:
                self._show_message(cap, "Enrolment cancelled.", duration=2.0)
                cv2.destroyWindow(_WINDOW)
                return False
            all_prototypes[cls_idx] = embeddings.mean(axis=0)

        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._save_path, all_prototypes)

        self._show_message(
            cap,
            f"Personalisation complete!  Prototypes saved ({NUM_GESTURE_CLASSES} classes).",
            duration=3.0,
        )
        cv2.destroyWindow(_WINDOW)
        return True

    def _record_class(
        self,
        cap: cv2.VideoCapture,
        cls_idx: int,
        cls_name: str,
    ) -> Optional[np.ndarray]:
        instruction = _GESTURE_INSTRUCTIONS.get(cls_name, cls_name)
        self._show_prompt(cap, cls_idx, cls_name, instruction)

        if not self._countdown(cap, cls_name):
            return None

        return self._capture_embeddings(cap, cls_name)

    def _show_prompt(
        self, cap: cv2.VideoCapture, cls_idx: int, cls_name: str, instruction: str
    ) -> None:
        """Show gesture name + instruction, wait for spacebar."""
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            header = f"Gesture {cls_idx + 1}/{NUM_GESTURE_CLASSES}:  {cls_name}"
            cv2.putText(frame, header, (20, 35), _FONT, 0.8, (100, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, instruction, (20, 70), _FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, "Press SPACE when ready  |  ESC to cancel", (20, 105), _FONT, 0.5, (150, 150, 255), 1, cv2.LINE_AA)

            cv2.imshow(_WINDOW, frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                return
            if key == 32:
                return

    def _countdown(self, cap: cv2.VideoCapture, cls_name: str) -> bool:
        """3-second countdown. Returns False if user presses ESC."""
        start = time.monotonic()
        while True:
            elapsed = time.monotonic() - start
            remaining = _COUNTDOWN_SECS - elapsed
            if remaining <= 0:
                return True

            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            text = str(int(remaining) + 1)
            sz = cv2.getTextSize(text, _FONT, 3.0, 4)[0]
            cx = (frame.shape[1] - sz[0]) // 2
            cy = (frame.shape[0] + sz[1]) // 2
            cv2.putText(frame, text, (cx, cy), _FONT, 3.0, (0, 200, 255), 4, cv2.LINE_AA)
            cv2.putText(frame, f"Recording {cls_name} soon...", (20, 30), _FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(_WINDOW, frame)
            if cv2.waitKey(30) & 0xFF == 27:
                return False

    def _capture_embeddings(self, cap: cv2.VideoCapture, cls_name: str) -> Optional[np.ndarray]:
        """Capture frames for CAPTURE_SECS, extract embeddings."""
        embeddings = []
        start = time.monotonic()

        while time.monotonic() - start < _CAPTURE_SECS:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)

            ts_ms = int(time.monotonic() * 1000)
            detection = self._landmarker.detect_bgr(frame, timestamp_ms=ts_ms)
            hand = detection.right or detection.left
            elapsed = time.monotonic() - start
            progress = min(1.0, elapsed / _CAPTURE_SECS)

            bar_w = int(frame.shape[1] * 0.8)
            bar_x = int(frame.shape[1] * 0.1)
            bar_y = frame.shape[0] - 40
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (80, 80, 80), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 20), (80, 220, 80), -1)

            status = f"Recording {cls_name}...  samples: {len(embeddings)}"
            cv2.putText(frame, status, (20, 30), _FONT, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

            if hand is not None:
                fv = extract_feature_vector(hand.landmark)
                emb = self._encoder_embed(fv)
                embeddings.append(emb)
                cv2.putText(frame, "HAND DETECTED", (20, 60), _FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No hand - show your hand!", (20, 60), _FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow(_WINDOW, frame)
            if cv2.waitKey(5) & 0xFF == 27:
                return None

        if len(embeddings) < _MIN_SAMPLES:
            self._show_message(
                cap,
                f"Only {len(embeddings)} samples captured (need {_MIN_SAMPLES}+). Retrying...",
                duration=2.0,
            )
            return self._capture_embeddings(cap, cls_name)

        return np.stack(embeddings)

    def _encoder_embed(self, feature_vector: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            x = torch.from_numpy(feature_vector).float().unsqueeze(0)
            return self._encoder(x).squeeze(0).numpy()

    def _show_message(self, cap: cv2.VideoCapture, text: str, duration: float = 2.0) -> None:
        start = time.monotonic()
        while time.monotonic() - start < duration:
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = cv2.flip(frame, 1)
            cv2.putText(frame, text, (20, frame.shape[0] // 2), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(_WINDOW, frame)
            cv2.waitKey(30)

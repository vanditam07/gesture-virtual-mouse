from __future__ import annotations

import time
from pathlib import Path

import cv2

from .config import GestureConfig, load_config
from .controller import Controller
from .enums import Gest, HLabel
from .hand_landmarker_manager import HandLandmarkerManager
from .hand_recog import HandRecog
from .trackbar_ui import TrackbarUI


_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _draw_hand(frame_bgr, hand_result) -> None:
    if hand_result is None:
        return
    h, w = frame_bgr.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_result.landmark]
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame_bgr, p, 3, (0, 0, 255), -1)


def _hud_line(frame_bgr, text: str, y: int) -> None:
    cv2.putText(frame_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


class GestureController:
    """Camera loop + MediaPipe Tasks hand landmarks + gesture-to-action mapping."""

    gc_mode = 0
    cap = None
    CAM_HEIGHT = None
    CAM_WIDTH = None
    dom_hand = True

    def __init__(self):
        GestureController.gc_mode = 1
        GestureController.cap = cv2.VideoCapture(0)
        GestureController.CAM_HEIGHT = GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        GestureController.CAM_WIDTH = GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        model_path = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
        self._landmarker = HandLandmarkerManager(model_path=model_path, num_hands=2)

        self._config_path = Path(__file__).resolve().parents[2] / "gesture_config.json"
        self._config = load_config(self._config_path)

    def start(self) -> None:
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)
        cfg = self._config
        trackbar = TrackbarUI(cfg, self._config_path)

        try:
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                success, frame = GestureController.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                frame = cv2.flip(frame, 1)
                ts_ms = int(time.monotonic() * 1000)
                detection = self._landmarker.detect_bgr(frame, timestamp_ms=ts_ms)

                trackbar.sync(cfg)

                if GestureController.dom_hand is True:
                    major_res = detection.right
                    minor_res = detection.left
                else:
                    major_res = detection.left
                    minor_res = detection.right

                handmajor.update_hand_result(major_res)
                handminor.update_hand_result(minor_res)
                handmajor.set_finger_state()
                handminor.set_finger_state()

                gest_name = None
                if major_res is not None or minor_res is not None:
                    gest_name = handminor.get_gesture(cfg)
                    if gest_name == cfg.resolve_gesture("scroll"):
                        Controller.handle_controls(gest_name, handminor.hand_result, cfg)
                    else:
                        gest_name = handmajor.get_gesture(cfg)
                        Controller.handle_controls(gest_name, handmajor.hand_result, cfg)
                else:
                    Controller.prev_hand = None

                _draw_hand(frame, major_res)
                _draw_hand(frame, minor_res)

                _hud_line(frame, f"hands: major={'Y' if major_res else 'N'}  minor={'Y' if minor_res else 'N'}", 20)
                _hud_line(frame, f"gesture_id: {gest_name if gest_name is not None else '-'}  v_flag: {int(Controller.flag)}", 45)
                _hud_line(frame, "Adjust settings in the 'Gesture Settings' trackbar window", 70)
                _hud_line(frame, "Exit: press Enter", 95)
                cv2.imshow("Gesture Controller", frame)

                if cv2.waitKey(5) & 0xFF == 13:
                    break
        finally:
            trackbar.destroy()
            self._landmarker.close()
            GestureController.cap.release()
            cv2.destroyAllWindows()

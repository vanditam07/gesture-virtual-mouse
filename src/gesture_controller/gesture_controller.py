from __future__ import annotations

import time
from pathlib import Path

import cv2

from .controller import Controller
from .enums import Gest, HLabel
from .hand_landmarker_manager import HandLandmarkerManager
from .hand_recog import HandRecog


_HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def _draw_hand(frame_bgr, hand_result) -> None:
    if hand_result is None:
        return

    h, w = frame_bgr.shape[:2]
    pts = []
    for lm in hand_result.landmark:
        pts.append((int(lm.x * w), int(lm.y * h)))

    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame_bgr, p, 3, (0, 0, 255), -1)


class GestureController:
    """
    Camera loop + MediaPipe Tasks hand landmarks + gesture-to-action mapping.
    """

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

    def start(self) -> None:
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)

        try:
            while GestureController.cap.isOpened() and GestureController.gc_mode:
                success, frame = GestureController.cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                frame = cv2.flip(frame, 1)
                ts_ms = int(time.monotonic() * 1000)
                detection = self._landmarker.detect_bgr(frame, timestamp_ms=ts_ms)

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

                if major_res is not None or minor_res is not None:
                    gest_name = handminor.get_gesture()
                    if gest_name == Gest.PINCH_MINOR:
                        Controller.handle_controls(gest_name, handminor.hand_result)
                    else:
                        gest_name = handmajor.get_gesture()
                        Controller.handle_controls(gest_name, handmajor.hand_result)
                else:
                    Controller.prev_hand = None

                _draw_hand(frame, major_res)
                _draw_hand(frame, minor_res)
                cv2.imshow("Gesture Controller", frame)

                if cv2.waitKey(5) & 0xFF == 13:
                    break
        finally:
            self._landmarker.close()
            GestureController.cap.release()
            cv2.destroyAllWindows()


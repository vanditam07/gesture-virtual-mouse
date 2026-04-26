from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

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

_KEY_ENTER = 13
_KEY_P = ord("p")


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


def _try_load_proto_classifier(
    encoder_path: Path, prototypes_path: Path
) -> Optional[object]:
    """Attempt to load the ProtoClassifier; return None if files are missing."""
    if not encoder_path.exists():
        return None
    try:
        from proto_net.classifier import ProtoClassifier
        proto_path = prototypes_path if prototypes_path.exists() else None
        return ProtoClassifier(encoder_path, proto_path)
    except Exception as exc:
        print(f"[proto] Could not load classifier: {exc}")
        return None


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

        model_dir = Path(__file__).resolve().parents[1] / "models"
        self._landmarker = HandLandmarkerManager(
            model_path=model_dir / "hand_landmarker.task", num_hands=2
        )

        self._config_path = Path(__file__).resolve().parents[2] / "gesture_config.json"
        self._config = load_config(self._config_path)

        self._encoder_path = model_dir / "pretrained_encoder.pth"
        self._prototypes_path = Path(__file__).resolve().parents[2] / "user_prototypes.npy"
        self._proto_clf = _try_load_proto_classifier(self._encoder_path, self._prototypes_path)

    def _launch_wizard(self) -> None:
        """Run the Personalisation Wizard, then reload prototypes."""
        try:
            from proto_net.wizard import PersonalisationWizard
            wiz = PersonalisationWizard(
                encoder_path=self._encoder_path,
                save_path=self._prototypes_path,
                landmarker_manager=self._landmarker,
            )
            success = wiz.run(GestureController.cap)
            if success:
                if self._proto_clf is not None:
                    self._proto_clf.load_prototypes(self._prototypes_path)
                else:
                    self._proto_clf = _try_load_proto_classifier(
                        self._encoder_path, self._prototypes_path
                    )
                print("[proto] Prototypes reloaded after enrolment.")
        except Exception as exc:
            print(f"[proto] Wizard error: {exc}")

    def start(self) -> None:
        handmajor = HandRecog(HLabel.MAJOR)
        handminor = HandRecog(HLabel.MINOR)
        cfg = self._config
        trackbar = TrackbarUI(cfg, self._config_path)
        proto_used_count = 0
        rule_fallback_count = 0

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
                source = "rule"

                if major_res is not None or minor_res is not None:
                    proto_used = False
                    active_hand = major_res or minor_res

                    if self._proto_clf is not None and self._proto_clf.is_ready and active_hand is not None:
                        from proto_net.feature_extraction import extract_feature_vector
                        fv = extract_feature_vector(active_hand.landmark)
                        proto_gest, conf, src = self._proto_clf.predict(fv)
                        if proto_gest is not None:
                            gest_name = proto_gest
                            source = f"proto ({conf:.0%})"
                            Controller.handle_controls(gest_name, active_hand, cfg)
                            proto_used = True
                            proto_used_count += 1

                    if not proto_used:
                        gest_name = handminor.get_gesture(cfg)
                        if gest_name == cfg.resolve_gesture("scroll"):
                            Controller.handle_controls(gest_name, handminor.hand_result, cfg)
                        else:
                            gest_name = handmajor.get_gesture(cfg)
                            Controller.handle_controls(gest_name, handmajor.hand_result, cfg)
                        rule_fallback_count += 1
                else:
                    Controller.prev_hand = None

                _draw_hand(frame, major_res)
                _draw_hand(frame, minor_res)

                proto_status = "ACTIVE" if (self._proto_clf and self._proto_clf.is_ready) else "off"
                total_classified = proto_used_count + rule_fallback_count
                proto_ratio = (
                    int((proto_used_count * 100) / total_classified)
                    if total_classified > 0
                    else 0
                )
                _hud_line(frame, f"hands: major={'Y' if major_res else 'N'}  minor={'Y' if minor_res else 'N'}", 20)
                _hud_line(frame, f"gesture: {gest_name if gest_name is not None else '-'}  [{source}]  v_flag: {int(Controller.flag)}", 45)
                _hud_line(
                    frame,
                    (
                        f"proto: {proto_status}  used={proto_used_count}  "
                        f"fallback={rule_fallback_count}  ({proto_ratio}% proto)"
                    ),
                    70,
                )
                _hud_line(frame, "P = personalise  |  Enter = exit", 95)
                cv2.imshow("Gesture Controller", frame)

                key = cv2.waitKey(5) & 0xFF
                if key == _KEY_ENTER:
                    break
                if key == _KEY_P:
                    self._launch_wizard()
        finally:
            trackbar.destroy()
            self._landmarker.close()
            GestureController.cap.release()
            cv2.destroyAllWindows()

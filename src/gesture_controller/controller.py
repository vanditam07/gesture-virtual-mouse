from __future__ import annotations

from ctypes import POINTER, cast
from typing import TYPE_CHECKING

import pyautogui
import screen_brightness_control as sbcontrol
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from .enums import Gest

if TYPE_CHECKING:
    from .config import GestureConfig

pyautogui.FAILSAFE = False


class Controller:
    """Executes OS actions according to detected gestures, driven by GestureConfig."""

    tx_old = 0
    ty_old = 0
    trial = True
    flag = False
    grabflag = False
    pinchmajorflag = False
    pinchminorflag = False
    pinchstartxcoord = None
    pinchstartycoord = None
    pinchdirectionflag = None
    prevpinchlv = 0
    pinchlv = 0
    framecount = 0
    prev_hand = None

    @staticmethod
    def getpinchylv(hand_result) -> float:
        return round((Controller.pinchstartycoord - hand_result.landmark[8].y) * 10, 1)

    @staticmethod
    def getpinchxlv(hand_result) -> float:
        return round((hand_result.landmark[8].x - Controller.pinchstartxcoord) * 10, 1)

    @staticmethod
    def changesystembrightness() -> None:
        current = sbcontrol.get_brightness(display=0) / 100.0
        current += Controller.pinchlv / 50.0
        current = min(1.0, max(0.0, current))
        sbcontrol.fade_brightness(int(100 * current), start=sbcontrol.get_brightness(display=0))

    @staticmethod
    def changesystemvolume() -> None:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        current = volume.GetMasterVolumeLevelScalar()
        current += Controller.pinchlv / 50.0
        current = min(1.0, max(0.0, current))
        volume.SetMasterVolumeLevelScalar(current, None)

    @staticmethod
    def scrollVertical() -> None:
        pyautogui.scroll(120 if Controller.pinchlv > 0.0 else -120)

    @staticmethod
    def scrollHorizontal() -> None:
        pyautogui.keyDown("shift")
        pyautogui.keyDown("ctrl")
        pyautogui.scroll(-120 if Controller.pinchlv > 0.0 else 120)
        pyautogui.keyUp("ctrl")
        pyautogui.keyUp("shift")

    @staticmethod
    def get_position(hand_result, cfg: GestureConfig):
        point = 9
        position = [hand_result.landmark[point].x, hand_result.landmark[point].y]
        sx, sy = pyautogui.size()
        x_old, y_old = pyautogui.position()
        x = int(position[0] * sx)
        y = int(position[1] * sy)

        if Controller.prev_hand is None:
            Controller.prev_hand = x, y

        delta_x = x - Controller.prev_hand[0]
        delta_y = y - Controller.prev_hand[1]
        distsq = delta_x ** 2 + delta_y ** 2
        Controller.prev_hand = [x, y]

        if distsq <= cfg.dead_zone:
            ratio = 0
        elif distsq <= cfg.mid_range:
            ratio = 0.07 * cfg.cursor_speed * (distsq ** 0.5)
        else:
            ratio = 2.1 * cfg.cursor_speed

        x, y = x_old + delta_x * ratio, y_old + delta_y * ratio
        return (x, y)

    @staticmethod
    def pinch_control_init(hand_result) -> None:
        Controller.pinchstartxcoord = hand_result.landmark[8].x
        Controller.pinchstartycoord = hand_result.landmark[8].y
        Controller.pinchlv = 0
        Controller.prevpinchlv = 0
        Controller.framecount = 0

    @staticmethod
    def pinch_control(hand_result, controlHorizontal, controlVertical, cfg: GestureConfig) -> None:
        if Controller.framecount == 5:
            Controller.framecount = 0
            Controller.pinchlv = Controller.prevpinchlv

            if Controller.pinchdirectionflag is True:
                controlHorizontal()
            elif Controller.pinchdirectionflag is False:
                controlVertical()

        lvx = Controller.getpinchxlv(hand_result)
        lvy = Controller.getpinchylv(hand_result)
        thresh = cfg.pinch_threshold

        if abs(lvy) > abs(lvx) and abs(lvy) > thresh:
            Controller.pinchdirectionflag = False
            if abs(Controller.prevpinchlv - lvy) < thresh:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount = 0
        elif abs(lvx) > thresh:
            Controller.pinchdirectionflag = True
            if abs(Controller.prevpinchlv - lvx) < thresh:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount = 0

    @staticmethod
    def handle_controls(gesture: int, hand_result, cfg: GestureConfig) -> None:
        rev = cfg.build_reverse_map()
        action = rev.get(gesture)

        x, y = None, None
        if gesture != int(Gest.PALM) and hand_result is not None:
            x, y = Controller.get_position(hand_result, cfg)

        # Release grab if gesture is no longer drag
        if action != "drag" and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button="left")

        if action != "volume_brightness" and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False

        if action != "scroll" and Controller.pinchminorflag:
            Controller.pinchminorflag = False

        if action is None or not cfg.is_enabled(action):
            return

        if action == "move_cursor":
            Controller.flag = True
            pyautogui.moveTo(x, y, duration=cfg.move_duration)

        elif action == "drag":
            if not Controller.grabflag:
                Controller.grabflag = True
                pyautogui.mouseDown(button="left")
            pyautogui.moveTo(x, y, duration=cfg.move_duration)

        elif action == "left_click" and Controller.flag:
            pyautogui.click()
            Controller.flag = False

        elif action == "right_click" and Controller.flag:
            pyautogui.click(button="right")
            Controller.flag = False

        elif action == "double_click" and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False

        elif action == "scroll":
            if Controller.pinchminorflag is False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
            Controller.pinch_control(hand_result, Controller.scrollHorizontal, Controller.scrollVertical, cfg)

        elif action == "volume_brightness":
            if Controller.pinchmajorflag is False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
            Controller.pinch_control(hand_result, Controller.changesystembrightness, Controller.changesystemvolume, cfg)

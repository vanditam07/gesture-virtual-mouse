from __future__ import annotations

from ctypes import POINTER, cast

import pyautogui
import screen_brightness_control as sbcontrol
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from .enums import Gest

pyautogui.FAILSAFE = False


class Controller:
    """
    Executes OS actions according to detected gestures.
    """

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
    pinch_threshold = 0.3

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
    def get_position(hand_result):
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
        distsq = delta_x**2 + delta_y**2
        Controller.prev_hand = [x, y]

        if distsq <= 25:
            ratio = 0
        elif distsq <= 900:
            ratio = 0.07 * (distsq ** (1 / 2))
        else:
            ratio = 2.1

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
    def pinch_control(hand_result, controlHorizontal, controlVertical) -> None:
        if Controller.framecount == 5:
            Controller.framecount = 0
            Controller.pinchlv = Controller.prevpinchlv

            if Controller.pinchdirectionflag is True:
                controlHorizontal()
            elif Controller.pinchdirectionflag is False:
                controlVertical()

        lvx = Controller.getpinchxlv(hand_result)
        lvy = Controller.getpinchylv(hand_result)

        if abs(lvy) > abs(lvx) and abs(lvy) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = False
            if abs(Controller.prevpinchlv - lvy) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvy
                Controller.framecount = 0
        elif abs(lvx) > Controller.pinch_threshold:
            Controller.pinchdirectionflag = True
            if abs(Controller.prevpinchlv - lvx) < Controller.pinch_threshold:
                Controller.framecount += 1
            else:
                Controller.prevpinchlv = lvx
                Controller.framecount = 0

    @staticmethod
    def handle_controls(gesture: Gest, hand_result) -> None:
        x, y = None, None
        if gesture != Gest.PALM:
            x, y = Controller.get_position(hand_result)

        if gesture != Gest.FIST and Controller.grabflag:
            Controller.grabflag = False
            pyautogui.mouseUp(button="left")

        if gesture != Gest.PINCH_MAJOR and Controller.pinchmajorflag:
            Controller.pinchmajorflag = False

        if gesture != Gest.PINCH_MINOR and Controller.pinchminorflag:
            Controller.pinchminorflag = False

        if gesture == Gest.V_GEST:
            Controller.flag = True
            pyautogui.moveTo(x, y, duration=0.1)
        elif gesture == Gest.FIST:
            if not Controller.grabflag:
                Controller.grabflag = True
                pyautogui.mouseDown(button="left")
            pyautogui.moveTo(x, y, duration=0.1)
        elif gesture == Gest.MID and Controller.flag:
            pyautogui.click()
            Controller.flag = False
        elif gesture == Gest.INDEX and Controller.flag:
            pyautogui.click(button="right")
            Controller.flag = False
        elif gesture == Gest.TWO_FINGER_CLOSED and Controller.flag:
            pyautogui.doubleClick()
            Controller.flag = False
        elif gesture == Gest.PINCH_MINOR:
            if Controller.pinchminorflag is False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchminorflag = True
            Controller.pinch_control(hand_result, Controller.scrollHorizontal, Controller.scrollVertical)
        elif gesture == Gest.PINCH_MAJOR:
            if Controller.pinchmajorflag is False:
                Controller.pinch_control_init(hand_result)
                Controller.pinchmajorflag = True
            Controller.pinch_control(hand_result, Controller.changesystembrightness, Controller.changesystemvolume)


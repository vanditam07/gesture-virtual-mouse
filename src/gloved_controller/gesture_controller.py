import cv2
import pyautogui
import time

from .geometry import draw_box
from .glove import Glove
from .marker import Marker
from .mouse import Mouse
from .roi import ROI
from .tracker import Tracker


class GestureController:
    gc_mode = 0
    pyautogui.FAILSAFE = False
    f_start_time = 0
    f_now_time = 0

    cam_width = 0
    cam_height = 0

    aru_marker = Marker()
    hand_roi = ROI(2.5, 2.5, 6, 0.45, 0.6, 0.4)
    glove = Glove()
    csrt_track = Tracker()
    mouse = Mouse()

    def __init__(self):
        GestureController.cap = cv2.VideoCapture(0)
        if GestureController.cap.isOpened():
            GestureController.cam_width = int(GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            GestureController.cam_height = int(GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            print("CANNOT OPEN CAMERA")

        GestureController.hand_roi.set_camera_dimensions(GestureController.cam_width, GestureController.cam_height)

        GestureController.gc_mode = 1
        GestureController.f_start_time = time.time()
        GestureController.f_now_time = time.time()

    def start(self):
        while True:
            if not GestureController.gc_mode:
                print("Exiting Gesture Controller")
                break

            fps = 30.0
            GestureController.f_start_time = time.time()
            while GestureController.f_now_time - GestureController.f_start_time <= 1.0 / fps:
                GestureController.f_now_time = time.time()

            ret, frame = GestureController.cap.read()
            frame = cv2.flip(frame, 1)

            GestureController.aru_marker.detect(frame)
            if GestureController.aru_marker.is_detected():
                GestureController.csrt_track.corners_to_tracker(GestureController.aru_marker.corners)
                GestureController.csrt_track.CSRT_tracker(frame)
            else:
                GestureController.csrt_track.tracker_bbox = None
                GestureController.csrt_track.CSRT_tracker(frame)
                GestureController.aru_marker.corners = GestureController.csrt_track.tracker_to_corner(GestureController.aru_marker.corners)

            if GestureController.aru_marker.is_detected():
                GestureController.hand_roi.findROI(frame, GestureController.aru_marker)
                GestureController.hand_roi.find_glove_hsv(frame, GestureController.aru_marker)
                FinalMask = GestureController.hand_roi.cropROI(frame)
                GestureController.glove.find_fingers(FinalMask)
                GestureController.glove.find_gesture(frame)
                GestureController.mouse.move_mouse(frame, GestureController.hand_roi.marker_top, GestureController.glove.gesture)

            if GestureController.aru_marker.is_detected():
                GestureController.aru_marker.draw_marker(frame)
                draw_box(frame, GestureController.hand_roi.roi_corners, (255, 0, 0))
                draw_box(frame, GestureController.hand_roi.hsv_corners, (0, 0, 250))
                cv2.imshow("FinalMask", FinalMask)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        GestureController.cap.release()
        cv2.destroyAllWindows()


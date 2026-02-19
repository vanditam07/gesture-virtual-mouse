import numpy as np
import cv2

from .geometry import ecu_dis, find_HSV


class ROI:
    def __init__(
        self,
        roi_alpha1=1.5,
        roi_alpha2=1.5,
        roi_beta=2.5,
        hsv_alpha=0.3,
        hsv_beta=0.5,
        hsv_lift_up=0.3,
    ):
        self.roi_alpha1 = roi_alpha1
        self.roi_alpha2 = roi_alpha2
        self.roi_beta = roi_beta
        self.roi_corners = None

        self.hsv_alpha = hsv_alpha
        self.hsv_beta = hsv_beta
        self.hsv_lift_up = hsv_lift_up
        self.hsv_corners = None

        self.marker_top = None
        self.hsv_glove = None

        self._cam_width = None
        self._cam_height = None

    def set_camera_dimensions(self, cam_width: int, cam_height: int) -> None:
        self._cam_width = cam_width
        self._cam_height = cam_height

    def _in_cam(self, val: int, axis: str) -> int:
        if self._cam_width is None or self._cam_height is None:
            return val
        if axis == "x":
            return max(0, min(int(val), int(self._cam_width)))
        if axis == "y":
            return max(0, min(int(val), int(self._cam_height)))
        return int(val)

    def findROI(self, frame, marker) -> None:
        rec_coor = marker.corners[0][0]
        c1 = (int(rec_coor[0][0]), int(rec_coor[0][1]))
        c2 = (int(rec_coor[1][0]), int(rec_coor[1][1]))
        c3 = (int(rec_coor[2][0]), int(rec_coor[2][1]))
        c4 = (int(rec_coor[3][0]), int(rec_coor[3][1]))

        try:
            marker.marker_x2y = np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) / np.sqrt(
                (c3[0] - c2[0]) ** 2 + (c3[1] - c2[1]) ** 2
            )
        except Exception:
            marker.marker_x2y = 999.0

        cx = (c1[0] + c2[0]) / 2
        cy = (c1[1] + c2[1]) / 2
        self.marker_top = [cx, cy]

        l = np.absolute(ecu_dis(c1, c4))

        try:
            slope_12 = (c1[1] - c2[1]) / (c1[0] - c2[0])
        except Exception:
            slope_12 = (c1[1] - c2[1]) * 999.0 + 0.1

        try:
            slope_14 = -1 / slope_12
        except Exception:
            slope_14 = -999.0

        sign = 1 if slope_14 < 0 else -1

        bot_rx = int(cx + self.roi_alpha2 * l * np.sqrt(1 / (1 + slope_12**2)))
        bot_ry = int(cy + self.roi_alpha2 * slope_12 * l * np.sqrt(1 / (1 + slope_12**2)))

        bot_lx = int(cx - self.roi_alpha1 * l * np.sqrt(1 / (1 + slope_12**2)))
        bot_ly = int(cy - self.roi_alpha1 * slope_12 * l * np.sqrt(1 / (1 + slope_12**2)))

        top_lx = int(bot_lx + sign * self.roi_beta * l * np.sqrt(1 / (1 + slope_14**2)))
        top_ly = int(bot_ly + sign * self.roi_beta * slope_14 * l * np.sqrt(1 / (1 + slope_14**2)))

        top_rx = int(bot_rx + sign * self.roi_beta * l * np.sqrt(1 / (1 + slope_14**2)))
        top_ry = int(bot_ry + sign * self.roi_beta * slope_14 * l * np.sqrt(1 / (1 + slope_14**2)))

        bot_lx = self._in_cam(bot_lx, "x")
        bot_ly = self._in_cam(bot_ly, "y")
        bot_rx = self._in_cam(bot_rx, "x")
        bot_ry = self._in_cam(bot_ry, "y")
        top_lx = self._in_cam(top_lx, "x")
        top_ly = self._in_cam(top_ly, "y")
        top_rx = self._in_cam(top_rx, "x")
        top_ry = self._in_cam(top_ry, "y")

        self.roi_corners = [(bot_lx, bot_ly), (bot_rx, bot_ry), (top_rx, top_ry), (top_lx, top_ly)]

    def find_glove_hsv(self, frame, marker) -> None:
        rec_coor = marker.corners[0][0]
        c1 = (int(rec_coor[0][0]), int(rec_coor[0][1]))
        c2 = (int(rec_coor[1][0]), int(rec_coor[1][1]))
        c3 = (int(rec_coor[2][0]), int(rec_coor[2][1]))
        c4 = (int(rec_coor[3][0]), int(rec_coor[3][1]))

        l = np.absolute(ecu_dis(c1, c4))

        try:
            slope_12 = (c1[1] - c2[1]) / (c1[0] - c2[0])
        except Exception:
            slope_12 = (c1[1] - c2[1]) * 999.0 + 0.1
        try:
            slope_14 = -1 / slope_12
        except Exception:
            slope_14 = -999.0

        sign = 1 if slope_14 < 0 else -1

        bot_rx = int(self.marker_top[0] + self.hsv_alpha * l * np.sqrt(1 / (1 + slope_12**2)))
        bot_ry = int(
            self.marker_top[1]
            - self.hsv_lift_up * l
            + self.hsv_alpha * slope_12 * l * np.sqrt(1 / (1 + slope_12**2))
        )

        bot_lx = int(self.marker_top[0] - self.hsv_alpha * l * np.sqrt(1 / (1 + slope_12**2)))
        bot_ly = int(
            self.marker_top[1]
            - self.hsv_lift_up * l
            - self.hsv_alpha * slope_12 * l * np.sqrt(1 / (1 + slope_12**2))
        )

        top_lx = int(bot_lx + sign * self.hsv_beta * l * np.sqrt(1 / (1 + slope_14**2)))
        top_ly = int(bot_ly + sign * self.hsv_beta * slope_14 * l * np.sqrt(1 / (1 + slope_14**2)))

        top_rx = int(bot_rx + sign * self.hsv_beta * l * np.sqrt(1 / (1 + slope_14**2)))
        top_ry = int(bot_ry + sign * self.hsv_beta * slope_14 * l * np.sqrt(1 / (1 + slope_14**2)))

        region = frame[top_ry:bot_ry, top_lx:bot_rx]
        b, g, r = np.mean(region, axis=(0, 1))

        self.hsv_glove = find_HSV([[r, g, b]])
        self.hsv_corners = [(bot_lx, bot_ly), (bot_rx, bot_ry), (top_rx, top_ry), (top_lx, top_ly)]

    def cropROI(self, frame):
        pts = np.array(self.roi_corners)

        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        croped = frame[y : y + h, x : x + w].copy()

        pts = pts - pts.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        dst = cv2.bitwise_and(croped, croped, mask=mask)

        bg = np.ones_like(croped, np.uint8) * 255
        cv2.bitwise_not(bg, bg, mask=mask)

        kernelOpen = np.ones((3, 3), np.uint8)
        kernelClose = np.ones((5, 5), np.uint8)

        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)

        lower_range = np.array([self.hsv_glove[0][0][0] // 1 - 5, 50, 50])
        upper_range = np.array([self.hsv_glove[0][0][0] // 1 + 5, 255, 255])

        mask = cv2.inRange(hsv, lower_range, upper_range)
        Opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        Closing = cv2.morphologyEx(Opening, cv2.MORPH_CLOSE, kernelClose)
        return Closing


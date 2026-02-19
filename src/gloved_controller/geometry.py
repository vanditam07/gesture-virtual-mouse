import numpy as np
import cv2


def ecu_dis(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def find_HSV(samples):
    try:
        color = np.uint8([samples])
    except Exception:
        color = np.uint8([[[105, 105, 50]]])
    return cv2.cvtColor(color, cv2.COLOR_RGB2HSV)


def draw_box(frame, points, color=(0, 255, 127)):
    if not points:
        return
    frame = cv2.line(frame, points[0], points[1], color, thickness=2, lineType=8)  # top
    frame = cv2.line(frame, points[1], points[2], color, thickness=2, lineType=8)  # right
    frame = cv2.line(frame, points[2], points[3], color, thickness=2, lineType=8)  # bottom
    frame = cv2.line(frame, points[3], points[0], color, thickness=2, lineType=8)  # left


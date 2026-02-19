import cv2
import math


class Glove:
    def __init__(self):
        self.fingers = 0
        self.arearatio = 0
        self.gesture = 0

    def find_fingers(self, FinalMask):
        conts, h = cv2.findContours(FinalMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        hull = [cv2.convexHull(c) for c in conts]

        try:
            cnt = max(conts, key=lambda x: cv2.contourArea(x))
            epsilon = 0.0005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            hull = cv2.convexHull(cnt)
            areahull = cv2.contourArea(hull)
            areacnt = cv2.contourArea(cnt)
            self.arearatio = ((areahull - areacnt) / areacnt) * 100
            hull = cv2.convexHull(approx, returnPoints=False)
            defects = cv2.convexityDefects(approx, hull)
        except Exception:
            print("No Contours found in FinalMask")
            defects = None
            approx = None

        l = 0
        try:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                d = (2 * ar) / a
                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

                if angle <= 90 and d > 30:
                    l += 1

                cv2.line(FinalMask, start, end, [255, 255, 255], 2)

            l += 1
        except Exception:
            l = 0
            print("No Defects found in mask")

        self.fingers = l

    def find_gesture(self, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.gesture = 0
        if self.fingers == 1:
            if self.arearatio < 15:
                cv2.putText(frame, "0", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                self.gesture = 0
            elif self.arearatio < 25:
                cv2.putText(frame, "2 fingers", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                self.gesture = 2
            else:
                cv2.putText(frame, "1 finger", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                self.gesture = 1

        elif self.fingers == 2:
            cv2.putText(frame, "2", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            self.gesture = 3


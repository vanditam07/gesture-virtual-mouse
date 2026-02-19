import cv2
import numpy as np
import time


class Tracker:
    def __init__(self):
        self.tracker_started = False
        self.tracker = None
        self.start_time = 0.0
        self.now_time = 0.0
        self.tracker_bbox = None

    def corners_to_tracker(self, corners):
        csrt_minX = int(min([corners[0][0][0][0], corners[0][0][1][0], corners[0][0][2][0], corners[0][0][3][0]]))
        csrt_maxX = int(max([corners[0][0][0][0], corners[0][0][1][0], corners[0][0][2][0], corners[0][0][3][0]]))
        csrt_minY = int(min([corners[0][0][0][1], corners[0][0][1][1], corners[0][0][2][1], corners[0][0][3][1]]))
        csrt_maxY = int(max([corners[0][0][0][1], corners[0][0][1][1], corners[0][0][2][1], corners[0][0][3][1]]))
        self.tracker_bbox = [csrt_minX, csrt_minY, csrt_maxX - csrt_minX, csrt_maxY - csrt_minY]

    def tracker_to_corner(self, final_bbox):
        if self.tracker_bbox is None:
            return None
        final_bbox = [[[1, 2], [3, 4], [5, 6], [7, 8]]]
        final_bbox[0][0] = [self.tracker_bbox[0], self.tracker_bbox[1]]
        final_bbox[0][1] = [self.tracker_bbox[0] + self.tracker_bbox[2], self.tracker_bbox[1]]
        final_bbox[0][2] = [self.tracker_bbox[0] + self.tracker_bbox[2], self.tracker_bbox[1] + self.tracker_bbox[3]]
        final_bbox[0][3] = [self.tracker_bbox[0], self.tracker_bbox[1] + self.tracker_bbox[3]]
        return [np.array(final_bbox, dtype="f")]

    def CSRT_tracker(self, frame):
        if self.tracker_bbox is None and self.tracker_started is False:
            return

        if self.tracker_started is False:
            if self.tracker is None:
                self.tracker = cv2.TrackerCSRT_create()

        if self.tracker_bbox is not None:
            try:
                self.start_time = time.time()
                self.tracker.init(frame, self.tracker_bbox)
                self.tracker_started = True
            except Exception:
                print("tracker.init failed")

        try:
            ok, self.tracker_bbox = self.tracker.update(frame)
        except Exception:
            ok = None
            print("tracker.update failed")

        self.now_time = time.time()

        if self.now_time - self.start_time >= 2.0:
            cv2.putText(
                frame,
                "Posture your hand correctly",
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            self.tracker_started = False
            self.tracker_bbox = None
            return

        if ok:
            p1 = (int(self.tracker_bbox[0]), int(self.tracker_bbox[1]))
            p2 = (int(self.tracker_bbox[0] + self.tracker_bbox[2]), int(self.tracker_bbox[1] + self.tracker_bbox[3]))
            cv2.rectangle(frame, p1, p2, (80, 255, 255), 2, 1)
        else:
            self.tracker_started = False
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            print("Tracking failure detected")


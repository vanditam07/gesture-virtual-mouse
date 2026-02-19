import cv2
import cv2.aruco as aruco
import glob
import numpy as np
import os


class Marker:
    def __init__(self, dict_type=aruco.DICT_4X4_50, thresh_constant=1):
        # OpenCV ArUco API differs across versions; support both.
        try:
            self.aruco_dict = aruco.getPredefinedDictionary(dict_type)
        except AttributeError:
            self.aruco_dict = aruco.Dictionary_get(dict_type)

        try:
            self.parameters = aruco.DetectorParameters()
        except AttributeError:
            self.parameters = aruco.DetectorParameters_create()
        self.parameters.adaptiveThreshConstant = thresh_constant
        self.corners = None
        self.marker_x2y = 1
        self.mtx, self.dist = Marker.calibrate()
        self._detector = None
        try:
            self._detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        except AttributeError:
            self._detector = None

    @staticmethod
    def calibrate():
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
        objpoints = []
        imgpoints = []

        # Keep compatibility with original layout: `src/calib_images/...`
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        p1 = str(os.path.join(src_dir, "calib_images", "checkerboard", "*.jpg"))
        images = glob.glob(p1)
        gray = None

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)
            if ret is True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                cv2.drawChessboardCorners(img, (7, 6), corners2, ret)

        if gray is None:
            # Fallback: allow runtime to proceed; downstream pose estimation may be off.
            return None, None

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist

    def detect(self, frame) -> None:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            self.corners, ids, rejected = self._detector.detectMarkers(gray_frame)
        else:
            self.corners, ids, rejected = aruco.detectMarkers(gray_frame, self.aruco_dict, parameters=self.parameters)

        if np.all(ids is not None) and self.mtx is not None and self.dist is not None:
            aruco.estimatePoseSingleMarkers(self.corners, 0.05, self.mtx, self.dist)
        else:
            self.corners = None

    def is_detected(self) -> bool:
        return bool(self.corners)

    def draw_marker(self, frame) -> None:
        aruco.drawDetectedMarkers(frame, self.corners)


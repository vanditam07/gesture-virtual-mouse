"""Extract a 77-dimensional feature vector from 21 MediaPipe hand landmarks.

Layout:
    [0:63]  Raw landmark coordinates (21 landmarks x 3: x, y, z), normalised
    [63:68] Fingertip-to-wrist Euclidean distances
    [68:73] Inter-finger angles at MCP joints (radians)
    [73:77] Finger curl ratios (index, middle, ring, pinky)
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

FEATURE_DIM = 77

_TIPS = [4, 8, 12, 16, 20]
_MCPS = [2, 5, 9, 13, 17]
_WRIST = 0

_FINGER_PAIRS = [
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),
    (4, 20),
]

_CURL_FINGERS = [
    (8, 5),    # index:  tip, MCP
    (12, 9),   # middle: tip, MCP
    (16, 13),  # ring:   tip, MCP
    (20, 17),  # pinky:  tip, MCP
]


def _euclidean(ax: float, ay: float, az: float,
               bx: float, by: float, bz: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)


def _angle_between(ux: float, uy: float, uz: float,
                   vx: float, vy: float, vz: float) -> float:
    """Angle in radians between two 3-D vectors."""
    dot = ux * vx + uy * vy + uz * vz
    mag_u = math.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
    mag_v = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    denom = mag_u * mag_v
    if denom < 1e-9:
        return 0.0
    return math.acos(max(-1.0, min(1.0, dot / denom)))


def extract_feature_vector(landmarks) -> np.ndarray:
    """Convert a list of 21 MediaPipe hand landmarks to a 77-dim numpy array.

    Parameters
    ----------
    landmarks : iterable
        21 objects each with ``.x``, ``.y``, ``.z`` attributes (normalised 0-1).

    Returns
    -------
    np.ndarray of shape (77,) and dtype float32.
    """
    lm: List = list(landmarks)
    if len(lm) < 21:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    raw = np.empty(63, dtype=np.float32)
    for i, pt in enumerate(lm[:21]):
        raw[i * 3] = pt.x
        raw[i * 3 + 1] = pt.y
        raw[i * 3 + 2] = pt.z

    wx, wy, wz = lm[_WRIST].x, lm[_WRIST].y, lm[_WRIST].z

    tip_dists = np.empty(5, dtype=np.float32)
    for i, tip in enumerate(_TIPS):
        tip_dists[i] = _euclidean(lm[tip].x, lm[tip].y, lm[tip].z, wx, wy, wz)

    angles = np.empty(5, dtype=np.float32)
    for i, (a, b) in enumerate(_FINGER_PAIRS):
        ux = lm[a].x - wx
        uy = lm[a].y - wy
        uz = lm[a].z - wz
        vx = lm[b].x - wx
        vy = lm[b].y - wy
        vz = lm[b].z - wz
        angles[i] = _angle_between(ux, uy, uz, vx, vy, vz)

    curls = np.empty(4, dtype=np.float32)
    for i, (tip, mcp) in enumerate(_CURL_FINGERS):
        tip_mcp = _euclidean(lm[tip].x, lm[tip].y, lm[tip].z,
                             lm[mcp].x, lm[mcp].y, lm[mcp].z)
        mcp_wrist = _euclidean(lm[mcp].x, lm[mcp].y, lm[mcp].z, wx, wy, wz)
        curls[i] = tip_mcp / mcp_wrist if mcp_wrist > 1e-6 else 0.0

    return np.concatenate([raw, tip_dists, angles, curls])

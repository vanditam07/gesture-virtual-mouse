"""Canonical 21-landmark templates for each gesture class.

Each template is a list of 21 (x, y, z) tuples representing a typical hand
pose.  These are used by the synthetic data generator in meta_train.py
to produce population-level training data via random perturbation.

NOTE: For production accuracy, replace these hand-crafted approximations
with real recorded landmark data from multiple users.
"""

from __future__ import annotations

import numpy as np

NUM_GESTURE_CLASSES = 8

CLASS_NAMES = [
    "PALM",
    "V_GEST",
    "FIST",
    "MID",
    "INDEX",
    "TWO_FINGER_CLOSED",
    "PINCH_MINOR",
    "PINCH_MAJOR",
]


def _open_hand() -> np.ndarray:
    """All fingers extended (palm / idle)."""
    return np.array([
        [0.50, 0.90, 0.00],  # 0  wrist
        [0.60, 0.80, -0.02], # 1  thumb_cmc
        [0.68, 0.70, -0.03], # 2  thumb_mcp
        [0.73, 0.60, -0.03], # 3  thumb_ip
        [0.78, 0.52, -0.03], # 4  thumb_tip
        [0.55, 0.60, -0.01], # 5  index_mcp
        [0.55, 0.45, -0.01], # 6  index_pip
        [0.55, 0.33, -0.01], # 7  index_dip
        [0.55, 0.22, -0.01], # 8  index_tip
        [0.48, 0.58, 0.00],  # 9  middle_mcp
        [0.48, 0.42, 0.00],  # 10 middle_pip
        [0.48, 0.30, 0.00],  # 11 middle_dip
        [0.48, 0.20, 0.00],  # 12 middle_tip
        [0.41, 0.60, 0.01],  # 13 ring_mcp
        [0.41, 0.45, 0.01],  # 14 ring_pip
        [0.41, 0.34, 0.01],  # 15 ring_dip
        [0.41, 0.25, 0.01],  # 16 ring_tip
        [0.34, 0.65, 0.02],  # 17 pinky_mcp
        [0.34, 0.52, 0.02],  # 18 pinky_pip
        [0.34, 0.43, 0.02],  # 19 pinky_dip
        [0.34, 0.35, 0.02],  # 20 pinky_tip
    ], dtype=np.float32)


def _fist() -> np.ndarray:
    """All fingers curled in (closed fist)."""
    base = _open_hand().copy()
    for tip, dip, pip_ in [(8, 7, 6), (12, 11, 10), (16, 15, 14), (20, 19, 18)]:
        mcp = pip_ - 1
        base[pip_] = base[mcp] + [0.0, 0.05, 0.02]
        base[dip]  = base[mcp] + [0.0, 0.08, 0.04]
        base[tip]  = base[mcp] + [0.0, 0.06, 0.05]
    base[3] = base[2] + [0.02, 0.05, 0.02]
    base[4] = base[2] + [0.01, 0.08, 0.03]
    return base


def _v_gesture() -> np.ndarray:
    """Index + middle extended and spread, rest curled."""
    base = _fist().copy()
    open_h = _open_hand()
    for i in [6, 7, 8]:
        base[i] = open_h[i].copy()
    for i in [10, 11, 12]:
        base[i] = open_h[i].copy()
    base[8][0] += 0.04
    base[12][0] -= 0.04
    return base


def _mid_finger() -> np.ndarray:
    """Only middle finger extended."""
    base = _fist().copy()
    open_h = _open_hand()
    for i in [10, 11, 12]:
        base[i] = open_h[i].copy()
    return base


def _index_finger() -> np.ndarray:
    """Only index finger extended."""
    base = _fist().copy()
    open_h = _open_hand()
    for i in [6, 7, 8]:
        base[i] = open_h[i].copy()
    return base


def _two_finger_closed() -> np.ndarray:
    """Index + middle extended, close together (not spread)."""
    base = _fist().copy()
    open_h = _open_hand()
    for i in [6, 7, 8, 10, 11, 12]:
        base[i] = open_h[i].copy()
    return base


def _pinch() -> np.ndarray:
    """Thumb + index pinching, rest curled (base for both major/minor)."""
    base = _fist().copy()
    open_h = _open_hand()
    for i in [6, 7]:
        base[i] = open_h[i].copy()
    base[8] = [0.68, 0.50, -0.02]
    base[3] = [0.66, 0.55, -0.02]
    base[4] = [0.68, 0.50, -0.02]
    return base


def _pinch_minor() -> np.ndarray:
    """Pinch with last 3 fingers extended (LAST3/LAST4 + thumb-index pinch)."""
    base = _open_hand().copy()
    base[8] = [0.68, 0.50, -0.02]
    base[3] = [0.66, 0.55, -0.02]
    base[4] = [0.68, 0.50, -0.02]
    return base


def get_canonical_templates() -> np.ndarray:
    """Return array of shape (NUM_GESTURE_CLASSES, 21, 3)."""
    return np.stack([
        _open_hand(),
        _v_gesture(),
        _fist(),
        _mid_finger(),
        _index_finger(),
        _two_finger_closed(),
        _pinch_minor(),
        _pinch(),
    ])

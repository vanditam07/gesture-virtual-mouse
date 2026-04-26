"""Real-time Prototypical Network classifier with confidence-gated fallback.

Per-frame pipeline:
    1. Receive 77-dim feature vector
    2. Encode to 64-dim embedding (frozen encoder, no_grad)
    3. Squared Euclidean distance to each user prototype
    4. Softmax over negative distances -> class probabilities
    5. Confidence gate: accept if max_prob >= threshold, else return None
    6. Majority-vote smoothing over k frames
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import EMBEDDING_DIM, ProtoEncoder
from .feature_extraction import FEATURE_DIM
from .gesture_templates import CLASS_NAMES, NUM_GESTURE_CLASSES

_GEST_INT_MAP = {
    0: 31,  # PALM
    1: 33,  # V_GEST
    2: 0,   # FIST
    3: 4,   # MID
    4: 8,   # INDEX
    5: 34,  # TWO_FINGER_CLOSED
    6: 36,  # PINCH_MINOR
    7: 35,  # PINCH_MAJOR
}


class ProtoClassifier:
    """Frozen proto-network inference with confidence gate and majority vote."""

    def __init__(
        self,
        encoder_path: Path,
        prototypes_path: Optional[Path] = None,
        confidence_threshold: float = 0.65,
        vote_window: int = 5,
    ) -> None:
        self._encoder = ProtoEncoder.load_checkpoint(encoder_path)
        self._threshold = confidence_threshold
        self._vote_window = vote_window
        self._vote_buffer: deque[int] = deque(maxlen=vote_window)

        self._prototypes: Optional[torch.Tensor] = None
        if prototypes_path is not None and prototypes_path.exists():
            self.load_prototypes(prototypes_path)

    @property
    def is_ready(self) -> bool:
        return self._prototypes is not None

    def load_prototypes(self, path: Path) -> None:
        data = np.load(path)
        self._prototypes = torch.from_numpy(data).float()
        expected = (NUM_GESTURE_CLASSES, EMBEDDING_DIM)
        if self._prototypes.shape != expected:
            raise ValueError(
                f"Prototype shape {self._prototypes.shape} != expected {expected}"
            )
        self._vote_buffer.clear()

    @torch.no_grad()
    def predict(self, feature_vector: np.ndarray) -> Tuple[Optional[int], float, str]:
        """Classify a single 77-dim feature vector.

        Always returns the best-matching prototype class (no confidence gate).
        The majority-vote window smooths out noisy frames.

        Returns
        -------
        (gest_int, confidence, source)
            gest_int : Gest integer ID (from enums), or None if no prototypes
            confidence : max softmax probability
            source : "proto"
        """
        if not self.is_ready:
            return None, 0.0, "no_prototypes"

        x = torch.from_numpy(feature_vector).float().unsqueeze(0)
        query_emb = self._encoder(x)

        dists = torch.cdist(query_emb, self._prototypes.unsqueeze(0)).squeeze(0).pow(2)
        probs = F.softmax(-dists, dim=-1).squeeze(0)
        confidence, cls_idx = probs.max(dim=-1)
        confidence = confidence.item()
        cls_idx = cls_idx.item()

        self._vote_buffer.append(cls_idx)
        voted_cls = max(set(self._vote_buffer), key=lambda c: list(self._vote_buffer).count(c))
        gest_int = _GEST_INT_MAP.get(voted_cls, 31)
        return gest_int, confidence, "proto"

    @torch.no_grad()
    def embed(self, feature_vector: np.ndarray) -> np.ndarray:
        """Return 64-dim embedding for a single feature vector (used by wizard)."""
        x = torch.from_numpy(feature_vector).float().unsqueeze(0)
        return self._encoder(x).squeeze(0).numpy()

    def class_name(self, proto_cls: int) -> str:
        if 0 <= proto_cls < len(CLASS_NAMES):
            return CLASS_NAMES[proto_cls]
        return f"unknown_{proto_cls}"

"""ProtoEncoder: MLP that maps a 77-dim landmark vector to a 64-dim L2-normalised embedding.

Architecture (from the research spec):
    Linear(77, 256) -> BatchNorm1d(256) -> ReLU
    Linear(256, 128) -> BatchNorm1d(128) -> ReLU
    Linear(128, 64)  -> L2 normalise
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extraction import FEATURE_DIM

EMBEDDING_DIM = 64


class ProtoEncoder(nn.Module):
    """Few-shot embedding network for hand gesture features."""

    def __init__(self, input_dim: int = FEATURE_DIM, embed_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embeddings.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        Tensor of shape (batch, embed_dim) on the unit hypersphere.
        """
        raw = self.net(x)
        return F.normalize(raw, p=2, dim=-1)

    def save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load_checkpoint(cls, path: Path, **kwargs) -> "ProtoEncoder":
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        return model

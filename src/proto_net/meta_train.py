"""Episodic meta-training for the ProtoEncoder using Prototypical Networks.

Generates synthetic population-level gesture data from canonical templates
and trains the encoder via N-way K-shot episodes (Snell et al., 2017).

Usage (from project root):
    python -m proto_net.meta_train          # run from src/
    python src/proto_net/meta_train.py      # or directly

NOTE: Synthetic data lets the training loop run end-to-end.  For production
accuracy, replace SyntheticGestureDataset with real multi-user recordings.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import ProtoEncoder
from .feature_extraction import FEATURE_DIM, extract_feature_vector
from .gesture_templates import NUM_GESTURE_CLASSES, get_canonical_templates


class _LandmarkProxy:
    """Lightweight stand-in for a MediaPipe landmark with .x/.y/.z attrs."""
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


class SyntheticGestureDataset:
    """Generates 77-dim feature vectors by perturbing canonical templates."""

    def __init__(
        self,
        samples_per_class: int = 250,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        shift_range: float = 0.1,
        seed: int = 42,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        templates = get_canonical_templates()  # (C, 21, 3)

        self.features: Dict[int, np.ndarray] = {}
        for cls_idx in range(NUM_GESTURE_CLASSES):
            vecs = []
            for _ in range(samples_per_class):
                lm = self._augment(templates[cls_idx], noise_std, scale_range, shift_range)
                proxies = [_LandmarkProxy(lm[j, 0], lm[j, 1], lm[j, 2]) for j in range(21)]
                vecs.append(extract_feature_vector(proxies))
            self.features[cls_idx] = np.stack(vecs)

    def _augment(
        self,
        template: np.ndarray,
        noise_std: float,
        scale_range: Tuple[float, float],
        shift_range: float,
    ) -> np.ndarray:
        lm = template.copy()
        lm += self._rng.normal(0, noise_std, lm.shape).astype(np.float32)
        scale = self._rng.uniform(*scale_range)
        lm *= scale
        shift = self._rng.uniform(-shift_range, shift_range, size=3).astype(np.float32)
        lm += shift
        return np.clip(lm, 0.0, 1.0)


def _sample_episode(
    dataset: SyntheticGestureDataset,
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one N-way K-shot episode.

    Returns support_x, support_y, query_x, query_y as tensors.
    """
    classes = random.sample(range(NUM_GESTURE_CLASSES), n_way)
    support_x, support_y, query_x, query_y = [], [], [], []

    for new_label, cls_idx in enumerate(classes):
        pool = dataset.features[cls_idx]
        indices = np.random.choice(len(pool), k_shot + q_query, replace=False)
        for i in indices[:k_shot]:
            support_x.append(pool[i])
            support_y.append(new_label)
        for i in indices[k_shot:]:
            query_x.append(pool[i])
            query_y.append(new_label)

    return (
        torch.tensor(np.stack(support_x), dtype=torch.float32),
        torch.tensor(support_y, dtype=torch.long),
        torch.tensor(np.stack(query_x), dtype=torch.float32),
        torch.tensor(query_y, dtype=torch.long),
    )


def _compute_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    n_way: int,
) -> torch.Tensor:
    """Mean embedding per class. Returns (n_way, embed_dim)."""
    protos = []
    for c in range(n_way):
        mask = labels == c
        protos.append(embeddings[mask].mean(dim=0))
    return torch.stack(protos)


def train(
    episodes: int = 10_000,
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 10,
    lr: float = 1e-3,
    val_interval: int = 500,
    val_episodes: int = 50,
    save_path: Path | None = None,
) -> ProtoEncoder:
    """Run episodic meta-training and return the trained encoder."""
    if save_path is None:
        save_path = Path(__file__).resolve().parents[1] / "models" / "pretrained_encoder.pth"

    dataset = SyntheticGestureDataset(samples_per_class=250, seed=42)
    val_dataset = SyntheticGestureDataset(samples_per_class=250, seed=999)

    encoder = ProtoEncoder()
    optimiser = torch.optim.Adam(encoder.parameters(), lr=lr)

    best_val_loss = float("inf")

    for ep in range(1, episodes + 1):
        encoder.train()
        sx, sy, qx, qy = _sample_episode(dataset, n_way, k_shot, q_query)
        s_emb = encoder(sx)
        q_emb = encoder(qx)

        protos = _compute_prototypes(s_emb, sy, n_way)
        dists = torch.cdist(q_emb, protos).pow(2)
        log_probs = F.log_softmax(-dists, dim=-1)
        loss = F.nll_loss(log_probs, qy)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if ep % 100 == 0:
            acc = (log_probs.argmax(dim=-1) == qy).float().mean().item()
            print(f"[episode {ep:>5}]  loss={loss.item():.4f}  acc={acc:.2%}")

        if ep % val_interval == 0:
            val_loss = _validate(encoder, val_dataset, n_way, k_shot, q_query, val_episodes)
            print(f"  >> validation loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                encoder.save_checkpoint(save_path)
                print(f"  >> saved best checkpoint -> {save_path}")

    encoder.save_checkpoint(save_path)
    print(f"Final checkpoint saved -> {save_path}")
    return encoder


@torch.no_grad()
def _validate(
    encoder: ProtoEncoder,
    dataset: SyntheticGestureDataset,
    n_way: int,
    k_shot: int,
    q_query: int,
    num_episodes: int,
) -> float:
    encoder.eval()
    total_loss = 0.0
    for _ in range(num_episodes):
        sx, sy, qx, qy = _sample_episode(dataset, n_way, k_shot, q_query)
        s_emb = encoder(sx)
        q_emb = encoder(qx)
        protos = _compute_prototypes(s_emb, sy, n_way)
        dists = torch.cdist(q_emb, protos).pow(2)
        log_probs = F.log_softmax(-dists, dim=-1)
        total_loss += F.nll_loss(log_probs, qy).item()
    return total_loss / num_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-train ProtoEncoder")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(
        episodes=args.episodes,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

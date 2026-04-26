"""Bootstrap script to prepare src/models/ after a fresh clone.

Downloads the MediaPipe hand-landmarker model and trains the ProtoEncoder
if neither artifact already exists.

Usage:
    python scripts/bootstrap.py          # run from project root
    python scripts/bootstrap.py --skip-training  # download model only
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _PROJECT_ROOT / "src" / "models"

_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
_HAND_LANDMARKER_PATH = _MODELS_DIR / "hand_landmarker.task"
_ENCODER_PATH = _MODELS_DIR / "pretrained_encoder.pth"


class ModelBootstrapper:
    """Handles downloading and generating required model files."""

    def __init__(self, skip_training: bool = False) -> None:
        self._skip_training = skip_training

    def run(self) -> None:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._download_hand_landmarker()
        self._ensure_encoder()
        print("\n[bootstrap] All models ready.")

    def _download_hand_landmarker(self) -> None:
        if _HAND_LANDMARKER_PATH.exists():
            print(f"[bootstrap] Hand landmarker already exists: {_HAND_LANDMARKER_PATH}")
            return

        print(f"[bootstrap] Downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(_HAND_LANDMARKER_URL, _HAND_LANDMARKER_PATH)
        print(f"[bootstrap] Saved to {_HAND_LANDMARKER_PATH}")

    def _ensure_encoder(self) -> None:
        if _ENCODER_PATH.exists():
            print(f"[bootstrap] Encoder checkpoint already exists: {_ENCODER_PATH}")
            return

        if self._skip_training:
            print("[bootstrap] --skip-training set; skipping encoder training.")
            return

        print("[bootstrap] Training ProtoEncoder (this takes a few minutes) ...")
        self._train_encoder()

    def _train_encoder(self) -> None:
        src_dir = str(_PROJECT_ROOT / "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        from proto_net.meta_train import train

        train(episodes=10_000, save_path=_ENCODER_PATH)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap model assets after fresh clone")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only download models; do not train the encoder.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ModelBootstrapper(skip_training=args.skip_training).run()


if __name__ == "__main__":
    main()

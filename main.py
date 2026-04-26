import argparse
import sys
from pathlib import Path
from typing import List, Optional


_MODELS_DIR = Path(__file__).resolve().parent / "src" / "models"
_REQUIRED_MODELS = ("hand_landmarker.task", "pretrained_encoder.pth")


def _check_models() -> None:
    """Warn clearly if model files are missing after a fresh clone."""
    missing = [name for name in _REQUIRED_MODELS if not (_MODELS_DIR / name).exists()]
    if not missing:
        return

    msg = (
        "\n[ERROR] Required model files are missing:\n"
        + "".join(f"  - src/models/{name}\n" for name in missing)
        + "\nRun the bootstrap script first:\n"
        + "    python scripts/bootstrap.py\n"
        + "\nSee README.md for details.\n"
    )
    sys.exit(msg)


class ProjectRunner:
    def __init__(self) -> None:
        self._repo_root = Path(__file__).resolve().parent
        self._src_dir = self._repo_root / "src"

    def run(self, mode: str) -> None:
        self._ensure_src_on_path()
        _check_models()

        if mode == "proton":
            self._run_proton()
            return
        if mode == "gesture":
            self._run_gesture()
            return
        if mode == "gloved":
            self._run_gloved()
            return

        raise ValueError(f"Unknown mode: {mode!r}")

    def _ensure_src_on_path(self) -> None:
        if not self._src_dir.exists():
            raise FileNotFoundError(f"Expected folder not found: {self._src_dir}")

        src_str = str(self._src_dir)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)

    def _run_proton(self) -> None:
        # Proton.py runs its main loop at import time.
        import Proton  # noqa: F401

    def _run_gesture(self) -> None:
        import Gesture_Controller

        controller = Gesture_Controller.GestureController()
        controller.start()

    def _run_gloved(self) -> None:
        import Gesture_Controller_Gloved

        controller = Gesture_Controller_Gloved.GestureController()
        controller.start()


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture Virtual Mouse runner")
    parser.add_argument(
        "--mode",
        choices=("proton", "gesture", "gloved"),
        default="proton",
        help="What to run: full assistant UI (proton), gesture-only, or gloved mode.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    ProjectRunner().run(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


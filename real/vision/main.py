"""Standalone vision entry point: runs the tracker without ROS.

Prints the filtered ball state every frame and (optionally) loads a
previously-saved camera calibration to undistort each frame before
detection.

Usage::

    python -m vision.main --calibration calibration/calibration_data.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ballplate.state import BallState
from vision.tracker.pipeline import TrackingPipeline

#: Default calibration ships next to this module; resolves to an absolute path
#: so the entry point works regardless of the caller's CWD.
_DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parent / "calibration_files" / "calibration_data.npz"
)


def _load_calibration(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not path.exists():
        print(f"No calibration file at {path}, running without undistortion.")
        return None, None
    data = np.load(str(path))
    print(f"Loaded camera calibration from {path}.")
    return data["camera_matrix"], data["dist_coeffs"]


def _print_callback(ball: BallState, debug: dict) -> None:
    if not ball.valid:
        return
    print(
        f"t={ball.timestamp:14.3f}s  "
        f"pos=({ball.x * 1000.0:+7.1f},{ball.y * 1000.0:+7.1f}) mm  "
        f"vel=({ball.vx * 1000.0:+7.1f},{ball.vy * 1000.0:+7.1f}) mm/s  "
        f"markers={debug['markers_found']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="vision.main")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=_DEFAULT_CALIBRATION,
        help="Path to a .npz produced by vision.tools.calibrate_camera.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable the OpenCV debug window (headless mode).",
    )
    args = parser.parse_args()

    camera_matrix, dist_coeffs = _load_calibration(args.calibration)
    pipeline = TrackingPipeline(camera_matrix, dist_coeffs)
    pipeline.run(callback=_print_callback, show_debug=not args.no_debug)


if __name__ == "__main__":
    main()

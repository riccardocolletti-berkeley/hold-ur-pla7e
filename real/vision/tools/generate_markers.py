"""Render the four corner ArUco markers as PNGs for printing.

Each marker is rendered with a generous white border so the detector still
finds it after print bleed and uneven cropping. Run as a module so the local
``vision`` package is on the import path::

    python -m vision.tools.generate_markers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from vision.config import MARKER_IDS

_DEFAULT_SIZE_PX = 200
_BORDER_PX = 40


def generate_markers(
    marker_ids: list[int] = MARKER_IDS,
    size_px: int = _DEFAULT_SIZE_PX,
    out_dir: Path = Path("."),
) -> None:
    """Save one PNG per marker id under ``out_dir``."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mid in marker_ids:
        img = cv2.aruco.generateImageMarker(aruco_dict, mid, size_px)
        bordered = cv2.copyMakeBorder(
            img,
            _BORDER_PX,
            _BORDER_PX,
            _BORDER_PX,
            _BORDER_PX,
            cv2.BORDER_CONSTANT,
            value=255,
        )
        path = out_dir / f"marker_{mid}.png"
        cv2.imwrite(str(path), bordered)
        print(f"Saved {path}  (print at ~40-50 mm)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vision.tools.generate_markers")
    parser.add_argument(
        "--size-px",
        type=int,
        default=_DEFAULT_SIZE_PX,
        help="Marker side length in pixels before bordering.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help="Directory to write the PNG files into."
    )
    args = parser.parse_args()
    generate_markers(size_px=args.size_px, out_dir=args.out_dir)


if __name__ == "__main__":
    main()

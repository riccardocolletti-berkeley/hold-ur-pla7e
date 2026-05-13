"""Interactive checkerboard camera calibration.

Run the script, hold the checkerboard in roughly twenty different poses, press
``s`` when the corners are drawn on screen to save a frame, then ``c`` to
compute the calibration. The resulting ``camera_matrix`` and ``dist_coeffs``
are written to ``calibration_data.npz`` next to the saved frames.

Usage::

    python -m vision.tools.calibrate_camera --camera-index 0 --out-dir calibration
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

#: Default 9x6 squares (8x5 inner corners) checkerboard with 30 mm squares.
_DEFAULT_CHECKERBOARD = (8, 5)
_DEFAULT_SQUARE_SIZE_M = 0.030


def collect_images(
    camera_index: int,
    save_dir: Path,
    checkerboard: tuple[int, int] = _DEFAULT_CHECKERBOARD,
) -> Path | None:
    """Capture frames with detectable checkerboard corners until ``c`` is pressed."""
    save_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera_index)
    count = 0

    print("Controls: s = save frame, c = compute calibration, q = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, checkerboard, corners, found)
            cv2.putText(
                view,
                "CORNERS FOUND - press 's' to save",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                view, "No corners detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        cv2.putText(
            view,
            f"Saved: {count} frames",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Calibration", view)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and found:
            path = save_dir / f"calib_{count:03d}.png"
            cv2.imwrite(str(path), frame)
            print(f"[{count + 1}] Saved {path}")
            count += 1
        elif key == ord("c"):
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return save_dir


def calibrate(
    image_dir: Path,
    checkerboard: tuple[int, int] = _DEFAULT_CHECKERBOARD,
    square_size_m: float = _DEFAULT_SQUARE_SIZE_M,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Fit intrinsics from the saved checkerboard images."""
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0 : checkerboard[0], 0 : checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_m

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []

    images = sorted(glob.glob(str(image_dir / "*.png")))
    print(f"Found {len(images)} calibration images.")

    gray = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, checkerboard, None)
        if not found:
            continue
        refined = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )
        obj_points.append(objp)
        img_points.append(refined)

    if not obj_points:
        print("No valid calibration images found.")
        return None, None

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    print(f"Reprojection error: {rms:.4f} pixels")
    print(f"Camera matrix:\n{camera_matrix}")

    out_path = image_dir / "calibration_data.npz"
    np.savez(str(out_path), camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Saved {out_path}")
    return camera_matrix, dist_coeffs


def _parse_inner_corners(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected COLS,ROWS (e.g. 8,5)")
    return int(parts[0]), int(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(prog="vision.tools.calibrate_camera")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("calibration"))
    parser.add_argument(
        "--inner-corners",
        type=_parse_inner_corners,
        default=_DEFAULT_CHECKERBOARD,
        metavar="COLS,ROWS",
        help="Inner corners of the checkerboard (default: 8,5).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=_DEFAULT_SQUARE_SIZE_M,
        help="Side of one checkerboard square in metres (default: 0.030).",
    )
    args = parser.parse_args()

    save_dir = collect_images(args.camera_index, args.out_dir, args.inner_corners)
    if save_dir is not None:
        calibrate(save_dir, args.inner_corners, args.square_size)


if __name__ == "__main__":
    main()

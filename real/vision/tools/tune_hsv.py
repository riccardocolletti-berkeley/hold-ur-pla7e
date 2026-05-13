"""Interactive HSV threshold tuner for the ball detector.

Opens a window with six trackbars (low/high H, S, V) and shows both the live
camera feed and the resulting binary mask side by side. Drag the bars until
only the ball is visible in the mask, then press ``q`` to print the chosen
range so it can be pasted back into :mod:`vision.config`.

Run as::

    python -m vision.tools.tune_hsv --camera-index 0
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from vision.config import BALL_HSV_LOWER, BALL_HSV_UPPER

_WINDOW = "HSV Tuner"
_CONTROLS = "Controls"
_WARMUP_FRAMES = 30


def tune(camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {camera_index}")
        return

    for _ in range(_WARMUP_FRAMES):
        cap.read()

    cv2.namedWindow(_CONTROLS, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_CONTROLS, 400, 300)
    cv2.createTrackbar("H_lo", _CONTROLS, int(BALL_HSV_LOWER[0]), 179, lambda _: None)
    cv2.createTrackbar("S_lo", _CONTROLS, int(BALL_HSV_LOWER[1]), 255, lambda _: None)
    cv2.createTrackbar("V_lo", _CONTROLS, int(BALL_HSV_LOWER[2]), 255, lambda _: None)
    cv2.createTrackbar("H_hi", _CONTROLS, int(BALL_HSV_UPPER[0]), 179, lambda _: None)
    cv2.createTrackbar("S_hi", _CONTROLS, int(BALL_HSV_UPPER[1]), 255, lambda _: None)
    cv2.createTrackbar("V_hi", _CONTROLS, int(BALL_HSV_UPPER[2]), 255, lambda _: None)

    print("Drag the sliders until only the ball is visible. Press 'q' to print values.")

    lo = np.array(BALL_HSV_LOWER)
    hi = np.array(BALL_HSV_UPPER)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = np.array(
            [
                cv2.getTrackbarPos("H_lo", _CONTROLS),
                cv2.getTrackbarPos("S_lo", _CONTROLS),
                cv2.getTrackbarPos("V_lo", _CONTROLS),
            ]
        )
        hi = np.array(
            [
                cv2.getTrackbarPos("H_hi", _CONTROLS),
                cv2.getTrackbarPos("S_hi", _CONTROLS),
                cv2.getTrackbarPos("V_hi", _CONTROLS),
            ]
        )

        mask = cv2.inRange(hsv, lo, hi)
        masked = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.putText(
            frame,
            f"Lo: H={lo[0]} S={lo[1]} V={lo[2]}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Hi: H={hi[0]} S={hi[1]} V={hi[2]}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Mask pixels: {cv2.countNonZero(mask)}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow(_WINDOW, np.hstack([frame, masked]))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nCopy these into vision/config.yaml under `ball_detection`:")
    print(f"  hsv_lower: {[int(x) for x in lo]}")
    print(f"  hsv_upper: {[int(x) for x in hi]}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="vision.tools.tune_hsv")
    parser.add_argument("--camera-index", type=int, default=4)
    args = parser.parse_args()
    tune(args.camera_index)


if __name__ == "__main__":
    main()

"""HSV-based ball detector.

Returns the most circular blob inside the configured HSV range that also
satisfies the minimum / maximum radius constraints. Tuning of the HSV bands
lives in :mod:`vision.tools.tune_hsv` to keep this class strictly runtime.
"""

from __future__ import annotations

import cv2
import numpy as np

from vision.config import (
    BALL_HSV_LOWER,
    BALL_HSV_UPPER,
    BALL_MAX_RADIUS_PX,
    BALL_MIN_RADIUS_PX,
)


class BallDetector:
    """Per-frame ball detection in pixel coordinates."""

    def __init__(self) -> None:
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray) -> tuple[float, float, float] | None:
        """Return ``(u, v, radius_px)`` for the most circular candidate, or ``None``."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)
        mask = cv2.erode(mask, self._kernel, iterations=1)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best: tuple[float, float, float] | None = None
        best_circularity = 0.0
        for c in contours:
            perimeter = cv2.arcLength(c, True)
            if perimeter <= 0.0:
                continue
            area = cv2.contourArea(c)
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)

            ((u, v), radius) = cv2.minEnclosingCircle(c)
            if radius < BALL_MIN_RADIUS_PX or radius > BALL_MAX_RADIUS_PX:
                continue

            if circularity > best_circularity:
                best_circularity = circularity
                best = (float(u), float(v), float(radius))

        return best

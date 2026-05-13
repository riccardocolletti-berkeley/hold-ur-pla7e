"""Turn a raw screen-space stroke into a safe plate-local trajectory.

Pipeline, in order: pixels-to-metres, Gaussian smoothing, optional close
(append the first sample to the end), arc-length resample to a fixed
sample count, and finally a clamp to the safe rectangle. The canvas
y-axis is flipped so plate-frame +y points "up" as drawn.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def pixels_to_meters(
    points_px,
    canvas_size_px,
    plate_size_m,
    ball_radius_m,
    margin_m,
    safe_inset_frac,
):
    """Map the inner ``(1 - 2 * safe_inset_frac)`` box of the canvas onto
    the safe plate rectangle, flipping y so plate-frame +y points up."""
    cw, ch = canvas_size_px
    px, py = plate_size_m

    safe_x = max(0.0, px / 2.0 - ball_radius_m - margin_m)
    safe_y = max(0.0, py / 2.0 - ball_radius_m - margin_m)

    half_w = (cw / 2.0) * (1.0 - safe_inset_frac)
    half_h = (ch / 2.0) * (1.0 - safe_inset_frac)

    pts = np.asarray(points_px, dtype=float)
    xm = (pts[:, 0] - cw / 2.0) / half_w * safe_x
    ym = -(pts[:, 1] - ch / 2.0) / half_h * safe_y
    return np.column_stack([xm, ym]), (safe_x, safe_y)


def smooth(points, sigma):
    """Gaussian smooth a 2D path; sigma is in samples."""
    if len(points) < 3:
        return points
    x = gaussian_filter1d(points[:, 0], sigma=sigma, mode="nearest")
    y = gaussian_filter1d(points[:, 1], sigma=sigma, mode="nearest")
    return np.column_stack([x, y])


def resample_arclength(points, n_out):
    """Resample to ``n_out`` points equally spaced along arc length."""
    if len(points) < 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total == 0.0:
        return np.repeat(points[:1], n_out, axis=0)
    s_target = np.linspace(0.0, total, n_out)
    x = np.interp(s_target, s, points[:, 0])
    y = np.interp(s_target, s, points[:, 1])
    return np.column_stack([x, y])


def clamp_to_safe(points, safe_x, safe_y):
    """Clamp to the safe rectangle so the ball never reaches the plate edge."""
    out = points.copy()
    np.clip(out[:, 0], -safe_x, safe_x, out=out[:, 0])
    np.clip(out[:, 1], -safe_y, safe_y, out=out[:, 1])
    return out


def force_close(points):
    """Append the first sample to the end so the arc-length resample sees
    the closing segment as a real edge."""
    if len(points) < 2:
        return points
    return np.vstack([points, points[:1]])


def process(
    raw_points_px,
    *,
    canvas_size_px,
    plate_size_m,
    ball_radius_m,
    mode,
    n_out=200,
    smooth_sigma=1.5,
    margin_m=0.01,
    safe_inset_frac=0.1,
):
    """Run the full pipeline; returns an ``(N, 2)`` array of plate-local metres."""
    pts_m, (safe_x, safe_y) = pixels_to_meters(
        raw_points_px,
        canvas_size_px,
        plate_size_m,
        ball_radius_m,
        margin_m,
        safe_inset_frac,
    )
    pts_m = smooth(pts_m, sigma=smooth_sigma)
    if mode == "closed":
        pts_m = force_close(pts_m)
    pts_m = resample_arclength(pts_m, n_out=n_out)
    pts_m = clamp_to_safe(pts_m, safe_x, safe_y)
    return pts_m

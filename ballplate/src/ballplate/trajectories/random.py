"""Random reference-trajectory factory for RL rollouts and benchmarks."""

from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline

from ballplate.state import PlateGeometry
from ballplate.trajectories.base import Reference
from ballplate.trajectories.parametric import Circle, Figure8, Stationary

ShapeName = Literal[
    "stationary",
    "slow_circle",
    "circle",
    "figure8",
    "random_spline",
]


def sample(
    shape: ShapeName,
    rng: np.random.Generator,
    plate: PlateGeometry,
    ball_radius: float,
    margin: float = 0.02,
) -> Reference:
    """Sample a random instance of ``shape`` that fits inside the safe area.

    Safe area = plate rectangle shrunk by ``ball_radius`` (no overhang)
    plus ``margin`` in metres.
    """
    hx = max(0.01, plate.half_x - ball_radius - margin)
    hy = max(0.01, plate.half_y - ball_radius - margin)
    rmax = max(0.01, min(hx, hy))

    if shape == "stationary":
        return Stationary()
    if shape == "slow_circle":
        return Circle(
            radius=float(rng.uniform(0.03, min(0.06, rmax))),
            period=float(rng.uniform(12.0, 20.0)),
        )
    if shape == "circle":
        return Circle(
            radius=float(rng.uniform(0.03, rmax)),
            period=float(rng.uniform(5.0, 15.0)),
        )
    if shape == "figure8":
        return Figure8(
            rx=float(rng.uniform(min(0.04, hx * 0.5), hx)),
            ry=float(rng.uniform(min(0.02, hy * 0.5), hy)),
            period=float(rng.uniform(6.0, 16.0)),
        )
    if shape == "random_spline":
        return _random_periodic_spline(rng, hx, hy)
    raise ValueError(f"Unknown shape: {shape!r}")


# =============================================================================
# Internals
# =============================================================================


def _random_periodic_spline(
    rng: np.random.Generator,
    hx: float,
    hy: float,
    n_waypoints: int = 6,
    period_range: tuple[float, float] = (8.0, 18.0),
) -> "_PeriodicSpline":
    """Closed cubic spline through ``n_waypoints`` random points in the safe area."""
    period = float(rng.uniform(*period_range))
    ts = np.linspace(0.0, period, n_waypoints + 1)
    pts = rng.uniform(-1.0, 1.0, size=(n_waypoints, 2))
    pts[:, 0] *= hx * 0.85  # inset so the spline can bulge without exiting
    pts[:, 1] *= hy * 0.85
    pts = np.vstack([pts, pts[:1]])  # close the loop for periodic boundary
    spl_x = CubicSpline(ts, pts[:, 0], bc_type="periodic")
    spl_y = CubicSpline(ts, pts[:, 1], bc_type="periodic")
    return _PeriodicSpline(spl_x, spl_y, period)


class _PeriodicSpline:
    """Periodic cubic-spline reference, built only by ``sample``."""

    def __init__(self, spl_x: CubicSpline, spl_y: CubicSpline, period: float):
        self._x = spl_x
        self._y = spl_y
        self._period = period

    @property
    def period(self) -> float:
        return self._period

    def evaluate(self, t: float) -> tuple[float, float, float, float]:
        tau = t % self._period
        return (
            float(self._x(tau)),
            float(self._y(tau)),
            float(self._x(tau, 1)),
            float(self._y(tau, 1)),
        )

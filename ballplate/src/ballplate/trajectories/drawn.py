"""Polyline reference trajectory loaded from / saved to JSON.

JSON schema (version 1)::

    {
        "version":     1,
        "points_xy_m": [[x0, y0], [x1, y1], ...],   # arc-length resampled
        "duration_s":  <float>,                      # one-direction traversal
        "end_mode":    "loop" | "stop" | "pingpong",
        "name":        <str, optional>,
        "created":     <unix seconds, optional>
    }

Points are assumed pre-resampled at constant arc-length spacing, so a
uniform time grid produces constant ground speed.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

EndMode = Literal["loop", "stop", "pingpong"]
_VALID_END_MODES = ("loop", "stop", "pingpong")
_SCHEMA_VERSION = 1


class DrawnPath:
    """Polyline reference resampled at constant arc-length.

    ``end_mode`` controls post-traversal behaviour:
        ``loop``      jump from the last point back to the first
        ``stop``      hold the last point with zero velocity
        ``pingpong``  traverse forward, then backward, repeating

    Not a frozen dataclass: caches per-segment velocities for fast
    ``evaluate`` calls.
    """

    def __init__(
        self,
        points_xy_m: np.ndarray | Sequence[Sequence[float]],
        duration_s: float,
        end_mode: EndMode = "loop",
    ):
        pts = np.asarray(points_xy_m, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
            raise ValueError(
                "points_xy_m must have shape (N, 2) with N >= 2; " f"got shape {pts.shape}"
            )
        if duration_s <= 0.0:
            raise ValueError(f"duration_s must be positive; got {duration_s}")
        if end_mode not in _VALID_END_MODES:
            raise ValueError(f"end_mode must be one of {_VALID_END_MODES}; got {end_mode!r}")

        self.points: np.ndarray = pts
        self.duration_s: float = float(duration_s)
        self.end_mode: EndMode = end_mode

        # Uniform time grid; combined with arc-length-resampled points this
        # gives constant ground speed.
        self._ts = np.linspace(0.0, self.duration_s, len(pts))
        dt = np.diff(self._ts)
        self._seg_vx = np.diff(pts[:, 0]) / dt
        self._seg_vy = np.diff(pts[:, 1]) / dt

    # ----------------------------------------------------------- Reference --

    @property
    def period(self) -> float:
        """One forward-traversal duration [s]."""
        return self.duration_s

    def evaluate(self, t: float) -> tuple[float, float, float, float]:
        tau, direction = self._phase(t)
        x = float(np.interp(tau, self._ts, self.points[:, 0]))
        y = float(np.interp(tau, self._ts, self.points[:, 1]))

        seg = int(np.searchsorted(self._ts, tau, side="right")) - 1
        seg = max(0, min(seg, len(self._seg_vx) - 1))
        vx = self._seg_vx[seg] * direction
        vy = self._seg_vy[seg] * direction
        return x, y, vx, vy

    def _phase(self, t: float) -> tuple[float, float]:
        """``(tau in [0, T], direction in {-1, 0, +1})`` for absolute ``t``."""
        T = self.duration_s
        if self.end_mode == "stop":
            if t >= T:
                return T, 0.0
            return max(0.0, t), 1.0
        if self.end_mode == "pingpong":
            cycle = 2.0 * T
            tau = t % cycle if t >= 0.0 else 0.0
            if tau <= T:
                return tau, 1.0
            return cycle - tau, -1.0
        # loop
        return (t % T if t >= 0.0 else 0.0), 1.0

    # --------------------------------------------------------- Serialization --

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DrawnPath":
        return cls(
            points_xy_m=data["points_xy_m"],
            duration_s=data["duration_s"],
            end_mode=data.get("end_mode", "loop"),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "DrawnPath":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _SCHEMA_VERSION,
            "points_xy_m": self.points.tolist(),
            "duration_s": self.duration_s,
            "end_mode": self.end_mode,
        }

    def to_file(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

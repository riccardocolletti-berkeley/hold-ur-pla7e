"""Ball-on-plate state types. Plate-local frame, SI units."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BallState:
    """Ball pose and velocity in the plate frame.

    ``valid=False`` flags a fallback sample (stale message, Kalman prediction
    without a fresh detection); callers should hold the last action or trigger
    a safety response rather than acting on it.
    """

    x: float
    y: float
    vx: float
    vy: float
    timestamp: float
    valid: bool = True


@dataclass(frozen=True)
class PlateGeometry:
    """Rectangular plate centered at the plate-frame origin."""

    size_x: float
    size_y: float
    thickness: float

    @property
    def half_x(self) -> float:
        return self.size_x / 2.0

    @property
    def half_y(self) -> float:
        return self.size_y / 2.0

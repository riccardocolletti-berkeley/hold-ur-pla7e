"""Closed-form parametric references implementing the ``Reference`` protocol.

Plate-local SI units (metres, metres/second). Frozen dataclasses; safe to
share across threads.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Stationary:
    """Fixed target point. ``period`` is a placeholder positive value."""

    x: float = 0.0
    y: float = 0.0

    @property
    def period(self) -> float:
        return 1.0

    def evaluate(self, t: float) -> tuple[float, float, float, float]:
        return self.x, self.y, 0.0, 0.0


@dataclass(frozen=True)
class Circle:
    """Circular path centred at the plate origin.

    With ``omega = 2 pi / period``::

        x(t)  =  R cos(omega t)
        y(t)  =  R sin(omega t)
        vx(t) = -R omega sin(omega t)
        vy(t) =  R omega cos(omega t)
    """

    radius: float  # path radius [m]
    period: float  # one revolution [s]

    def evaluate(self, t: float) -> tuple[float, float, float, float]:
        omega = 2.0 * math.pi / self.period
        c, s = math.cos(omega * t), math.sin(omega * t)
        return (
            self.radius * c,
            self.radius * s,
            -self.radius * omega * s,
            self.radius * omega * c,
        )


@dataclass(frozen=True)
class Figure8:
    """Lissajous figure-eight centred at the plate origin.

    With ``omega = 2 pi / period`` (period = one full pass)::

        x(t)  =  rx sin(omega t)
        y(t)  =  ry sin(2 omega t)
        vx(t) =  rx omega cos(omega t)
        vy(t) =  2 ry omega cos(2 omega t)
    """

    rx: float  # half-amplitude along x [m]
    ry: float  # half-amplitude along y [m]
    period: float  # one pass [s]

    def evaluate(self, t: float) -> tuple[float, float, float, float]:
        omega = 2.0 * math.pi / self.period
        s, c = math.sin(omega * t), math.cos(omega * t)
        s2, c2 = math.sin(2.0 * omega * t), math.cos(2.0 * omega * t)
        return (
            self.rx * s,
            self.ry * s2,
            self.rx * omega * c,
            2.0 * self.ry * omega * c2,
        )

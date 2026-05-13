"""Decoupled PID for the planar ball-on-plate problem.

Pure-Python: takes a ``BallState`` plus a reference (pos, vel, acc) and
returns a 2-D virtual control ``(ux, uy)``. The platform-specific wrapper
turns that into a plate-tilt or joint command.
"""

from dataclasses import dataclass
from math import inf, sqrt

from ballplate.state import BallState


@dataclass(frozen=True)
class PidGains:
    """Gains and saturation limits for :class:`PidController`.

    ``error_clip`` caps the L2 norm of the 2-D position error (direction
    preserved): when ``||(ex, ey)|| > error_clip`` the vector is rescaled
    to that radius before feeding P and I. The derivative term sees the
    raw velocity error; upstream callers can pre-clip it via
    ``ballplate.safety.velocity_clip``.
    """

    kp: float
    ki: float
    kd: float
    kff: float = 0.0
    error_clip: float = inf
    windup_limit: float = inf
    max_output: float = inf


class PidController:
    """Stateful PID controller, one integrator per axis."""

    def __init__(self, gains: PidGains):
        self.gains = gains
        self._integral_x: float = 0.0
        self._integral_y: float = 0.0

    @property
    def integral(self) -> tuple[float, float]:
        """Current ``(Ix, Iy)`` integrator state."""
        return self._integral_x, self._integral_y

    def reset(self) -> None:
        """Zero both integrators."""
        self._integral_x = 0.0
        self._integral_y = 0.0

    def step(
        self,
        ball: BallState,
        target_pos: tuple[float, float],
        target_vel: tuple[float, float] = (0.0, 0.0),
        target_acc: tuple[float, float] = (0.0, 0.0),
        dt: float = 0.0,
    ) -> tuple[float, float]:
        """Advance one tick and return ``(ux, uy)`` in the plate frame.

        Pass ``dt = 0`` to skip integration on this step (useful for a
        one-shot snapshot read).
        """
        g = self.gains

        ex = ball.x - target_pos[0]
        ey = ball.y - target_pos[1]
        # Isotropic clamp on the position-error norm.
        err_norm = sqrt(ex * ex + ey * ey)
        if err_norm > g.error_clip:
            s = g.error_clip / err_norm
            ex *= s
            ey *= s
        evx = ball.vx - target_vel[0]
        evy = ball.vy - target_vel[1]
        tax, tay = target_acc

        # Anti-windup: clamp the integrator before it enters the P sum.
        self._integral_x = _clip(self._integral_x + ex * dt, -g.windup_limit, g.windup_limit)
        self._integral_y = _clip(self._integral_y + ey * dt, -g.windup_limit, g.windup_limit)

        ux = g.kp * ex + g.ki * self._integral_x + g.kd * evx - g.kff * tax
        uy = g.kp * ey + g.ki * self._integral_y + g.kd * evy - g.kff * tay

        ux = _clip(ux, -g.max_output, g.max_output)
        uy = _clip(uy, -g.max_output, g.max_output)
        return ux, uy


def _clip(value: float, lo: float, hi: float) -> float:
    """Scalar clamp to ``[lo, hi]``."""
    return max(lo, min(hi, value))

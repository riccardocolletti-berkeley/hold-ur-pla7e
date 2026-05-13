"""Composable wrappers around ``Reference`` instances."""

from dataclasses import dataclass

from ballplate.trajectories.base import Reference


@dataclass(frozen=True)
class SmoothApproach:
    """Blend from a fixed start pose into an inner reference.

    For ``t < blend_time`` the wrapper interpolates between the constant
    start ``(start_x, start_y)`` and the inner reference with a cubic
    smoothstep ``s(tau) = 3*tau^2 - 2*tau^3``, ``tau = t / blend_time``.
    The smoothstep is C^1 at both ends, so velocity is continuous at
    ``t = 0`` (zero from rest) and at ``t = blend_time`` (matches the
    inner reference). For ``t >= blend_time`` the inner reference passes
    through unchanged. ``blend_time <= 0`` disables the wrapper.
    """

    inner: Reference
    start_x: float
    start_y: float
    blend_time: float

    @property
    def period(self) -> float:
        return self.inner.period

    def evaluate(self, t: float) -> tuple[float, float, float, float]:
        if self.blend_time <= 0.0 or t >= self.blend_time:
            return self.inner.evaluate(t)

        ix, iy, ivx, ivy = self.inner.evaluate(t)

        tau = t / self.blend_time
        s = 3.0 * tau * tau - 2.0 * tau * tau * tau
        ds_dt = (6.0 * tau - 6.0 * tau * tau) / self.blend_time

        x = (1.0 - s) * self.start_x + s * ix
        y = (1.0 - s) * self.start_y + s * iy
        # d/dt[(1-s) p0 + s q(t)] = ds_dt (q - p0) + s q_dot
        vx = ds_dt * (ix - self.start_x) + s * ivx
        vy = ds_dt * (iy - self.start_y) + s * ivy
        return x, y, vx, vy

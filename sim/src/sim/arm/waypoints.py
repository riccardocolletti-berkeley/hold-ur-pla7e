"""Joint-space cubic-spline waypoint planner for the UR5e arm.

The arm follows a sequence of joint-position waypoints connected by zero-
velocity cubic segments. Each segment is independent per joint, which keeps
the math trivial; the controllers we use (PID and RL) drive the plate via a
small joint-space correction added on top of the planned trajectory.
"""

from collections.abc import Sequence

import numpy as np


class JointTrajectory:
    """Per-joint cubic spline through joint-space waypoints.

    Boundary conditions: zero velocity at every waypoint (the arm rests
    momentarily before each segment). For seamless motion across waypoints
    consider feeding planned velocities; the simple zero-velocity case
    matches the existing controllers.
    """

    def __init__(self, waypoints: Sequence[np.ndarray], durations: Sequence[float]):
        if len(waypoints) < 2:
            raise ValueError(f"Need at least two waypoints; got {len(waypoints)}")
        if len(durations) != len(waypoints) - 1:
            raise ValueError(f"Expected {len(waypoints) - 1} durations, got {len(durations)}")

        self.n_segments = len(waypoints) - 1
        self.durations = np.asarray(durations, dtype=float)
        # Cumulative segment times: segment i covers (cum[i], cum[i+1]].
        self.cum_times = np.concatenate([[0.0], np.cumsum(self.durations)])
        self.total_time = float(self.cum_times[-1])
        self.n_joints = int(np.asarray(waypoints[0]).shape[0])

        self.segments = []
        for seg in range(self.n_segments):
            wp0 = np.asarray(waypoints[seg], dtype=float)
            wp1 = np.asarray(waypoints[seg + 1], dtype=float)
            joint_coeffs = [
                _cubic_coeffs(wp0[j], wp1[j], 0.0, 0.0, self.durations[seg])
                for j in range(self.n_joints)
            ]
            self.segments.append(joint_coeffs)

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(q, v, a)`` at time `t`. Saturates to the last segment past `total_time`."""
        t_clamped = float(np.clip(t, 0.0, self.total_time - 1e-8))
        seg = int(np.searchsorted(self.cum_times[1:], t_clamped, side="right"))
        seg = min(seg, self.n_segments - 1)
        t_local = t_clamped - self.cum_times[seg]

        q = np.zeros(self.n_joints)
        v = np.zeros(self.n_joints)
        a = np.zeros(self.n_joints)
        for j in range(self.n_joints):
            q[j], v[j], a[j] = _eval_cubic(self.segments[seg][j], t_local)
        return q, v, a


# ============================================================================
# Cubic-spline math (module-level so it stays unit-testable in isolation)
# ============================================================================


def _cubic_coeffs(
    q0: float, qf: float, v0: float, vf: float, T: float
) -> tuple[float, float, float, float]:
    """Coefficients (a0, a1, a2, a3) of q(t) = a0 + a1 t + a2 t² + a3 t³.

    Boundary conditions: ``q(0) = q0``, ``q(T) = qf``, ``q̇(0) = v0``, ``q̇(T) = vf``.
    """
    a0 = q0
    a1 = v0
    a2 = (3.0 * (qf - q0) / T**2) - (2.0 * v0 / T) - (vf / T)
    a3 = (-2.0 * (qf - q0) / T**3) + ((vf + v0) / T**2)
    return a0, a1, a2, a3


def _eval_cubic(coeffs: tuple[float, float, float, float], t: float) -> tuple[float, float, float]:
    """Evaluate position, velocity, and acceleration of a cubic at time `t`."""
    a0, a1, a2, a3 = coeffs
    q = a0 + a1 * t + a2 * t**2 + a3 * t**3
    v = a1 + 2.0 * a2 * t + 3.0 * a3 * t**2
    a = 2.0 * a2 + 6.0 * a3 * t
    return q, v, a

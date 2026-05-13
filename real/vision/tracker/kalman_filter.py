"""Tilt-aware Kalman filter for planar ball tracking.

Constant-velocity model augmented with the deterministic gravity component
induced by the plate tilt. The control input ``(a_x, a_y)`` is computed from
the commanded plate angles via :func:`tilt_to_accel`; without any call to
:meth:`BallKalmanFilter.set_control` the filter degrades cleanly to a pure
constant-velocity estimator.

State and measurement are expressed in the table frame, in SI units.
"""

import math

import numpy as np

from vision.config import (
    GRAVITY_M_S2,
    KF_MEASUREMENT_NOISE,
    KF_PROCESS_NOISE,
    ROLLING_FACTOR,
)


def tilt_to_accel(
    theta_x: float, theta_y: float, g: float = GRAVITY_M_S2, alpha: float = ROLLING_FACTOR
) -> tuple[float, float]:
    """Convert plate tilt angles (radians) to in-plane acceleration (m/s²).

    Sign convention matches the table frame defined in :mod:`vision.config`:

        theta_y > 0  =>  +x edge dips down  =>  ball rolls toward +x  =>  a_x > 0
        theta_x > 0  =>  +y edge rises up   =>  ball rolls toward -y  =>  a_y < 0

    Flip the corresponding sign here if the physical setup uses a different
    rotation convention.
    """
    a_x = +alpha * g * math.sin(theta_y)
    a_y = -alpha * g * math.sin(theta_x)
    return a_x, a_y


class BallKalmanFilter:
    """Linear KF for a 2D ball with plate tilt as a control input.

    State:        ``[x, y, vx, vy]`` (m, m/s)
    Measurement:  ``[x, y]``         (m)
    Control:      ``[a_x, a_y]``     (m/s², table frame)
    """

    def __init__(self, dt: float = 1.0 / 60.0):
        self.initialized = False
        self.x = np.zeros(4)
        self.u = np.zeros(2)

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.R = np.eye(2) * (KF_MEASUREMENT_NOISE**2)
        self.P = np.eye(4) * 1.0

        self._build_dt_matrices(dt)

    # ------------------------------------------------------------------ time --

    def _build_dt_matrices(self, dt: float) -> None:
        """Rebuild F, B, and Q for the supplied step length."""
        self.dt = dt

        self.F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.B = np.array(
            [
                [0.5 * dt * dt, 0.0],
                [0.0, 0.5 * dt * dt],
                [dt, 0.0],
                [0.0, dt],
            ]
        )

        # Continuous-white-noise-acceleration Q. KF_PROCESS_NOISE is the
        # standard deviation of the unmodelled acceleration in m/s².
        sigma_a_sq = KF_PROCESS_NOISE**2
        self.Q = (
            np.array(
                [
                    [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                    [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
                    [dt**3 / 2.0, 0.0, dt**2, 0.0],
                    [0.0, dt**3 / 2.0, 0.0, dt**2],
                ]
            )
            * sigma_a_sq
        )

    # ----------------------------------------------------------- lifecycle --

    def reset(self, x: float, y: float) -> None:
        """Initialise the filter at a known position with zero velocity."""
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.eye(4) * 1.0
        self.u = np.zeros(2)
        self.initialized = True

    def set_control(self, a_x: float, a_y: float) -> None:
        """Set the in-plane acceleration applied during the next predict."""
        self.u = np.array([a_x, a_y])

    # ------------------------------------------------------------------ run --

    def predict(self, dt: float | None = None) -> None:
        """Propagate the state. Rebuilds F/B/Q if ``dt`` changes meaningfully."""
        # 0.1 ms tolerance. Wall-clock jitter at 60 fps is on the order
        # of microseconds; rebuilding F/B/Q for that would be wasteful.
        if dt is not None and abs(dt - self.dt) > 1e-4:
            self._build_dt_matrices(dt)
        self.x = self.F @ self.x + self.B @ self.u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_x: float, z_y: float) -> None:
        """Fuse a position measurement into the current estimate."""
        z = np.array([z_x, z_y])
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.H) @ self.P

    # ---------------------------------------------------------- accessors --

    def get_position(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    def get_velocity(self) -> tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

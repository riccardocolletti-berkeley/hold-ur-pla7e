"""Linear receding-horizon MPC for the planar ball-on-plate problem.

Output ``(ux, uy)`` follows the same convention as the PID: a desired
plate-tilt offset in radians, plate-local frame.

Per-axis decoupled plant under the small-angle approximation::

    ẍ = -alpha * g * u_x        (alpha = 5/7 for a uniform solid sphere)

Discretised exactly at ``params.dt`` (continuous-time ``A`` is nilpotent
of order 2)::

    A = [[1, dt], [0, 1]]
    B = [[-alpha*g*dt**2/2], [-alpha*g*dt]]

The unconstrained condensed QP has a closed form. Per-axis gain matrices
are pre-factored at construction time; only the first row is needed at
runtime (receding horizon), so each tick is two dot products per axis
plus an output clip. Constraints beyond the output box would need a real
QP solver (osqp etc.) and are deliberately omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any

import numpy as np

from ballplate.hardware import HardwareSpec
from ballplate.state import BallState
from ballplate.trajectories.base import Reference

# Fallback constants for tests / standalone scripts. Production callers
# should source these from ``ballplate.hardware`` via :meth:`MpcParams.from_hardware`.
_DEFAULT_ROLLING_FACTOR: float = 5.0 / 7.0
_DEFAULT_GRAVITY: float = 9.81


@dataclass(frozen=True)
class MpcParams:
    """Tuning for :class:`MpcController`.

    The per-axis cost minimised over the horizon is::

        sum_{k=1..N-1}  q_pos*(x_k - x_ref_k)^2 + q_vel*(v_k - v_ref_k)^2
                      + r_u * u_{k-1}^2
        + qf_pos*(x_N - x_ref_N)^2 + qf_vel*(v_N - v_ref_N)^2

    ``r_u > 0`` is required to keep the Hessian invertible. ``qf_*``
    weights the terminal state and yields the infinite-horizon-LQR limit
    for large ``N``.
    """

    horizon: int = 25
    dt: float = 1.0 / 60.0
    q_pos: float = 100.0
    q_vel: float = 10.0
    r_u: float = 1.0
    qf_pos: float = 500.0
    qf_vel: float = 50.0
    max_output: float = inf
    rolling_factor: float = _DEFAULT_ROLLING_FACTOR
    gravity: float = _DEFAULT_GRAVITY

    @classmethod
    def from_hardware(cls, hw: HardwareSpec, **overrides: Any) -> MpcParams:
        """Canonical constructor: pull ``rolling_factor`` and ``gravity`` from a HardwareSpec.

        ``hw`` is anything with a ``physics`` attribute (typically
        :class:`ballplate.hardware.HardwareSpec`). Other fields take their
        defaults; ``**overrides`` lets callers set them.
        """
        return cls(
            rolling_factor=hw.physics.rolling_factor,
            gravity=hw.physics.gravity,
            **overrides,
        )

    def validate(self) -> None:
        """Raise ``ValueError`` if any field is out of range."""
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        for name in ("q_pos", "q_vel", "qf_pos", "qf_vel"):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        # r_u = 0 would make H = Gamma^T Q Gamma rank-deficient when q_vel = 0.
        if self.r_u <= 0.0:
            raise ValueError(f"r_u must be > 0, got {self.r_u}")
        if self.rolling_factor <= 0.0:
            raise ValueError(f"rolling_factor must be > 0, got {self.rolling_factor}")
        if self.gravity <= 0.0:
            raise ValueError(f"gravity must be > 0, got {self.gravity}")


class MpcController:
    """Linear receding-horizon MPC, same call surface as :class:`PidController`.

    ``step`` builds a constant-reference horizon; ``step_with_reference``
    samples a :class:`ballplate.trajectories.Reference` along the horizon
    so the controller anticipates target motion (matters for circles and
    drawn paths). ``step_with_horizon`` takes a pre-sampled ``(N, 4)``
    array directly.
    """

    def __init__(self, params: MpcParams):
        params.validate()
        self.params = params
        self._build_axis_gains()

        # Scratch buffers for the hot path (500 Hz on the real arm).
        n = params.horizon
        self._ref_x_buf = np.empty(2 * n, dtype=float)
        self._ref_y_buf = np.empty(2 * n, dtype=float)
        self._horizon_buf = np.empty((n, 4), dtype=float)
        # Horizon time offsets: t_k = (k+1) * dt for k = 0..N-1.
        self._horizon_offsets = (np.arange(1, n + 1) * params.dt).astype(float)

    # =================================================== gain construction --

    def _build_axis_gains(self) -> None:
        """Pre-factor the per-axis gains.

        x and y decouple to identical 2-state-1-input subproblems. The
        condensed QP::

            min_U  ||Phi*s0 + Gamma*U - R||_Q^2  +  ||U||_R_block^2

        has closed-form optimum ``U* = K_x s0 + K_r R`` where
        ``H = Gamma^T Q Gamma + R_block`` is SPD. Under receding horizon
        we only apply ``U*[0]``, so we cache just the first row of ``K_x``
        and ``K_r``.
        """
        p = self.params
        n_steps = p.horizon
        dt = p.dt
        beta = p.rolling_factor * p.gravity  # alpha * g

        a_mat = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
        b_mat = np.array([[-beta * dt * dt / 2.0], [-beta * dt]], dtype=float)

        # Phi[k] = A^{k+1} for k = 0..N-1, stacked so the zero-input rollout
        # from s0 is ``Phi @ s0``.
        phi = np.zeros((2 * n_steps, 2), dtype=float)
        a_power = np.eye(2)
        for k in range(n_steps):
            a_power = a_mat @ a_power
            phi[2 * k : 2 * (k + 1), :] = a_power

        # Gamma[k, j] = A^{k-j} B for j <= k else 0; shape (2N, N).
        # Cache A^i B once (O(N), not O(N^2)).
        ab_cache: list[np.ndarray] = []
        ab = b_mat.copy()
        ab_cache.append(ab)
        for _ in range(n_steps - 1):
            ab = a_mat @ ab
            ab_cache.append(ab)
        gamma = np.zeros((2 * n_steps, n_steps), dtype=float)
        for k in range(n_steps):
            for j in range(k + 1):
                gamma[2 * k : 2 * (k + 1), j : j + 1] = ab_cache[k - j]

        # Block diagonal Q: stage Q for k = 1..N-1, terminal Qf at k = N.
        q_stage = np.diag([p.q_pos, p.q_vel])
        q_term = np.diag([p.qf_pos, p.qf_vel])
        q_block = np.zeros((2 * n_steps, 2 * n_steps), dtype=float)
        for k in range(n_steps - 1):
            q_block[2 * k : 2 * (k + 1), 2 * k : 2 * (k + 1)] = q_stage
        q_block[2 * (n_steps - 1) : 2 * n_steps, 2 * (n_steps - 1) : 2 * n_steps] = q_term

        r_block = p.r_u * np.eye(n_steps)

        h_mat = gamma.T @ q_block @ gamma + r_block
        l_mat = gamma.T @ q_block

        # Keep only the first row of each (N, .) gain since the hot path
        # only applies U*[0]. Stored as 1-D arrays for plain dot products.
        k_x = -np.linalg.solve(h_mat, l_mat @ phi)  # (N, 2)
        k_r = np.linalg.solve(h_mat, l_mat)  # (N, 2N)
        self._k_x_first = np.ascontiguousarray(k_x[0])  # (2,)
        self._k_r_first = np.ascontiguousarray(k_r[0])  # (2N,)

    # ============================================================ accessors --

    @property
    def horizon(self) -> int:
        return self.params.horizon

    @property
    def dt(self) -> float:
        return self.params.dt

    def reset(self) -> None:
        """No-op kept for interface parity with :class:`PidController`."""
        pass

    # ================================================================= step --

    def step(
        self,
        ball: BallState,
        target_pos: tuple[float, float],
        target_vel: tuple[float, float] = (0.0, 0.0),
        target_acc: tuple[float, float] = (0.0, 0.0),
        dt: float = 0.0,
    ) -> tuple[float, float]:
        """Step with a constant reference over the horizon.

        ``target_acc`` and the ``dt`` argument exist for parity with the
        PID interface and are unused (MPC's dt lives in ``MpcParams``).
        """
        del target_acc, dt
        # Stack into [x_1, v_1, x_2, v_2, ...] per axis (length 2N).
        self._ref_x_buf[0::2] = target_pos[0]
        self._ref_x_buf[1::2] = target_vel[0]
        self._ref_y_buf[0::2] = target_pos[1]
        self._ref_y_buf[1::2] = target_vel[1]
        return self._solve_both_axes(ball, self._ref_x_buf, self._ref_y_buf)

    def step_with_reference(
        self,
        ball: BallState,
        reference: Reference,
        now: float,
    ) -> tuple[float, float]:
        """Sample ``reference`` at ``now + k*dt`` (k = 1..N) and step the MPC."""
        buf = self._horizon_buf
        offsets = self._horizon_offsets
        for k in range(buf.shape[0]):
            xr, yr, vxr, vyr = reference.evaluate(now + offsets[k])
            buf[k, 0] = xr
            buf[k, 1] = yr
            buf[k, 2] = vxr
            buf[k, 3] = vyr
        return self.step_with_horizon(ball, buf)

    def step_with_horizon(
        self,
        ball: BallState,
        ref_horizon: np.ndarray,
    ) -> tuple[float, float]:
        """Step with a pre-sampled reference horizon.

        ``ref_horizon`` is ``(N, 4)``; row ``k`` holds
        ``(x_ref, y_ref, vx_ref, vy_ref)`` at time ``t + (k+1)*dt``. Row 0
        is the reference at the next step, not the current state.
        """
        n = self.params.horizon
        if ref_horizon.shape != (n, 4):
            raise ValueError(f"ref_horizon must have shape ({n}, 4), got {ref_horizon.shape}")
        self._ref_x_buf[0::2] = ref_horizon[:, 0]
        self._ref_x_buf[1::2] = ref_horizon[:, 2]
        self._ref_y_buf[0::2] = ref_horizon[:, 1]
        self._ref_y_buf[1::2] = ref_horizon[:, 3]
        return self._solve_both_axes(ball, self._ref_x_buf, self._ref_y_buf)

    # =========================================================== internals --

    def _solve_both_axes(
        self,
        ball: BallState,
        ref_x: np.ndarray,
        ref_y: np.ndarray,
    ) -> tuple[float, float]:
        """Apply the precomputed first-row gain to both axes."""
        kx = self._k_x_first
        kr = self._k_r_first
        ux = float(kx[0] * ball.x + kx[1] * ball.vx + kr @ ref_x)
        uy = float(kx[0] * ball.y + kx[1] * ball.vy + kr @ ref_y)

        umax = self.params.max_output
        ux = float(np.clip(ux, -umax, umax))
        uy = float(np.clip(uy, -umax, umax))
        return ux, uy

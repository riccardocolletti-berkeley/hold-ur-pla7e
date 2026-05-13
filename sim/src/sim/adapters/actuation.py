"""Translate a 2-D virtual control signal into MuJoCo joint torques.

Pipeline::

    (ux, uy) + plate_R  ->  omega_des  ->  delta_q  ->  clip  ->  low-pass
    q_target           ->  joint PD + gravity comp  ->  data.ctrl

The actuator writes into ``data.ctrl`` but does not advance the
simulator; the runner is responsible for calling ``mujoco.mj_step``.
"""

from collections.abc import Sequence

import mujoco
import numpy as np


def alpha_from_tau(dt: float, tau: float) -> float:
    """Return the discrete first-order low-pass alpha for timestep `dt`."""
    if tau <= 0.0:
        return 1.0
    return float(1.0 - np.exp(-float(dt) / float(tau)))


class JointActuator:
    """Stateful actuator: holds the low-pass filter buffer + cached gains."""

    def __init__(
        self,
        model,
        plate_site_id: int,
        nu: int,
        kp_joint: np.ndarray,
        kd_joint: np.ndarray,
        alpha: float,
        max_tilt: float,
        locked_joints: Sequence[int] = (),
    ):
        self.model = model
        self.plate_site_id = int(plate_site_id)
        self.nu = int(nu)
        self.Kp = np.diag(np.asarray(kp_joint, dtype=float))
        self.Kd = np.diag(np.asarray(kd_joint, dtype=float))
        self.alpha = float(alpha)
        self.max_tilt = float(max_tilt)
        self._dq_filtered = np.zeros(self.nu, dtype=float)
        # Free joint indices solve the lstsq; locked columns are dropped so the
        # tilt is realised by the wrist alone (or whichever subset is free).
        self._free_cols = np.setdiff1d(
            np.arange(self.nu),
            np.asarray(locked_joints, dtype=int),
            assume_unique=False,
        )

    def reset(self) -> None:
        """Clear the low-pass filter; call between independent runs."""
        self._dq_filtered[:] = 0.0

    def apply(
        self,
        data,
        ux: float,
        uy: float,
        plate_mat: np.ndarray,
        q_planned: np.ndarray,
        v_planned: np.ndarray,
    ) -> None:
        """Write the next torque vector into ``data.ctrl``.

        ``q_planned`` / ``v_planned`` are the nominal joint trajectory at the
        current step (for instance from `sim.arm.waypoints`). When the arm is
        locked, both can be the home pose with zero velocity.
        """
        x_hat = plate_mat[:, 0]
        y_hat = plate_mat[:, 1]

        # Desired plate angular velocity in world frame; sign convention matches
        # the report (positive ux → tilt that pushes ball back along -x).
        omega_des = -(ux * y_hat - uy * x_hat)

        # Jacobian step: ω = Jr · q̇ ⇒ Δq solves the underdetermined system in
        # least-squares sense. Using mj_jacSite gives us Jr at the plate centre.
        # Locked columns are dropped so lstsq only spends authority on the free
        # joints; the locked entries stay at zero and the joint PD pulls them
        # back to the planned home pose.
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.plate_site_id)
        Jr = jacr[:, : self.nu]
        dq_raw = np.zeros(self.nu)
        dq_raw[self._free_cols] = np.linalg.lstsq(Jr[:, self._free_cols], omega_des, rcond=None)[0]

        # Clip per-joint authority before filtering.
        dq_raw = np.clip(dq_raw, -self.max_tilt, self.max_tilt)

        # First-order low-pass to remove high-frequency content that would
        # otherwise excite the arm structure.
        self._dq_filtered[:] = self.alpha * dq_raw + (1.0 - self.alpha) * self._dq_filtered

        # Joint-space PD with gravity compensation.
        q_target = q_planned + self._dq_filtered
        e = q_target - data.qpos[: self.nu]
        edot = v_planned - data.qvel[: self.nu]
        bias = data.qfrc_bias[: self.nu]
        data.ctrl[:] = self.Kp @ e + self.Kd @ edot + bias

"""MuJoCo wrapper around :class:`ballplate.controllers.pid.PidController`.

Reads ``mujoco.MjData`` into a ``BallState`` (via ``BallStateReader``),
runs the pure-Python PID core against the sampled reference plus a
forward-difference target acceleration, and pushes the resulting
``(ux, uy)`` through ``JointActuator`` (Jacobian, clip, low-pass, joint
PD, gravity compensation) into ``data.ctrl``. The wrapper never calls
``mj_step``: the runner owns the simulator clock so the same code is
reused from the RL training loop without time coupling.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from ballplate.controllers.pid import PidController as _CorePid
from ballplate.controllers.pid import PidGains
from sim.adapters.actuation import JointActuator
from sim.adapters.state import BallStateReader


class PidController:
    """Bundles the pure PID core with the MuJoCo reader and joint actuator."""

    def __init__(
        self,
        model: Any,
        nu: int,
        plate_site_id: int,
        ball_body_id: int,
        ball_gains: PidGains,
        joint_kp: np.ndarray,
        joint_kd: np.ndarray,
        actuation_alpha: float,
        max_tilt: float,
        locked_joints: Sequence[int] = (),
    ):
        self.model = model
        self._pid = _CorePid(ball_gains)
        self._reader = BallStateReader(
            model=model, plate_site_id=plate_site_id, ball_body_id=ball_body_id
        )
        self._actuator = JointActuator(
            model=model,
            plate_site_id=plate_site_id,
            nu=nu,
            kp_joint=np.asarray(joint_kp, dtype=float),
            kd_joint=np.asarray(joint_kd, dtype=float),
            alpha=actuation_alpha,
            max_tilt=max_tilt,
            locked_joints=locked_joints,
        )

    # ----------------------------------------------------------- lifecycle --

    def reset(self) -> None:
        """Clear the PID integrator and the actuator's low-pass filter."""
        self._pid.reset()
        self._actuator.reset()

    # ------------------------------------------------------------------ run --

    def step(self, data, joint_traj, sim_time: float, ball_traj) -> None:
        """Compute and apply one control step. Caller advances the simulator."""
        dt = float(self.model.opt.timestep)

        ball, plate_mat = self._reader.read(data, timestamp=sim_time)

        # Planned arm trajectory at the current sim time. Past `total_time`
        # the planner saturates to the last waypoint by construction.
        joint_t = min(sim_time, joint_traj.total_time - 1e-8)
        q_planned, v_planned, _ = joint_traj.evaluate(joint_t)

        # Reference + finite-difference acceleration for the feed-forward term.
        tx, ty, tvx, tvy = ball_traj.evaluate(sim_time)
        _, _, tvx_next, tvy_next = ball_traj.evaluate(sim_time + dt)
        tax = (tvx_next - tvx) / dt
        tay = (tvy_next - tvy) / dt

        ux, uy = self._pid.step(
            ball=ball,
            target_pos=(tx, ty),
            target_vel=(tvx, tvy),
            target_acc=(tax, tay),
            dt=dt,
        )

        self._actuator.apply(
            data=data,
            ux=ux,
            uy=uy,
            plate_mat=plate_mat,
            q_planned=q_planned,
            v_planned=v_planned,
        )

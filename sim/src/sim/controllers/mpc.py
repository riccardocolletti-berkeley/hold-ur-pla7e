"""MuJoCo wrapper around :class:`ballplate.controllers.mpc.MpcController`.

Mirrors the call surface of :mod:`sim.controllers.pid` so the runner
can swap controllers without touching the surrounding plumbing. The
wrapper reads ``mujoco.MjData`` into a ``BallState``, calls
``step_with_reference`` so the MPC samples the reference along its
horizon (this is the point of MPC over PID for moving targets), and
forwards the resulting ``(ux, uy)`` to the same ``JointActuator`` the
PID uses.

The MPC's planning step ``MpcParams.dt`` is decoupled from the
simulator's integration step ``model.opt.timestep``. The sim ticks at
its own rate; the MPC re-plans every wrapper call and emits the first
control of its horizon (receding-horizon principle), so the wrapper
can run at the sim's native rate without growing the horizon length.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ballplate.controllers.mpc import MpcController as _CoreMpc
from ballplate.controllers.mpc import MpcParams
from ballplate.trajectories import Reference
from sim.adapters.actuation import JointActuator
from sim.adapters.state import BallStateReader


class MpcController:
    """Bundles the pure MPC core with the MuJoCo reader and joint actuator."""

    def __init__(
        self,
        model: Any,
        nu: int,
        plate_site_id: int,
        ball_body_id: int,
        params: MpcParams,
        joint_kp: np.ndarray,
        joint_kd: np.ndarray,
        actuation_alpha: float,
        max_tilt: float,
        locked_joints: Sequence[int] = (),
    ):
        self.model = model
        self._mpc = _CoreMpc(params)
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
        """Reset the actuator's low-pass filter. ``_mpc.reset()`` is a no-op
        for the closed-form MPC, called only for interface parity.
        """
        self._mpc.reset()
        self._actuator.reset()

    # ------------------------------------------------------------------ run --

    def step(self, data, joint_traj, sim_time: float, ball_traj: Reference) -> None:
        """Compute and apply one MPC control step. Caller advances the simulator."""
        ball, plate_mat = self._reader.read(data, timestamp=sim_time)

        # Planned arm trajectory at the current sim time. Past `total_time`
        # the planner saturates to the last waypoint by construction.
        joint_t = min(sim_time, joint_traj.total_time - 1e-8)
        q_planned, v_planned, _ = joint_traj.evaluate(joint_t)

        # Sample the reference at the MPC's planning dt (not the sim dt) so
        # the optimisation sees the upcoming target motion.
        ux, uy = self._mpc.step_with_reference(
            ball=ball,
            reference=ball_traj,
            now=sim_time,
        )

        self._actuator.apply(
            data=data,
            ux=ux,
            uy=uy,
            plate_mat=plate_mat,
            q_planned=q_planned,
            v_planned=v_planned,
        )

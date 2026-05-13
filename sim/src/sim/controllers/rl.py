"""Residual RL controller for MuJoCo.

Top-level import of ``ballplate.controllers.rl`` (and Stable-Baselines 3
via the lazy resolver), so this module is only importable under the
``sim[rl]`` extra. PID-only deployments should keep using
:mod:`sim.controllers.pid`.

The wrapper runs the residual scheme: a PID baseline (the same core
``sim.controllers.pid`` uses) is evaluated every sim substep; the
trained policy is queried only at training rate, held across
``frame_skip`` substeps and blended into a cached correction by a
first-order low-pass on ``command_alpha``; the two contributions are
summed before being clipped by ``omega_clip_rad_s`` and pushed through
the joint actuator. The policy's ``env_meta.json`` overrides
(``actuator_tau_s``, ``max_tilt_rad``, ``omega_clip_rad_s``,
``velocity_clip_mps``, ``command_alpha``, ``policy_hz``,
``lookahead_s``) keep deploy in distribution with training.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from ballplate.controllers.pid import PidController, PidGains
from ballplate.controllers.rl import RLPolicy, build_observation
from ballplate.state import BallState
from sim.adapters.actuation import JointActuator, alpha_from_tau
from sim.adapters.state import BallStateReader


class RLController:
    """Residual controller: PID baseline + RL correction + MuJoCo joint actuator."""

    def __init__(
        self,
        model: Any,
        nu: int,
        plate_site_id: int,
        ball_body_id: int,
        policy: RLPolicy,
        ball_gains: PidGains,
        joint_kp: np.ndarray,
        joint_kd: np.ndarray,
        actuation_alpha: float,
        max_tilt: float,
        locked_joints: Sequence[int] = (),
    ):
        self.model = model
        self._policy = policy
        self._reader = BallStateReader(
            model=model, plate_site_id=plate_site_id, ball_body_id=ball_body_id
        )
        self._pid = PidController(ball_gains)
        if policy.actuator_tau_s is not None:
            actuation_alpha = alpha_from_tau(float(model.opt.timestep), policy.actuator_tau_s)
        self._actuator = JointActuator(
            model=model,
            plate_site_id=plate_site_id,
            nu=nu,
            kp_joint=np.asarray(joint_kp, dtype=float),
            kd_joint=np.asarray(joint_kd, dtype=float),
            alpha=actuation_alpha,
            max_tilt=policy.max_tilt_rad if policy.max_tilt_rad is not None else max_tilt,
            locked_joints=locked_joints,
        )

        # Sim ticks at ``model.opt.timestep``; the policy was trained at
        # ``policy_hz``. Hold each policy query for
        # ``round(1 / (policy_hz * sim_dt))`` substeps so deploy matches the
        # training rate without re-running the network every substep.
        self.frame_skip: int = max(
            1, round(1.0 / (self._policy.policy_hz * float(model.opt.timestep)))
        )
        self._cached_action = np.zeros(2, dtype=float)
        self._hold_count: int = 0
        self._command_alpha = float(np.clip(policy.command_alpha, 0.0, 1.0))

    # ----------------------------------------------------------- lifecycle --

    def reset(self) -> None:
        """Clear the actuator filter, PID integrator, and the cached action."""
        self._actuator.reset()
        self._pid.reset()
        self._cached_action[:] = 0.0
        self._hold_count = 0

    # ------------------------------------------------------------------ run --

    def step(self, data, joint_traj, sim_time: float, ball_traj) -> None:
        """Compute and apply one control step. Caller advances the simulator."""
        ball, plate_mat = self._reader.read(data, timestamp=sim_time)
        if self._policy.velocity_clip_mps is not None:
            limit = self._policy.velocity_clip_mps
            ball = BallState(
                x=ball.x,
                y=ball.y,
                vx=float(np.clip(ball.vx, -limit, limit)),
                vy=float(np.clip(ball.vy, -limit, limit)),
                timestamp=ball.timestamp,
                valid=ball.valid,
            )

        # PID baseline at sim rate, every call; the gains have the same
        # semantics as in ``sim.controllers.pid``.
        sim_dt = float(self.model.opt.timestep)
        tx, ty, tvx, tvy = ball_traj.evaluate(sim_time)
        _, _, tvx_next, tvy_next = ball_traj.evaluate(sim_time + sim_dt)
        tax = (tvx_next - tvx) / sim_dt
        tay = (tvy_next - tvy) / sim_dt
        ux_pid, uy_pid = self._pid.step(
            ball=ball,
            target_pos=(tx, ty),
            target_vel=(tvx, tvy),
            target_acc=(tax, tay),
            dt=sim_dt,
        )

        # Re-query the network only at frame_skip boundaries so the policy's
        # effective rate matches training.
        if self._hold_count == 0:
            tx_la, ty_la, _, _ = ball_traj.evaluate(sim_time + self._policy.lookahead_s)
            zp = plate_mat[:, 2]
            pitch = float(np.arctan2(zp[0], zp[2]))
            roll = float(np.arctan2(zp[1], zp[2]))

            obs = build_observation(
                ball=ball,
                target_now=(tx, ty, tvx, tvy),
                target_lookahead_pos=(tx_la, ty_la),
                plate_pitch=pitch,
                plate_roll=roll,
            )
            correction = np.asarray(self._policy.predict(obs), dtype=float)
            self._cached_action[:] = (
                1.0 - self._command_alpha
            ) * self._cached_action + self._command_alpha * correction

        self._hold_count = (self._hold_count + 1) % self.frame_skip

        joint_t = min(sim_time, joint_traj.total_time - 1e-8)
        q_planned, v_planned, _ = joint_traj.evaluate(joint_t)

        ux = ux_pid + float(self._cached_action[0])
        uy = uy_pid + float(self._cached_action[1])
        if self._policy.omega_clip_rad_s is not None:
            norm = float(np.hypot(ux, uy))
            if norm > self._policy.omega_clip_rad_s:
                scale = self._policy.omega_clip_rad_s / norm
                ux *= scale
                uy *= scale
        self._actuator.apply(
            data=data,
            ux=ux,
            uy=uy,
            plate_mat=plate_mat,
            q_planned=q_planned,
            v_planned=v_planned,
        )

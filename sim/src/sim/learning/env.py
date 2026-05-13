"""Gymnasium environment wrapping the MuJoCo ball-on-plate scene.

Composed entirely from `ballplate` types and `sim` adapters; nothing about
training logic lives here that another platform's RL pipeline could not
reuse:

    * Observation: 12-element float32 vector built by
      `ballplate.controllers.rl.observation.build_observation`.
    * Action: 2-element vector in ``[-1, 1]`` scaled by `action_scale`
      to a plate-frame ``(ux, uy)`` virtual control signal.

The arm is held at the configured home pose for the entire training run.
"""

from collections.abc import Mapping, Sequence
from typing import ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from ballplate.controllers.pid import PidController, PidGains
from ballplate.controllers.rl.observation import OBSERVATION_DIM, build_observation
from ballplate.hardware import HardwareSpec
from ballplate.hardware import default_path as hardware_default_path
from ballplate.hardware import load as load_hardware
from ballplate.state import BallState, PlateGeometry
from ballplate.trajectories import sample as sample_reference
from sim.adapters import BallStateReader, JointActuator
from sim.adapters.actuation import alpha_from_tau
from sim.learning import randomize
from sim.learning.reward import step_reward
from sim.scene import SceneConfig, build_scene


class BallPlateEnv(gym.Env):
    """Single-process Gymnasium env for the ball-on-plate task."""

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        sim_cfg: dict,
        env_cfg: dict,
        dr_ranges: Mapping[str, tuple],
        reward_weights: Mapping[str, float],
        shapes: Sequence[str] = ("stationary", "slow_circle", "circle", "figure8"),
        seed: int | None = None,
        hardware: HardwareSpec | None = None,
    ):
        super().__init__()

        hw = hardware if hardware is not None else load_hardware(hardware_default_path())
        scene_cfg = self._scene_config(hw)
        self.model = build_scene(scene_cfg)
        self.data = mujoco.MjData(self.model)
        self.nu = self.model.nu

        self.plate_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self.plate_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
        self.ball_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball")

        # Plate has multiple geoms (slab + optional decal); pick the largest
        # so domain randomization writes friction onto the contact slab, not
        # the visual decal.
        plate_geoms = [
            g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == self.plate_body_id
        ]
        self.plate_geom_id = max(
            plate_geoms,
            key=lambda g: self.model.geom_size[g, 0] * self.model.geom_size[g, 1],
        )

        self._plate_geometry = PlateGeometry(
            size_x=scene_cfg.plate_size[0],
            size_y=scene_cfg.plate_size[1],
            thickness=scene_cfg.plate_thickness,
        )
        self._ball_radius = scene_cfg.ball_radius

        self._reader = BallStateReader(self.model, self.plate_site_id, self.ball_body_id)
        sim_dt = float(self.model.opt.timestep)
        actuator_alpha = float(env_cfg.get("actuator_alpha", sim_cfg["pid"].get("alpha", 1.0)))
        if "actuator_tau_s" in env_cfg:
            actuator_alpha = alpha_from_tau(sim_dt, float(env_cfg["actuator_tau_s"]))
        self._actuator = JointActuator(
            model=self.model,
            plate_site_id=self.plate_site_id,
            nu=self.nu,
            kp_joint=np.asarray(sim_cfg["gains"]["Kp_joint"], dtype=float),
            kd_joint=np.asarray(sim_cfg["gains"]["Kd_joint"], dtype=float),
            alpha=actuator_alpha,
            max_tilt=float(env_cfg.get("max_tilt_rad", sim_cfg["pid"].get("max_tilt", 0.1))),
        )

        # Residual control: the policy output is added on top of a baseline
        # PID, so the network only has to learn the correction. The PID runs
        # at sim rate inside _apply_action so its gains keep the same meaning
        # they have in sim.controllers.pid.
        pid_cfg = sim_cfg["pid"]
        self._pid = PidController(
            PidGains(
                kp=float(pid_cfg["Kp_ball"]),
                ki=float(pid_cfg["Ki_ball"]),
                kd=float(pid_cfg["Kd_ball"]),
                kff=float(pid_cfg.get("Kff_ball", 0.0)),
                windup_limit=float(pid_cfg["windup_limit"]),
            )
        )

        self.home = np.asarray(sim_cfg["trajectory"]["home"], dtype=float)
        self._zero_v = np.zeros_like(self.home)

        self.policy_dt = 1.0 / float(env_cfg["policy_hz"])
        self.frame_skip = max(1, round(self.policy_dt / float(self.model.opt.timestep)))
        self.episode_steps = int(env_cfg["episode_steps"])
        self.action_scale = float(env_cfg["action_scale"])
        self.command_alpha = float(np.clip(env_cfg.get("command_alpha", 1.0), 0.0, 1.0))
        self.omega_clip = (
            None if "omega_clip_rad_s" not in env_cfg else float(env_cfg["omega_clip_rad_s"])
        )
        self.velocity_clip = (
            None if "velocity_clip_mps" not in env_cfg else float(env_cfg["velocity_clip_mps"])
        )
        # `lookahead_s` may be a scalar or a [t0, t1] list; only the upper bound matters here.
        la = env_cfg["lookahead_s"]
        self.lookahead_s = float(la[-1] if isinstance(la, list | tuple) else la)

        self.dr_ranges = dr_ranges
        self.weights = reward_weights
        self.shapes = list(shapes)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBSERVATION_DIM,), dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self._reset_internal()

    # ----------------------------------------------------------- Construction --

    @staticmethod
    def _scene_config(hw: HardwareSpec) -> SceneConfig:
        return SceneConfig(
            plate_size=hw.plate.size,
            plate_thickness=hw.plate.thickness,
            plate_mass=hw.plate.mass,
            plate_friction=hw.plate.friction,
            ball_radius=hw.ball.radius,
            ball_mass=hw.ball.mass,
            ball_friction=hw.ball.friction,
            adapter_position=hw.adapter.position,
            adapter_orientation=hw.adapter.orientation,
        )

    # ---------------------------------------------------------------- Internal --

    def _reset_internal(self) -> None:
        self.step_count = 0
        self.t = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self._command = np.zeros(2, dtype=np.float32)
        self.prev_pos_err = None

        # Fresh reference for this episode; uses ballplate's pure sampler.
        shape = self.shapes[self.rng.integers(len(self.shapes))]
        self.target_ref = sample_reference(
            shape=shape,
            rng=self.rng,
            plate=self._plate_geometry,
            ball_radius=self._ball_radius,
        )

        # Domain-randomise the model in place.
        self.dr_params = randomize.sample(self.dr_ranges, self.rng)
        randomize.apply(
            self.model,
            self.data,
            self.dr_params,
            self.ball_geom_id,
            self.plate_geom_id,
            self.ball_body_id,
        )

        # Action latency: zero-length buffer if the sampled delay is zero.
        # `round()` keeps sub-step samples on the closest discrete step; a
        # plain `int()` truncation silently rounded everything below one
        # full step down to zero delay.
        self.action_delay_steps = round(self.dr_params["action_delay_ms"] / 1000.0 / self.policy_dt)
        self.action_buffer = [np.zeros(2, dtype=np.float32)] * self.action_delay_steps

        self._actuator.reset()
        self._pid.reset()

    def _place_ball(self) -> None:
        """Reseat the ball at a randomised offset on the plate centre."""
        plate_pos = self.data.site_xpos[self.plate_site_id]
        rad = self.dr_params["ball_init_radius"]
        ang = self.rng.uniform(0.0, 2.0 * np.pi)
        offset = rad * np.array([np.cos(ang), np.sin(ang), 0.0])
        bq = self.nu
        self.data.qpos[bq : bq + 3] = plate_pos + offset + np.array([0.0, 0.0, 0.015])
        self.data.qpos[bq + 3] = 1.0
        self.data.qpos[bq + 4 : bq + 7] = 0.0
        self.data.qvel[bq : bq + 6] = 0.0

    def _read_obs(self) -> np.ndarray:
        ball, plate_mat = self._reader.read(self.data, timestamp=self.t)

        x, y = ball.x, ball.y
        vx, vy = ball.vx, ball.vy

        # Optional Gaussian noise on the position component of the ball state.
        sigma = self.dr_params["obs_noise_std"]
        if sigma > 0:
            x += self.rng.normal(0.0, sigma)
            y += self.rng.normal(0.0, sigma)

        if self.velocity_clip is not None:
            vx = float(np.clip(vx, -self.velocity_clip, self.velocity_clip))
            vy = float(np.clip(vy, -self.velocity_clip, self.velocity_clip))

        if sigma > 0 or self.velocity_clip is not None:
            ball = BallState(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                timestamp=ball.timestamp,
                valid=ball.valid,
            )

        tx, ty, tvx, tvy = self.target_ref.evaluate(self.t)
        tx_la, ty_la, _, _ = self.target_ref.evaluate(self.t + self.lookahead_s)

        zp = plate_mat[:, 2]
        pitch = float(np.arctan2(zp[0], zp[2]))
        roll = float(np.arctan2(zp[1], zp[2]))

        return build_observation(
            ball=ball,
            target_now=(tx, ty, tvx, tvy),
            target_lookahead_pos=(tx_la, ty_la),
            plate_pitch=pitch,
            plate_roll=roll,
        )

    def _has_dropped(self) -> bool:
        ball_z = float(self.data.xpos[self.ball_body_id, 2])
        plate_z = float(self.data.site_xpos[self.plate_site_id, 2])
        return ball_z < plate_z - 0.05

    def _apply_action(self, action: np.ndarray) -> None:
        # PID baseline runs at sim rate (called every substep) so the gains
        # tuned in sim.yaml behave the same as in sim.controllers.pid.
        sim_dt = float(self.model.opt.timestep)
        ball_pid, _ = self._reader.read(self.data, timestamp=self.t)
        tx, ty, tvx, tvy = self.target_ref.evaluate(self.t)
        _, _, tvx_next, tvy_next = self.target_ref.evaluate(self.t + sim_dt)
        tax = (tvx_next - tvx) / sim_dt
        tay = (tvy_next - tvy) / sim_dt
        ux_pid, uy_pid = self._pid.step(
            ball=ball_pid,
            target_pos=(tx, ty),
            target_vel=(tvx, tvy),
            target_acc=(tax, tay),
            dt=sim_dt,
        )
        ux = ux_pid + float(action[0]) * self.action_scale
        uy = uy_pid + float(action[1]) * self.action_scale
        if self.omega_clip is not None:
            norm = float(np.hypot(ux, uy))
            if norm > self.omega_clip:
                scale = self.omega_clip / norm
                ux *= scale
                uy *= scale
        plate_mat = self.data.site_xmat[self.plate_site_id].reshape(3, 3)
        self._actuator.apply(
            data=self.data,
            ux=ux,
            uy=uy,
            plate_mat=plate_mat,
            q_planned=self.home,
            v_planned=self._zero_v,
        )

    def set_shapes(self, shapes: Sequence[str]) -> None:
        """Curriculum hook: switch the active shape list at the next reset."""
        self.shapes = list(shapes)

    # ------------------------------------------------------------------ Gym --

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        # DR (and the mj_setConst it triggers when mass changes) must run
        # before we lay down the home pose: mj_setConst internally resets
        # qpos to qpos0, so any pose written before it would be wiped out.
        self._reset_internal()
        self.data.qpos[: self.nu] = self.home
        mujoco.mj_forward(self.model, self.data)
        self._place_ball()
        mujoco.mj_forward(self.model, self.data)
        return self._read_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        if self.action_delay_steps == 0:
            delayed = action
        else:
            self.action_buffer.append(action)
            delayed = self.action_buffer.pop(0)

        self._command = (
            (1.0 - self.command_alpha) * self._command + self.command_alpha * delayed
        ).astype(np.float32)
        applied = self._command.copy()

        for _ in range(self.frame_skip):
            self._apply_action(applied)
            mujoco.mj_step(self.model, self.data)
        self.t += self.policy_dt
        self.step_count += 1

        obs = self._read_obs()
        dropped = self._has_dropped()

        ball_pos = (float(obs[0]), float(obs[1]))
        ball_vel = (float(obs[2]), float(obs[3]))
        tgt_pos = (float(obs[4]), float(obs[5]))
        tgt_vel = (float(obs[6]), float(obs[7]))

        steps_remaining_frac = max(0.0, 1.0 - self.step_count / self.episode_steps)
        reward, pos_err = step_reward(
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            target_pos=tgt_pos,
            target_vel=tgt_vel,
            action=applied,
            prev_action=self.prev_action,
            prev_pos_err=self.prev_pos_err,
            dropped=dropped,
            steps_remaining_frac=steps_remaining_frac,
            weights=self.weights,
        )
        self.prev_action = applied
        self.prev_pos_err = pos_err

        terminated = bool(dropped)
        truncated = self.step_count >= self.episode_steps
        info = {"target": tgt_pos, "dr": self.dr_params}
        return obs, float(reward), terminated, truncated, info

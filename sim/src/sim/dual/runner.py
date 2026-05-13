"""Side-by-side dual-arm viewer: PID (left) vs PID+RL (right).

Runs both controllers on the same simulator clock so the visual difference
between the two ball traces is unambiguously the controller, not the random
seed or the timing. Arm A receives PID-only commands; arm B receives PID +
trained-policy correction. Same target reference, same home pose, same
domain (no DR: this is a visual demo, not a benchmark).

Typical invocation::

    uv run mjpython -m sim.dual --rl-run-dir sim/policies/ppo_residual_v2 \\
        --shape circle --radius 0.07 --period 8

The ``--shape`` / ``--radius`` / ``--period`` / ``--drawn-file`` flags
mirror ``sim.runner`` so the same trajectories can be inspected in either.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from ballplate.control import default_path as control_default_path
from ballplate.control import load as load_control
from ballplate.control import parse_lock_spec
from ballplate.controllers.pid import PidController, PidGains
from ballplate.controllers.rl import RLPolicy, build_observation
from ballplate.hardware import HardwareSpec
from ballplate.hardware import default_path as hardware_default_path
from ballplate.hardware import load as load_hardware
from ballplate.trajectories import (
    Circle,
    DrawnPath,
    Figure8,
    Reference,
    SmoothApproach,
    Stationary,
)
from sim.adapters import BallStateReader
from sim.adapters.actuation import alpha_from_tau
from sim.arm import JointTrajectory, freeze
from sim.benchmark.domains import DOMAINS
from sim.dual.scene import ARM_B_SUFFIX, DualSceneConfig, build_dual_scene
from sim.scene import SceneConfig
from sim.viz import BallTrail, TargetOverlay

# Colour palette: same blue/gold as the benchmark plots so figures across
# the report tell a consistent visual story.
_PID_RGBA = (0.20, 0.55, 0.85, 1.0)  # blue: PID-only arm
_RESIDUAL_RGBA = (0.99, 0.71, 0.08, 1.0)  # gold: PID + RL arm
_TARGET_RGB = (1.0, 0.85, 0.0)  # yellow: shared reference path


def _add_trail_to_scene(trail: BallTrail, scn: Any, data: Any) -> None:
    """Append a BallTrail's fading dots without resetting ``scn.ngeom``.

    Mirrors :meth:`BallTrail.render` minus the ``scn.ngeom = 0`` line so the
    dual viewer can stack two trails plus the overlays in one frame instead
    of having each trail clobber the previous one.
    """
    points = trail._points
    if not points:
        return
    rot = data.xmat[trail.plate_body_id].reshape(3, 3)
    plate_pos = data.xpos[trail.plate_body_id]
    now = points[-1][1]
    identity = np.eye(3).flatten()
    size = np.array([trail.point_size, 0.0, 0.0])
    for local, t in points:
        if scn.ngeom >= scn.maxgeom:
            break
        alpha = 1.0 - (now - t) / trail.fade_seconds
        if alpha <= 0.0:
            continue
        world_pos = plate_pos + rot @ local
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=size,
            pos=world_pos,
            mat=identity,
            rgba=np.array([*trail.color, alpha], dtype=np.float32),
        )
        scn.ngeom += 1


# Canonical UR joint names; the suffix appended for arm B is added at lookup.
_UR_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


# ============================================================================
# Per-arm actuator: same pipeline as sim.adapters.JointActuator, but slices
# the model's qpos / qvel / ctrl to only the indices belonging to this arm.
# ============================================================================


class DualArmActuator:
    """Apply a 2-D virtual control signal to one arm in the dual model."""

    def __init__(
        self,
        model: Any,
        plate_site_id: int,
        qpos_idx: list[int],
        qvel_idx: list[int],
        ctrl_idx: list[int],
        kp_joint: np.ndarray,
        kd_joint: np.ndarray,
        alpha: float,
        max_tilt: float,
        locked_joints: list[int],
    ) -> None:
        self.model = model
        self.plate_site_id = int(plate_site_id)
        self.qpos_idx = np.asarray(qpos_idx, dtype=int)
        self.qvel_idx = np.asarray(qvel_idx, dtype=int)
        self.ctrl_idx = np.asarray(ctrl_idx, dtype=int)
        self.Kp = np.diag(np.asarray(kp_joint, dtype=float))
        self.Kd = np.diag(np.asarray(kd_joint, dtype=float))
        self.alpha = float(alpha)
        self.max_tilt = float(max_tilt)
        self.n_joints = int(self.ctrl_idx.size)
        self._dq_filtered = np.zeros(self.n_joints, dtype=float)
        self._free_cols = np.setdiff1d(
            np.arange(self.n_joints),
            np.asarray(locked_joints, dtype=int),
            assume_unique=False,
        )

    def reset(self) -> None:
        self._dq_filtered[:] = 0.0

    def apply(
        self,
        data: Any,
        ux: float,
        uy: float,
        plate_mat: np.ndarray,
        q_planned: np.ndarray,
        v_planned: np.ndarray,
    ) -> None:
        x_hat = plate_mat[:, 0]
        y_hat = plate_mat[:, 1]
        omega_des = -(ux * y_hat - uy * x_hat)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.plate_site_id)
        Jr_arm = jacr[:, self.qvel_idx]

        dq_raw = np.zeros(self.n_joints, dtype=float)
        if self._free_cols.size:
            dq_raw[self._free_cols] = np.linalg.lstsq(
                Jr_arm[:, self._free_cols],
                omega_des,
                rcond=None,
            )[0]
        dq_raw = np.clip(dq_raw, -self.max_tilt, self.max_tilt)
        self._dq_filtered[:] = self.alpha * dq_raw + (1.0 - self.alpha) * self._dq_filtered

        q_target = q_planned + self._dq_filtered
        e = q_target - data.qpos[self.qpos_idx]
        edot = v_planned - data.qvel[self.qvel_idx]
        bias = data.qfrc_bias[self.qvel_idx]
        data.ctrl[self.ctrl_idx] = self.Kp @ e + self.Kd @ edot + bias


# ============================================================================
# Per-arm controller: residual PID + optional RL correction. The RL part is
# active when ``rl_policy`` is non-None; otherwise the arm runs pure PID.
# ============================================================================


class DualArmController:
    """Drive one arm of the dual model with PID (and optional RL residual)."""

    def __init__(
        self,
        model: Any,
        plate_site_id: int,
        ball_body_id: int,
        ball_gains: PidGains,
        actuator: DualArmActuator,
        rl_policy: RLPolicy | None = None,
    ) -> None:
        self.model = model
        self._reader = BallStateReader(
            model=model,
            plate_site_id=plate_site_id,
            ball_body_id=ball_body_id,
        )
        self._pid = PidController(ball_gains)
        self._actuator = actuator
        self._policy = rl_policy
        # Cache for the network correction so the residual rate matches
        # what the policy was trained on (1 / policy_hz).
        if rl_policy is not None:
            self.frame_skip = max(
                1,
                round(1.0 / (rl_policy.policy_hz * float(model.opt.timestep))),
            )
            self._cached_correction = np.zeros(2, dtype=float)
            self._command_alpha = float(np.clip(rl_policy.command_alpha, 0.0, 1.0))
            self._lookahead_s = float(rl_policy.lookahead_s)
            self._action_scale = float(rl_policy.action_scale)
            self._hold_count = 0
        else:
            self.frame_skip = 1
            self._cached_correction = np.zeros(2, dtype=float)
            self._command_alpha = 1.0
            self._lookahead_s = 0.0
            self._action_scale = 0.0
            self._hold_count = 0

    def reset(self) -> None:
        self._pid.reset()
        self._actuator.reset()
        self._cached_correction[:] = 0.0
        self._hold_count = 0

    def step(
        self,
        data: Any,
        joint_traj: JointTrajectory,
        sim_time: float,
        ball_traj: Reference,
    ) -> None:
        sim_dt = float(self.model.opt.timestep)
        ball, plate_mat = self._reader.read(data, timestamp=sim_time)

        # PID baseline (sim rate).
        tx, ty, tvx, tvy = ball_traj.evaluate(sim_time)
        _, _, tvx_n, tvy_n = ball_traj.evaluate(sim_time + sim_dt)
        tax = (tvx_n - tvx) / sim_dt
        tay = (tvy_n - tvy) / sim_dt
        ux_pid, uy_pid = self._pid.step(
            ball=ball,
            target_pos=(tx, ty),
            target_vel=(tvx, tvy),
            target_acc=(tax, tay),
            dt=sim_dt,
        )

        # Optional RL correction (policy rate via frame_skip).
        if self._policy is not None and self._hold_count == 0:
            tx_la, ty_la, _, _ = ball_traj.evaluate(sim_time + self._lookahead_s)
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
            self._cached_correction[:] = (
                1.0 - self._command_alpha
            ) * self._cached_correction + self._command_alpha * correction
        if self._policy is not None:
            self._hold_count = (self._hold_count + 1) % self.frame_skip
            ux = ux_pid + float(self._cached_correction[0])
            uy = uy_pid + float(self._cached_correction[1])
        else:
            ux, uy = ux_pid, uy_pid

        joint_t = min(sim_time, joint_traj.total_time - 1e-8)
        q_planned, v_planned, _ = joint_traj.evaluate(joint_t)
        # The actuator owns the slice indices; q_planned and v_planned are the
        # 6-element home/zero vectors (one per arm joint).
        self._actuator.apply(data, ux, uy, plate_mat, q_planned, v_planned)


# ============================================================================
# Helpers: index lookup, scene config, reference trajectory
# ============================================================================


def _arm_indices(
    model: Any,
    suffix: str = "",
) -> tuple[list[int], list[int], list[int], int, int, int]:
    """Return ``(qpos_idx, qvel_idx, ctrl_idx, plate_site_id, plate_body_id, ball_body_id)``."""
    joint_ids: list[int] = []
    for name in _UR_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name + suffix)
        if jid < 0:
            raise RuntimeError(
                f"Joint '{name + suffix}' not found in dual model: "
                "scene composition probably renamed something inconsistently."
            )
        joint_ids.append(jid)

    qpos_idx = [int(model.jnt_qposadr[j]) for j in joint_ids]
    qvel_idx = [int(model.jnt_dofadr[j]) for j in joint_ids]

    # Find the actuator that drives each joint by checking actuator_trnid[a, 0].
    ctrl_idx: list[int] = []
    for jid in joint_ids:
        match = None
        for a in range(model.nu):
            if int(model.actuator_trnid[a, 0]) == jid:
                match = a
                break
        if match is None:
            raise RuntimeError(f"No actuator drives joint id {jid}.")
        ctrl_idx.append(match)

    plate_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "plate_center" + suffix,
    )
    plate_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "plate" + suffix,
    )
    ball_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "ball" + suffix,
    )
    if plate_site_id < 0 or plate_body_id < 0 or ball_body_id < 0:
        raise RuntimeError(
            f"Missing plate_center{suffix!r}, plate{suffix!r} or ball{suffix!r} " "in dual model."
        )
    return qpos_idx, qvel_idx, ctrl_idx, plate_site_id, plate_body_id, ball_body_id


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


def _build_reference(cfg: dict, args: argparse.Namespace) -> Reference:
    """Map ``ball_trajectory.shape`` (and CLI overrides) to a `Reference`."""
    bt = dict(cfg.get("ball_trajectory", {}))
    if args.shape:
        bt["shape"] = args.shape
    if args.drawn_file:
        bt["shape"] = "drawn"
        bt["file"] = args.drawn_file
    if args.radius is not None:
        bt["radius"] = args.radius
    if args.period is not None:
        bt["period"] = args.period

    shape = bt.get("shape", "circle")
    if shape in ("stationary", "center"):
        return Stationary(x=float(bt.get("hold_x", 0.0)), y=float(bt.get("hold_y", 0.0)))
    if shape == "circle":
        return Circle(radius=float(bt["radius"]), period=float(bt["period"]))
    if shape == "figure8":
        r = float(bt["radius"])
        return Figure8(rx=r, ry=r / 2.0, period=float(bt["period"]))
    if shape == "drawn":
        if "file" not in bt:
            raise SystemExit("--drawn-file is required when shape='drawn'")
        return DrawnPath.from_file(bt["file"])
    raise SystemExit(f"Unknown ball trajectory shape: {shape!r}")


def _is_dropped(
    data: Any,
    ball_body_id: int,
    plate_site_id: int,
    threshold: float = 0.05,
) -> bool:
    """True when the ball is more than ``threshold`` metres below its plate."""
    return bool(data.xpos[ball_body_id, 2] < data.site_xpos[plate_site_id, 2] - threshold)


def _place_ball(
    data: Any,
    model: Any,
    plate_site_id: int,
    ball_body_id: int,
    settle_height_m: float = 0.05,
) -> None:
    """Reseat one of the two balls on top of its plate centre.

    Looks up the ball's free-joint qpos start via ``jnt_qposadr`` so the
    indexing works regardless of the order MuJoCo numbered the joints.
    """
    plate_pos = data.site_xpos[plate_site_id]
    free_jnt = -1
    for j in range(model.njnt):
        if (
            int(model.jnt_bodyid[j]) == ball_body_id
            and int(model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE
        ):
            free_jnt = j
            break
    if free_jnt < 0:
        raise RuntimeError(f"Ball body {ball_body_id} has no free joint.")
    qpos_start = int(model.jnt_qposadr[free_jnt])
    qvel_start = int(model.jnt_dofadr[free_jnt])
    data.qpos[qpos_start : qpos_start + 3] = plate_pos + np.array([0.0, 0.0, settle_height_m])
    data.qpos[qpos_start + 3] = 1.0
    data.qpos[qpos_start + 4 : qpos_start + 7] = 0.0
    data.qvel[qvel_start : qvel_start + 6] = 0.0


# ============================================================================
# CLI + main loop
# ============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sim.dual")
    parser.add_argument(
        "--config",
        default="sim/config/sim.yaml",
        help="Path to the sim YAML config (relative to repo root).",
    )
    parser.add_argument(
        "--rl-run-dir", required=True, help="Trained PPO run dir for the right-hand arm."
    )
    parser.add_argument(
        "--shape", default="circle", help="stationary | circle | figure8 | drawn (default circle)."
    )
    parser.add_argument(
        "--drawn-file", default=None, help="Path to a drawn-trajectory JSON; implies --shape drawn."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Override ball_trajectory.radius (built-in shapes only).",
    )
    parser.add_argument(
        "--period", type=float, default=None, help="Override ball_trajectory.period."
    )
    parser.add_argument(
        "--lock-joints",
        default="config",
        help="Freeze joints (per arm): 'config' | 'all' | 'none' | '0,1,2'.",
    )
    parser.add_argument(
        "--offset", type=float, default=0.6, help="Half of the inter-arm spacing along x [m]."
    )
    parser.add_argument(
        "--settle-time", type=float, default=2.0, help="Smooth-approach blend window [s]."
    )
    parser.add_argument(
        "--domain",
        default="standard",
        choices=tuple(DOMAINS),
        help="Named scenario from sim.benchmark.domains "
        "(matches the offline benchmark settings).",
    )
    # Per-knob overrides applied AFTER the domain so you can crank a single
    # variable past any of the canned settings.
    parser.add_argument(
        "--ball-friction", type=float, default=None, help="Override ball-geom sliding friction (μ)."
    )
    parser.add_argument(
        "--plate-friction",
        type=float,
        default=None,
        help="Override plate-geom sliding friction (μ).",
    )
    parser.add_argument(
        "--ball-mass",
        type=float,
        default=None,
        help="Override ball mass [kg]; rescales inertia + mj_setConst.",
    )
    parser.add_argument(
        "--gravity-scale",
        type=float,
        default=None,
        help="Multiplier on Earth gravity along world z.",
    )
    return parser.parse_args()


def _plate_geom_id(model: Any, plate_body_id: int) -> int:
    """Return the largest geom in the plate body: that's the contact slab."""
    geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == plate_body_id]
    if not geoms:
        raise RuntimeError(f"Plate body {plate_body_id} has no geoms.")
    return max(geoms, key=lambda g: model.geom_size[g, 0] * model.geom_size[g, 1])


def _apply_domain_to_dual(
    model: Any,
    data: Any,
    domain_name: str,
    ball_a: int,
    ball_b: int,
    plate_body_a: int,
    plate_body_b: int,
) -> None:
    """Apply a benchmark domain's DR settings symmetrically to both arms.

    Each ``dr_ranges`` entry is collapsed to its midpoint (the benchmark
    domains are point ranges anyway: ``[v, v]`` for the isolation tests).
    Mirrors ``sim.learning.randomize.apply`` but does it once per arm so
    both plates and balls share the same physical conditions.
    """
    domain = DOMAINS[domain_name]
    point_values = {key: 0.5 * (lo + hi) for key, (lo, hi) in domain.dr_ranges.items()}
    if "ball_mass" in point_values:
        new_mass = float(point_values["ball_mass"])
        for bid in (ball_a, ball_b):
            old = float(model.body_mass[bid])
            if old > 0.0:
                model.body_inertia[bid] *= new_mass / old
            model.body_mass[bid] = new_mass
        mujoco.mj_setConst(model, data)
    if "ball_friction" in point_values:
        for bid in (ball_a, ball_b):
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) == bid:
                    model.geom_friction[g, 0] = float(point_values["ball_friction"])
    if "plate_friction" in point_values:
        for pbid in (plate_body_a, plate_body_b):
            gid = _plate_geom_id(model, pbid)
            model.geom_friction[gid, 0] = float(point_values["plate_friction"])
    if "gravity_scale" in point_values:
        model.opt.gravity[2] = -9.81 * float(point_values["gravity_scale"])


def main() -> None:
    args = _parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hw = load_hardware(hardware_default_path())
    dual_cfg = DualSceneConfig(
        base=_scene_config(hw),
        base_offset_x=float(args.offset),
    )
    model = build_dual_scene(dual_cfg)
    data = mujoco.MjData(model)

    qpos_a, qvel_a, ctrl_a, plate_site_a, plate_body_a, ball_a = _arm_indices(model, suffix="")
    qpos_b, qvel_b, ctrl_b, plate_site_b, plate_body_b, ball_b = _arm_indices(
        model, suffix=ARM_B_SUFFIX
    )

    # Recolour the two ball geoms so left vs right is unmistakable in the
    # viewer; everything else (trail dots, log line) reuses the same palette.
    for geom_name, rgba in (("ball", _PID_RGBA), ("ball" + ARM_B_SUFFIX, _RESIDUAL_RGBA)):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid >= 0:
            model.geom_rgba[gid] = rgba

    # Joint-locking spec applied identically to both arms.
    control = load_control(control_default_path())
    home = np.asarray(cfg["trajectory"]["home"], dtype=float)
    locked = parse_lock_spec(
        args.lock_joints,
        n_joints=len(home),
        config_indices=control.locked_joints,
    )

    # Apply the chosen domain's physical perturbation symmetrically to both
    # arms BEFORE building the actuators, so any mass-derived constants
    # (mj_setConst) and gravity changes are in effect from step 0.
    _apply_domain_to_dual(
        model,
        data,
        args.domain,
        ball_a=ball_a,
        ball_b=ball_b,
        plate_body_a=plate_body_a,
        plate_body_b=plate_body_b,
    )

    # Per-knob CLI overrides on top of the domain. Applied second so they win
    # over the domain's value: useful for "crank one variable to the moon"
    # demos.
    if args.ball_friction is not None:
        for bid in (ball_a, ball_b):
            for ggid in range(model.ngeom):
                if int(model.geom_bodyid[ggid]) == bid:
                    model.geom_friction[ggid, 0] = float(args.ball_friction)
    if args.plate_friction is not None:
        for pbid in (plate_body_a, plate_body_b):
            gid = _plate_geom_id(model, pbid)
            model.geom_friction[gid, 0] = float(args.plate_friction)
    if args.ball_mass is not None:
        new_mass = float(args.ball_mass)
        for bid in (ball_a, ball_b):
            old = float(model.body_mass[bid])
            if old > 0.0:
                model.body_inertia[bid] *= new_mass / old
            model.body_mass[bid] = new_mass
        mujoco.mj_setConst(model, data)
    if args.gravity_scale is not None:
        model.opt.gravity[2] = -9.81 * float(args.gravity_scale)

    # Per-arm actuator. Both arms reuse the same joint-level gains from sim.yaml.
    # Override actuator alpha and max_tilt from the domain's env_cfg so the
    # plate stack matches what the policy was trained on for that scenario.
    p = cfg["pid"]
    g = cfg["gains"]
    ball_gains = PidGains(
        kp=float(p["Kp_ball"]),
        ki=float(p["Ki_ball"]),
        kd=float(p["Kd_ball"]),
        kff=float(p.get("Kff_ball", 0.0)),
        windup_limit=float(p["windup_limit"]),
    )
    domain_env_cfg = DOMAINS[args.domain].env_cfg
    sim_dt = float(model.opt.timestep)
    if "actuator_tau_s" in domain_env_cfg:
        actuator_alpha = alpha_from_tau(sim_dt, float(domain_env_cfg["actuator_tau_s"]))
    else:
        actuator_alpha = float(p["alpha"])
    actuator_max_tilt = float(domain_env_cfg.get("max_tilt_rad", p["max_tilt"]))
    actuator_a = DualArmActuator(
        model=model,
        plate_site_id=plate_site_a,
        qpos_idx=qpos_a,
        qvel_idx=qvel_a,
        ctrl_idx=ctrl_a,
        kp_joint=g["Kp_joint"],
        kd_joint=g["Kd_joint"],
        alpha=actuator_alpha,
        max_tilt=actuator_max_tilt,
        locked_joints=locked.tolist(),
    )
    actuator_b = DualArmActuator(
        model=model,
        plate_site_id=plate_site_b,
        qpos_idx=qpos_b,
        qvel_idx=qvel_b,
        ctrl_idx=ctrl_b,
        kp_joint=g["Kp_joint"],
        kd_joint=g["Kd_joint"],
        alpha=actuator_alpha,
        max_tilt=actuator_max_tilt,
        locked_joints=locked.tolist(),
    )

    # Arm A: PID only. Arm B: PID + RL residual.
    rl_policy = RLPolicy(Path(args.rl_run_dir))
    ctrl_arm_a = DualArmController(
        model=model,
        plate_site_id=plate_site_a,
        ball_body_id=ball_a,
        ball_gains=ball_gains,
        actuator=actuator_a,
        rl_policy=None,
    )
    ctrl_arm_b = DualArmController(
        model=model,
        plate_site_id=plate_site_b,
        ball_body_id=ball_b,
        ball_gains=ball_gains,
        actuator=actuator_b,
        rl_policy=rl_policy,
    )

    # Initial pose: both arms at home, balls reseated above their plates.
    data.qpos[qpos_a] = home
    data.qpos[qpos_b] = home
    mujoco.mj_forward(model, data)
    _place_ball(data, model, plate_site_a, ball_a)
    _place_ball(data, model, plate_site_b, ball_b)
    mujoco.mj_forward(model, data)

    # Joint-space waypoint trajectory shared between the two arms (same home,
    # same dest_offset). With the default lock spec (upper arm frozen) this is
    # essentially a hold pattern, which is what the demo wants.
    t = cfg["trajectory"]
    dest = freeze(home + np.asarray(t["dest_offset"], dtype=float), home, locked)
    mid = freeze(home + np.asarray(t["mid_offset"], dtype=float), home, locked)
    waypoints = [home, home, mid, dest, dest, mid, home]
    joint_traj = JointTrajectory(waypoints, t["durations"])

    # Reference: smooth-approach from each ball's current plate-frame position.
    inner_ref = _build_reference(cfg, args)
    reader_a = BallStateReader(model, plate_site_a, ball_a)
    reader_b = BallStateReader(model, plate_site_b, ball_b)
    ball0_a, _ = reader_a.read(data, timestamp=0.0)
    ball0_b, _ = reader_b.read(data, timestamp=0.0)
    ref_a = SmoothApproach(
        inner=inner_ref, start_x=ball0_a.x, start_y=ball0_a.y, blend_time=float(args.settle_time)
    )
    ref_b = SmoothApproach(
        inner=inner_ref, start_x=ball0_b.x, start_y=ball0_b.y, blend_time=float(args.settle_time)
    )

    # Per-arm visual aids: a coloured ball trail and the static target overlay.
    half_thick = hw.plate.thickness / 2.0
    trail_a = BallTrail(ball_a, plate_body_a, half_thick, color=_PID_RGBA[:3])
    trail_b = BallTrail(ball_b, plate_body_b, half_thick, color=_RESIDUAL_RGBA[:3])
    overlay_a = TargetOverlay(
        reference=inner_ref,
        plate_body_id=plate_body_a,
        plate_half_thickness=half_thick,
        period=float(inner_ref.period),
        color=_TARGET_RGB,
    )
    overlay_b = TargetOverlay(
        reference=inner_ref,
        plate_body_id=plate_body_b,
        plate_half_thickness=half_thick,
        period=float(inner_ref.period),
        color=_TARGET_RGB,
    )

    render_fps = float(cfg.get("render_fps", 60))
    sim_dt = float(model.opt.timestep)
    steps_per_frame = max(1, round(1.0 / (render_fps * sim_dt)))

    print(
        f"Dual demo: PID on left arm (x={-dual_cfg.base_offset_x:+.2f}), "
        f"PID+RL on right (x={+dual_cfg.base_offset_x:+.2f}). "
        f"shape={args.shape}, domain={args.domain}, lock={args.lock_joints}."
    )
    print(f"  domain: {DOMAINS[args.domain].description}")
    if locked.size:
        print(f"Locked joints (both arms): {locked.tolist()}")

    arms = (
        ("PID", ball_a, plate_site_a, ctrl_arm_a, trail_a),
        ("PID+RL", ball_b, plate_site_b, ctrl_arm_b, trail_b),
    )
    # Per-arm cooldown after a respawn: if the contact solver is destabilised
    # (e.g. friction > ~2 in MuJoCo), the dropped check could fire again on
    # the next sim step before the ball has resettled, producing a respawn
    # loop at sim rate. The cooldown gives the contact pair half a second to
    # resolve before we are willing to declare the ball dropped again.
    respawn_lockout_until = {"PID": 0.0, "PID+RL": 0.0}
    _RESPAWN_COOLDOWN_S = 0.5

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_time = 0.0
        while viewer.is_running():
            frame_start = time.monotonic()
            for _ in range(steps_per_frame):
                ctrl_arm_a.step(data, joint_traj, sim_time, ref_a)
                ctrl_arm_b.step(data, joint_traj, sim_time, ref_b)
                mujoco.mj_step(model, data)
                sim_time += sim_dt

            # Per-arm drop recovery: respawn the dropped ball on its plate and
            # zero the controller's internal state so the integrator does not
            # carry stale error from the failed run.
            for label, ball_bid, plate_sid, ctrl, trail in arms:
                if sim_time < respawn_lockout_until[label]:
                    continue
                if _is_dropped(data, ball_bid, plate_sid):
                    print(f"[{label}] ball dropped at t={sim_time:.2f}s: respawning")
                    _place_ball(data, model, plate_sid, ball_bid)
                    mujoco.mj_forward(model, data)
                    ctrl.reset()
                    trail.clear()
                    respawn_lockout_until[label] = sim_time + _RESPAWN_COOLDOWN_S

            # Visual aids: refresh both trails, then composit them with the
            # static target overlays into a single user_scn so neither side
            # clobbers the other.
            trail_a.record(data)
            trail_b.record(data)
            viewer.user_scn.ngeom = 0
            _add_trail_to_scene(trail_a, viewer.user_scn, data)
            _add_trail_to_scene(trail_b, viewer.user_scn, data)
            overlay_a.add_to_scene(viewer.user_scn, data)
            overlay_b.add_to_scene(viewer.user_scn, data)

            viewer.sync()
            sleep = (1.0 / render_fps) - (time.monotonic() - frame_start)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()

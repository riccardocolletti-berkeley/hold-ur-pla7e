"""Sim runner: parse args, build the world, run the viewer loop.

Wires every other module in `sim/` together. The runner owns the simulator
clock: it is the only place that calls `mujoco.mj_step`. Controllers
write to ``data.ctrl``; the runner advances the world.

Typical invocations::

    mjpython -m sim.runner --controller pid --shape circle
    mjpython -m sim.runner --controller pid --drawn-file path/to/draw.json
    mjpython -m sim.runner --controller rl  --rl-run-dir runs/ppo_dr_v1
"""

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
from ballplate.controllers.mpc import MpcParams
from ballplate.controllers.pid import PidGains
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
from sim.adapters import BallDropMonitor, BallStateReader
from sim.arm import JointTrajectory, freeze, parse_lock_spec
from sim.controllers import MpcController, PidController
from sim.scene import SceneConfig, build_scene
from sim.viz import BallTrail, TargetOverlay

# ============================================================================
# Config helpers
# ============================================================================


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _scene_config(hw: HardwareSpec) -> SceneConfig:
    """Build the MuJoCo scene parameters from the shared hardware spec."""
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
    if args.hold_x is not None:
        bt["hold_x"] = args.hold_x
    if args.hold_y is not None:
        bt["hold_y"] = args.hold_y

    shape = bt.get("shape", "stationary")
    if shape in ("stationary", "center"):
        return Stationary(
            x=float(bt.get("hold_x", 0.0)),
            y=float(bt.get("hold_y", 0.0)),
        )
    if shape == "circle":
        return Circle(radius=float(bt["radius"]), period=float(bt["period"]))
    if shape == "figure8":
        # Map a single `radius` to the 2:1 axis ratio used by the figure-8
        # trajectory generator.
        r = float(bt["radius"])
        return Figure8(rx=r, ry=r / 2.0, period=float(bt["period"]))
    if shape == "drawn":
        if "file" not in bt:
            raise SystemExit("ball_trajectory.file required when shape='drawn'")
        return DrawnPath.from_file(bt["file"])
    raise SystemExit(f"Unknown ball trajectory shape: {shape!r}")


def _build_joint_trajectory(
    cfg: dict, lock_spec: str
) -> tuple[JointTrajectory, np.ndarray, np.ndarray]:
    t = cfg["trajectory"]
    home = np.asarray(t["home"], dtype=float)
    control = load_control(control_default_path())
    locked = parse_lock_spec(
        lock_spec,
        n_joints=len(home),
        config_indices=control.locked_joints,
    )
    dest = freeze(home + np.asarray(t["dest_offset"], dtype=float), home, locked)
    mid = freeze(home + np.asarray(t["mid_offset"], dtype=float), home, locked)
    waypoints = [home, home, mid, dest, dest, mid, home]
    return JointTrajectory(waypoints, t["durations"]), home, locked


def _make_pid(
    cfg: dict,
    model: Any,
    nu: int,
    plate_site_id: int,
    ball_body_id: int,
    locked: np.ndarray,
) -> PidController:
    p = cfg["pid"]
    g = cfg["gains"]
    gains = PidGains(
        kp=float(p["Kp_ball"]),
        ki=float(p["Ki_ball"]),
        kd=float(p["Kd_ball"]),
        kff=float(p.get("Kff_ball", 0.0)),
        windup_limit=float(p["windup_limit"]),
    )
    return PidController(
        model=model,
        nu=nu,
        plate_site_id=plate_site_id,
        ball_body_id=ball_body_id,
        ball_gains=gains,
        joint_kp=g["Kp_joint"],
        joint_kd=g["Kd_joint"],
        actuation_alpha=float(p["alpha"]),
        max_tilt=float(p["max_tilt"]),
        locked_joints=locked.tolist(),
    )


def _make_mpc(
    cfg: dict,
    model: Any,
    nu: int,
    plate_site_id: int,
    ball_body_id: int,
    locked: np.ndarray,
    hw: HardwareSpec,
) -> MpcController:
    """Build the MPC controller from the ``mpc:`` block of the config.

    The MPC reuses the PID's ``alpha`` / ``max_tilt`` / joint-level gains,
    since those govern the post-PID actuator stack (Jacobian projection,
    low-pass, joint PD). Only the ball-level dynamics live in the MPC
    block: planning horizon, dt, and the LQR weights.
    """
    p = cfg["pid"]
    g = cfg["gains"]
    m = cfg.get("mpc", {})
    # ``from_hardware`` pulls rolling_factor / gravity from the shared
    # hardware spec, keeping the MPC's plant model in lockstep with the
    # simulator and the real arm.
    params = MpcParams.from_hardware(
        hw,
        horizon=int(m.get("horizon", 25)),
        dt=float(m.get("dt", 1.0 / 60.0)),
        q_pos=float(m.get("q_pos", 100.0)),
        q_vel=float(m.get("q_vel", 10.0)),
        r_u=float(m.get("r_u", 1.0)),
        qf_pos=float(m.get("qf_pos", 500.0)),
        qf_vel=float(m.get("qf_vel", 50.0)),
        max_output=float(p["max_tilt"]),
    )
    return MpcController(
        model=model,
        nu=nu,
        plate_site_id=plate_site_id,
        ball_body_id=ball_body_id,
        params=params,
        joint_kp=g["Kp_joint"],
        joint_kd=g["Kd_joint"],
        actuation_alpha=float(p["alpha"]),
        max_tilt=float(p["max_tilt"]),
        locked_joints=locked.tolist(),
    )


def _make_rl(
    cfg: dict,
    model: Any,
    nu: int,
    plate_site_id: int,
    ball_body_id: int,
    run_dir: Path,
    locked: np.ndarray,
):
    # Imported lazily here to keep PID-only runs free of the SB3 dependency.
    from ballplate.controllers.rl import RLPolicy
    from sim.controllers.rl import RLController

    p = cfg["pid"]
    g = cfg["gains"]
    ball_gains = PidGains(
        kp=float(p["Kp_ball"]),
        ki=float(p["Ki_ball"]),
        kd=float(p["Kd_ball"]),
        kff=float(p.get("Kff_ball", 0.0)),
        windup_limit=float(p["windup_limit"]),
    )
    return RLController(
        model=model,
        nu=nu,
        plate_site_id=plate_site_id,
        ball_body_id=ball_body_id,
        policy=RLPolicy(run_dir),
        ball_gains=ball_gains,
        joint_kp=g["Kp_joint"],
        joint_kd=g["Kd_joint"],
        actuation_alpha=float(p["alpha"]),
        max_tilt=float(p["max_tilt"]),
        locked_joints=locked.tolist(),
    )


# ============================================================================
# CLI
# ============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sim.runner")
    parser.add_argument(
        "--config",
        default="sim/config/sim.yaml",
        help="Path to the sim YAML config (relative to repo root).",
    )
    parser.add_argument("--controller", choices=["pid", "mpc", "rl"], default="pid")
    parser.add_argument(
        "--shape",
        default=None,
        help="Override ball_trajectory.shape " "(stationary|center|circle|figure8|drawn).",
    )
    parser.add_argument(
        "--drawn-file", default=None, help="Path to a drawn JSON; implies --shape drawn."
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=None,
        help="Override ball_trajectory.radius (built-in shapes only).",
    )
    parser.add_argument(
        "--hold-x",
        type=float,
        default=None,
        dest="hold_x",
        help="Stationary target x in plate-local metres (default 0).",
    )
    parser.add_argument(
        "--hold-y",
        type=float,
        default=None,
        dest="hold_y",
        help="Stationary target y in plate-local metres (default 0).",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=None,
        help="Override ball_trajectory.settle_time "
        "(blend window of the smooth-approach wrapper).",
    )
    parser.add_argument(
        "--lock-joints",
        default="config",
        help="Freeze arm joints at home: 'config' (read config/control.yaml), "
        "'all', 'none', or comma-separated indices like '0,1,2'.",
    )
    parser.add_argument("--on-drop", choices=["stop", "restart"], default="restart")
    parser.add_argument(
        "--record", default=None, help="Record an mp4 to this path (requires the [video] extra)."
    )
    parser.add_argument("--record-duration", type=float, default=15.0)
    parser.add_argument(
        "--rl-run-dir",
        default=None,
        help="Trained-run directory passed to ballplate.RLPolicy. "
        "Required when --controller rl.",
    )
    return parser.parse_args()


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    args = _parse_args()
    cfg = _load_config(Path(args.config))
    if args.settle_time is not None:
        cfg.setdefault("ball_trajectory", {})["settle_time"] = args.settle_time

    # Build the simulation world from the shared hardware spec.
    hw = load_hardware(hardware_default_path())
    model = build_scene(_scene_config(hw))
    data = mujoco.MjData(model)
    nu = model.nu

    plate_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "plate_center")
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    plate_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")

    joint_traj, home, locked = _build_joint_trajectory(cfg, args.lock_joints)
    if len(locked):
        print(f"Locked joints: {locked.tolist()} (held at home)")

    # Initialise the robot at home and place the ball at the plate centre so
    # the smooth-approach wrapper has a real starting pose to blend from.
    data.qpos[:nu] = home
    mujoco.mj_forward(model, data)

    monitor = BallDropMonitor(model, ball_body_id, plate_site_id)
    monitor.reset_ball(data)
    mujoco.mj_forward(model, data)

    # Reference + smooth approach (blend_time = 0 collapses to passthrough).
    inner_ref = _build_reference(cfg, args)
    blend_time = float(cfg.get("ball_trajectory", {}).get("settle_time", 0.0))
    reader = BallStateReader(model, plate_site_id, ball_body_id)
    ball0, _ = reader.read(data, timestamp=0.0)
    ball_traj = SmoothApproach(
        inner=inner_ref,
        start_x=ball0.x,
        start_y=ball0.y,
        blend_time=blend_time,
    )

    # Visualisation: live trail + static reference overlay glued to the plate.
    half_thick = hw.plate.thickness / 2.0
    trail = BallTrail(ball_body_id, plate_body_id, half_thick)
    overlay = TargetOverlay(
        reference=inner_ref,
        plate_body_id=plate_body_id,
        plate_half_thickness=half_thick,
        period=float(inner_ref.period),
    )

    # Controller.
    if args.controller == "pid":
        ctrl = _make_pid(cfg, model, nu, plate_site_id, ball_body_id, locked)
    elif args.controller == "mpc":
        ctrl = _make_mpc(cfg, model, nu, plate_site_id, ball_body_id, locked, hw)
    else:
        if args.rl_run_dir is None:
            raise SystemExit("--rl-run-dir is required when --controller rl")
        ctrl = _make_rl(cfg, model, nu, plate_site_id, ball_body_id, Path(args.rl_run_dir), locked)

    # Optional video recording.
    recorder = None
    if args.record:
        from sim.viz.recorder import Recorder

        recorder = Recorder(model, data, args.record, fps=int(cfg["render_fps"]))
        print(f"Recording {args.record_duration}s to {args.record}")

    # Main loop. The runner is the sole owner of the simulator clock.
    render_fps = float(cfg["render_fps"])
    steps_per_frame = int(1.0 / (render_fps * model.opt.timestep))
    sim_dt = float(model.opt.timestep)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_time = 0.0
        while viewer.is_running():
            frame_start = time.monotonic()

            for _ in range(steps_per_frame):
                ctrl.step(data, joint_traj, sim_time, ball_traj)
                mujoco.mj_step(model, data)
                sim_time += sim_dt

            if monitor.is_dropped(data):
                if args.on_drop == "stop":
                    print(f"Ball dropped at t={sim_time:.2f}s. Stopping.")
                    break
                print(f"Ball dropped at t={sim_time:.2f}s. Restarting ball.")
                monitor.reset_ball(data)
                mujoco.mj_forward(model, data)
                ctrl.reset()
                trail.clear()

            trail.record(data)
            trail.render(viewer, data)
            overlay.add_to_scene(viewer.user_scn, data)

            if recorder is not None:
                recorder.capture()
                if sim_time >= args.record_duration:
                    recorder.save()
                    recorder = None
                    print("Recording done.")

            viewer.sync()

            # Sleep to approximately maintain `render_fps`. Real sim time can
            # drift faster than wall-clock; the viewer will simply skip frames.
            elapsed = time.monotonic() - frame_start
            sleep_time = (1.0 / render_fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()

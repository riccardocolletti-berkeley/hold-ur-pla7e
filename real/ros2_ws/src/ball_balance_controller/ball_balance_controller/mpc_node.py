"""ROS 2 ball-on-plate MPC node for the UR arm.

Sibling to :mod:`ball_balance_controller.controller` (PID). The plumbing
(kinematics, TF, joint streaming, safety, recovery) is duplicated rather
than factored out so a regression here cannot affect the PID bring-up.

Pipeline (matches the PID node from omega_des down)::

    /ball_state    ->  MPC (ballplate)  ->  (ux, uy)
    Reference(t..) ->  horizon samples ----'
    TF base->plate ->  R_WP(q)
                                 |
                                 v
                       omega_des  =  -ux * y_plate + uy * x_plate    (world)
                       Delta_q    =  J_rot^+ * omega_des             (free joints)
                       Delta_q_f  =  alpha * Delta_q + (1-alpha) * Delta_q_f
                       q_cmd      =  q_current + Delta_q_f           (locked -> 0)
                                 |
                                 v
                       JointTrajectory  ->  /scaled_joint_trajectory_controller/joint_trajectory

What the MPC adds over the PID:

* Reference anticipation. Each tick pre-samples the reference at ``N``
  future steps and plans against that horizon, collapsing the tracking
  lag that PID has to live with on circle / figure8 / drawn paths.
* Finite-horizon LQR cost. Stage cost on position + velocity error and
  control effort; the terminal cost emulates infinite-horizon LQR for
  ``N`` large enough.

Not implemented yet: hard state constraints, hard rate constraints. The
output box ``max_tilt_rad`` is enforced by post-clipping; anything
tighter would need a real QP solver (osqp etc.).

The launch / params surface mirrors the PID node, so operators can swap
executables without changing the rest of the flow::

    ros2 launch ball_balance_controller mpc_balance.launch.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import rclpy
from ball_tracker_msgs.msg import BallState as BallStateMsg
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data

from ball_balance_controller.jacobian import JacobianProvider
from ball_balance_controller.joint_state_reader import DEFAULT_JOINT_NAMES, JointStateReader
from ball_balance_controller.tf_plate_reader import PlateTfReader
from ball_balance_controller.trajectory_client import JointTrajectoryClient
from ballplate.control import default_path as control_default_path
from ballplate.control import load as load_control
from ballplate.control import parse_lock_spec
from ballplate.controllers.mpc import MpcController, MpcParams
from ballplate.hardware import HardwareSpec
from ballplate.hardware import default_path as hardware_default_path
from ballplate.hardware import load as load_hardware
from ballplate.state import BallState, PlateGeometry
from ballplate.trajectories import Circle, DrawnPath, Figure8, Reference, Stationary

#: How often to re-emit a sticky warning while its condition persists.
_PERSISTENT_LOG_PERIOD_S = 1.0

#: How often to emit the per-tick control summary (ball pose, |ω|, |Δq|).
_INFO_LOG_PERIOD_S = 1.0

#: Allowed reference shapes (used by config validation).
_REFERENCE_SHAPES = ("stationary", "circle", "figure8", "drawn")


@dataclass(frozen=True)
class MpcNodeConfig:
    """Frozen snapshot of every node parameter, validated at init.

    Most fields mirror ``ControllerConfig`` (the PID node) so the
    plumbing-level params (URDF path, home pose, command rate, safety
    windows, etc.) carry over verbatim. The PID-specific gains
    (kp/ki/kd/kff/integral_limit) are replaced by the MPC's LQR weights
    (q_pos/q_vel/r_u/qf_pos/qf_vel) plus a horizon and a planning dt.
    """

    # Robot / kinematics
    urdf_path: str
    home_joints: tuple[float, ...]
    world_frame: str
    plate_frame: str
    locked_joints_spec: str

    # MPC tuning
    horizon: int
    dt: float
    q_pos: float
    q_vel: float
    r_u: float
    qf_pos: float
    qf_vel: float

    # Safety / shaping (shared with PID node)
    max_tilt_rad: float
    max_joint_delta_rad: float
    max_dt_s: float

    # Timing / filtering (shared with PID node)
    command_rate_hz: float
    command_alpha: float
    trajectory_time_s: float
    startup_home_time_s: float
    return_to_base_rate: float

    # Reference trajectory
    reference_shape: str
    reference_radius_m: float
    reference_period_s: float
    hold_x_m: float
    hold_y_m: float
    drawn_file: str

    # Tracker freshness
    stale_ball_state_s: float
    missing_ball_grace_s: float
    markers_min_visible: int

    def validate(self) -> None:
        checks: list[tuple[bool, str]] = [
            (Path(self.urdf_path).is_file(), f"urdf_path does not exist: {self.urdf_path!r}"),
            (
                len(self.home_joints) == 6,
                f"home_joints must have 6 entries, got {len(self.home_joints)}",
            ),
            (self.horizon >= 1, f"horizon must be >= 1, got {self.horizon}"),
            (self.dt > 0.0, f"mpc dt must be > 0, got {self.dt}"),
            (self.q_pos >= 0.0, f"q_pos must be >= 0, got {self.q_pos}"),
            (self.q_vel >= 0.0, f"q_vel must be >= 0, got {self.q_vel}"),
            (
                self.r_u > 0.0,
                f"r_u must be > 0 (otherwise the MPC Hessian is rank-deficient), got {self.r_u}",
            ),
            (self.qf_pos >= 0.0, f"qf_pos must be >= 0, got {self.qf_pos}"),
            (self.qf_vel >= 0.0, f"qf_vel must be >= 0, got {self.qf_vel}"),
            (self.max_tilt_rad > 0.0, f"max_tilt_rad must be > 0, got {self.max_tilt_rad}"),
            (
                self.max_joint_delta_rad > 0.0,
                f"max_joint_delta_rad must be > 0, got {self.max_joint_delta_rad}",
            ),
            (self.max_dt_s > 0.0, f"max_dt_s must be > 0, got {self.max_dt_s}"),
            (
                self.command_rate_hz > 0.0,
                f"command_rate_hz must be > 0, got {self.command_rate_hz}",
            ),
            (
                0.0 <= self.command_alpha <= 1.0,
                f"command_alpha must be in [0,1], got {self.command_alpha}",
            ),
            (
                self.trajectory_time_s >= 0.0,
                f"trajectory_time_s must be >= 0, got {self.trajectory_time_s}",
            ),
            (
                self.startup_home_time_s > 0.0,
                f"startup_home_time_s must be > 0, got {self.startup_home_time_s}",
            ),
            (
                0.0 <= self.return_to_base_rate <= 1.0,
                f"return_to_base_rate must be in [0,1], got {self.return_to_base_rate}",
            ),
            (
                self.reference_shape in _REFERENCE_SHAPES,
                f"reference_shape must be one of {_REFERENCE_SHAPES}, got {self.reference_shape!r}",
            ),
            (
                self.reference_radius_m >= 0.0,
                f"reference_radius_m must be >= 0, got {self.reference_radius_m}",
            ),
            (
                self.reference_period_s > 0.0,
                f"reference_period_s must be > 0, got {self.reference_period_s}",
            ),
            (
                self.stale_ball_state_s > 0.0,
                f"stale_ball_state_s must be > 0, got {self.stale_ball_state_s}",
            ),
            (
                self.missing_ball_grace_s > 0.0,
                f"missing_ball_grace_s must be > 0, got {self.missing_ball_grace_s}",
            ),
            (
                self.markers_min_visible >= 0,
                f"markers_min_visible must be >= 0, got {self.markers_min_visible}",
            ),
        ]
        if self.reference_shape == "drawn":
            checks.append(
                (bool(self.drawn_file), "reference_shape='drawn' requires non-empty drawn_file")
            )

        problems = [msg for ok, msg in checks if not ok]
        if problems:
            raise SystemExit(
                "MPC controller config failed validation:\n  - " + "\n  - ".join(problems)
            )


class BallBalanceMpcController(Node):
    """Closed-loop ball-on-plate MPC controller running on top of the UR arm."""

    def __init__(self) -> None:
        super().__init__("ball_balance_mpc")

        self._declare_parameters()
        self._cfg = self._init_config()

        self._hw: HardwareSpec = load_hardware(hardware_default_path())
        self._plate_geom = PlateGeometry(
            size_x=self._hw.plate.size[0],
            size_y=self._hw.plate.size[1],
            thickness=self._hw.plate.thickness,
        )

        self._init_runtime_state()
        self._init_mpc()
        self._init_kinematics()
        self._init_reference()
        self._init_ros_io()

        self.get_logger().info("BallBalanceMpcController ready.")

    # ============================================================ config --

    def _init_config(self) -> MpcNodeConfig:
        cfg = MpcNodeConfig(
            urdf_path=str(self._p("urdf_path")),
            home_joints=tuple(float(v) for v in self._p("home_joints")),
            world_frame=str(self._p("world_frame")),
            plate_frame=str(self._p("plate_frame")),
            locked_joints_spec=str(self._p("locked_joints_spec")),
            horizon=int(self._p("horizon")),
            dt=float(self._p("dt")),
            q_pos=float(self._p("q_pos")),
            q_vel=float(self._p("q_vel")),
            r_u=float(self._p("r_u")),
            qf_pos=float(self._p("qf_pos")),
            qf_vel=float(self._p("qf_vel")),
            max_tilt_rad=float(self._p("max_tilt_rad")),
            max_joint_delta_rad=float(self._p("max_joint_delta_rad")),
            max_dt_s=float(self._p("max_dt_s")),
            command_rate_hz=float(self._p("command_rate_hz")),
            command_alpha=float(self._p("command_alpha")),
            trajectory_time_s=float(self._p("trajectory_time_s")),
            startup_home_time_s=float(self._p("startup_home_time_s")),
            return_to_base_rate=float(self._p("return_to_base_rate")),
            reference_shape=str(self._p("reference_shape")),
            reference_radius_m=float(self._p("reference_radius_m")),
            reference_period_s=float(self._p("reference_period_s")),
            hold_x_m=float(self._p("hold_x_m")),
            hold_y_m=float(self._p("hold_y_m")),
            drawn_file=str(self._p("drawn_file")),
            stale_ball_state_s=float(self._p("stale_ball_state_s")),
            missing_ball_grace_s=float(self._p("missing_ball_grace_s")),
            markers_min_visible=int(self._p("markers_min_visible")),
        )
        cfg.validate()
        self._log_config(cfg)
        return cfg

    def _log_config(self, cfg: MpcNodeConfig) -> None:
        lines = ["MPC controller config (loaded once at init):"]
        for f in fields(cfg):
            val = getattr(cfg, f.name)
            lines.append(f"  {f.name:24s} = {val!r}")
        self.get_logger().info("\n".join(lines))

    # ============================================================= params --

    def _declare_parameters(self) -> None:
        # Robot / kinematics
        self.declare_parameter("urdf_path", Parameter.Type.STRING)
        self.declare_parameter("home_joints", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("world_frame", Parameter.Type.STRING)
        self.declare_parameter("plate_frame", Parameter.Type.STRING)
        self.declare_parameter("locked_joints_spec", Parameter.Type.STRING)

        # MPC tuning
        self.declare_parameter("horizon", Parameter.Type.INTEGER)
        for name in ("dt", "q_pos", "q_vel", "r_u", "qf_pos", "qf_vel"):
            self.declare_parameter(name, Parameter.Type.DOUBLE)

        # Safety / shaping
        for name in ("max_tilt_rad", "max_joint_delta_rad", "max_dt_s"):
            self.declare_parameter(name, Parameter.Type.DOUBLE)

        # Timing / filtering
        for name in (
            "command_rate_hz",
            "command_alpha",
            "trajectory_time_s",
            "startup_home_time_s",
            "return_to_base_rate",
        ):
            self.declare_parameter(name, Parameter.Type.DOUBLE)

        # Reference trajectory
        self.declare_parameter("reference_shape", Parameter.Type.STRING)
        self.declare_parameter("reference_radius_m", Parameter.Type.DOUBLE)
        self.declare_parameter("reference_period_s", Parameter.Type.DOUBLE)
        self.declare_parameter("hold_x_m", Parameter.Type.DOUBLE)
        self.declare_parameter("hold_y_m", Parameter.Type.DOUBLE)
        self.declare_parameter("drawn_file", Parameter.Type.STRING)

        # Tracker freshness limits
        self.declare_parameter("stale_ball_state_s", Parameter.Type.DOUBLE)
        self.declare_parameter("missing_ball_grace_s", Parameter.Type.DOUBLE)
        self.declare_parameter("markers_min_visible", Parameter.Type.INTEGER)

    def _p(self, name: str):
        param = self.get_parameter(name)
        if param.type_ == Parameter.Type.NOT_SET:
            raise SystemExit(
                f"Required parameter '{name}' is not set. The launch file "
                f"must load config/mpc_params.yaml (no in-code fallbacks)."
            )
        return param.value

    # ======================================================== runtime state --

    def _init_runtime_state(self) -> None:
        self._home_joints = np.asarray(self._cfg.home_joints, dtype=float)
        self._joint_names = list(DEFAULT_JOINT_NAMES)

        control = load_control(control_default_path())
        self._locked_joints = parse_lock_spec(
            self._cfg.locked_joints_spec,
            n_joints=len(self._home_joints),
            config_indices=control.locked_joints,
        ).astype(int)
        self._free_cols = np.setdiff1d(
            np.arange(len(self._home_joints)),
            self._locked_joints,
            assume_unique=False,
        )
        if self._locked_joints.size:
            self.get_logger().info(f"Locked joints: {self._locked_joints.tolist()} (held at home)")

        self._last_ball_msg: BallStateMsg | None = None
        self._last_ball_time: float | None = None
        self._last_tracking_valid_time: float | None = None

        self._dq_filtered = np.zeros(len(self._home_joints), dtype=float)
        self._last_step_time = time.monotonic()

        self._sticky_warnings: dict[str, tuple[float, float]] = {}
        self._info_last_log: float = 0.0
        self._engage_after: float = float("inf")

    def _init_mpc(self) -> None:
        # ``from_hardware`` keeps the MPC's plant model in lockstep with
        # ``config/hardware.yaml`` (same source of truth the simulator and
        # static-TF launch use). ``max_output`` is the per-axis tilt cap;
        # the world-frame projection guard below catches anything that cap
        # alone would miss when the plate is rotated.
        params = MpcParams.from_hardware(
            self._hw,
            horizon=self._cfg.horizon,
            dt=self._cfg.dt,
            q_pos=self._cfg.q_pos,
            q_vel=self._cfg.q_vel,
            r_u=self._cfg.r_u,
            qf_pos=self._cfg.qf_pos,
            qf_vel=self._cfg.qf_vel,
            max_output=self._cfg.max_tilt_rad,
        )
        self._mpc = MpcController(params)

    # ====================================================== kinematics deps --

    def _init_kinematics(self) -> None:
        self._jacobian = JacobianProvider(self._cfg.urdf_path)
        self.get_logger().info(
            f"Loaded ikpy chain from {self._cfg.urdf_path} "
            f"({self._jacobian.n_links} links, {self._jacobian.n_joints} active)"
        )
        if self._jacobian.n_joints != len(self._home_joints):
            raise RuntimeError(
                f"URDF active joints ({self._jacobian.n_joints}) does not match "
                f"home_joints length ({len(self._home_joints)})."
            )

    # ============================================================ reference --

    def _init_reference(self) -> None:
        shape = self._cfg.reference_shape
        radius = self._cfg.reference_radius_m
        period = self._cfg.reference_period_s

        self._reference: Reference
        if shape == "stationary":
            self._reference = Stationary(x=self._cfg.hold_x_m, y=self._cfg.hold_y_m)
        elif shape == "circle":
            self._reference = Circle(radius=radius, period=period)
        elif shape == "figure8":
            self._reference = Figure8(rx=radius, ry=radius / 2.0, period=period)
        elif shape == "drawn":
            self._reference = DrawnPath.from_file(self._cfg.drawn_file)
        else:
            raise SystemExit(f"Unknown reference_shape: {shape!r}")

    # ============================================================== ROS IO --

    def _init_ros_io(self) -> None:
        self._cb_control = MutuallyExclusiveCallbackGroup()
        self._cb_subs = ReentrantCallbackGroup()

        self._joint_reader = JointStateReader(
            self,
            self._joint_names,
            callback_group=self._cb_subs,
        )
        self._tf_plate = PlateTfReader(
            self,
            world_frame=self._cfg.world_frame,
            plate_frame=self._cfg.plate_frame,
        )
        self._traj_client = JointTrajectoryClient(
            self,
            self._joint_names,
            joint_state_reader=self._joint_reader,
            callback_group=self._cb_control,
        )

        self.create_subscription(
            BallStateMsg,
            "/ball_state",
            self._on_ball_state,
            qos_profile_sensor_data,
            callback_group=self._cb_subs,
        )
        self._pub_plate_cmd = self.create_publisher(Vector3Stamped, "/plate_cmd", 10)
        # Live reference point the controller is currently chasing, in the
        # plate frame. Published so the visualizer renders the controller's
        # actual setpoint instead of re-evaluating the reference on its own
        # clock (the two clocks were previously unrelated).
        self._pub_reference_target = self.create_publisher(PointStamped, "/reference_target", 10)

        startup_time = self._cfg.startup_home_time_s
        self.get_logger().info(f"Moving to home pose ({startup_time:.1f}s) …")
        self._traj_client.send(self._home_joints.tolist(), trajectory_time_s=startup_time)
        self._engage_after = time.monotonic() + startup_time

        self.create_timer(
            1.0 / self._cfg.command_rate_hz,
            self._on_step,
            callback_group=self._cb_control,
        )

    # ============================================================ callbacks --

    def _on_ball_state(self, msg: BallStateMsg) -> None:
        self._last_ball_msg = msg
        self._last_ball_time = time.monotonic()
        if msg.tracking_valid:
            self._last_tracking_valid_time = self._last_ball_time

    # ====================================================== control step --

    def _on_step(self) -> None:
        now = time.monotonic()
        # ``dt`` from the timer is informational here; the MPC's
        # discretisation step is fixed at construction.
        self._tick_dt(now)

        if now < self._engage_after or self._traj_client.is_busy():
            return

        if self._last_ball_msg is None or self._last_ball_time is None:
            self._warn_sticky("no /ball_state", now, "is the ball tracker running?")
            return
        self._clear_sticky("no /ball_state", now)

        if self._safety_should_recover(now):
            self._return_to_home()
            return

        msg = self._last_ball_msg

        if not msg.tracking_valid:
            return

        if not msg.ball_found:
            self._warn_sticky("KF coast", now, "no fresh ball detection (motion blur?)")
        else:
            self._clear_sticky("KF coast", now)

        # Forward-extrapolate position to "now" using the latest velocity
        # to cut the camera-rate sensing lag (same trick as the PID node).
        elapsed_since_msg = max(0.0, now - self._last_ball_time)
        ball = BallState(
            x=msg.x + msg.vx * elapsed_since_msg,
            y=msg.y + msg.vy * elapsed_since_msg,
            vx=msg.vx,
            vy=msg.vy,
            timestamp=now,
            valid=True,
        )

        # Tap the current reference point for the visualizer. Sampled here
        # rather than inside the MPC because ``step_with_reference`` already
        # owns the horizon walk and does not return the t=now sample.
        x_ref, y_ref, _, _ = self._reference.evaluate(now)
        self._publish_reference_target((x_ref, y_ref))

        # MPC step: pre-sample the reference along the horizon so the
        # controller anticipates the moving target. Using
        # ``step_with_reference`` gives the MPC its full predictive value;
        # falling back to ``step`` (constant target) collapses to a tracking
        # quality similar to a feed-forward-less PID.
        ux, uy = self._mpc.step_with_reference(
            ball=ball,
            reference=self._reference,
            now=now,
        )

        q_current = self._joint_reader.latest()
        plate_mat = self._tf_plate.latest_rotation()
        if q_current is None or plate_mat is None:
            self._warn_sticky(
                "TF/joint missing",
                now,
                lambda: (
                    f"q={'ok' if q_current is not None else 'None'} "
                    f"plate={'ok' if plate_mat is not None else 'None'}"
                ),
            )
            return
        self._clear_sticky("TF/joint missing", now)

        omega_des = self._project_to_omega(ux, uy, plate_mat)

        # IK on free joints only.
        dq_raw = self._omega_to_joint_delta(omega_des, q_current)
        np.clip(dq_raw, -self._cfg.max_joint_delta_rad, self._cfg.max_joint_delta_rad, out=dq_raw)

        alpha = self._cfg.command_alpha
        self._dq_filtered = alpha * dq_raw + (1.0 - alpha) * self._dq_filtered
        if self._locked_joints.size:
            self._dq_filtered[self._locked_joints] = 0.0

        q_target = q_current + self._dq_filtered

        if now - self._info_last_log >= _INFO_LOG_PERIOD_S:
            self._info_last_log = now
            drift = float(np.max(np.abs(q_target - self._home_joints)))
            omega_norm = float(np.linalg.norm(omega_des))
            dq_norm = float(np.linalg.norm(self._dq_filtered))
            self.get_logger().info(
                f"[mpc] ball=({msg.x:+.3f},{msg.y:+.3f}) "
                f"u=({ux:+.3f},{uy:+.3f}) "
                f"|ω|={omega_norm:.3f}  |Δq|={dq_norm:.3f}  drift={drift:.3f} rad"
            )

        self._traj_client.send(
            q_target.tolist(),
            trajectory_time_s=self._cfg.trajectory_time_s,
        )
        self._publish_plate_cmd(omega_des)

    # ========================================================= internals --

    def _tick_dt(self, now: float) -> float:
        raw = now - self._last_step_time
        self._last_step_time = now
        if raw <= 0.0:
            return 1e-3
        return min(max(raw, 1e-3), self._cfg.max_dt_s)

    # ----------------------------------------------------- sticky-log helpers --

    def _warn_sticky(self, name: str, now: float, message) -> None:
        state = self._sticky_warnings.get(name)
        if state is None:
            self._sticky_warnings[name] = (now, now)
            text = message() if callable(message) else message
            self.get_logger().warning(f"{name}: {text}")
            return
        entered_at, last_logged = state
        if now - last_logged >= _PERSISTENT_LOG_PERIOD_S:
            self._sticky_warnings[name] = (entered_at, now)
            text = message() if callable(message) else message
            self.get_logger().warning(f"{name}: {text} (active for {now - entered_at:.1f}s)")

    def _clear_sticky(self, name: str, now: float) -> None:
        state = self._sticky_warnings.pop(name, None)
        if state is not None:
            entered_at, _ = state
            self.get_logger().info(f"{name}: cleared after {now - entered_at:.1f}s")

    def _safety_should_recover(self, now: float) -> bool:
        msg_age = now - (self._last_ball_time or 0.0)
        if msg_age > self._cfg.stale_ball_state_s:
            self._warn_sticky("ball_state stale", now, lambda: f"msg age {msg_age * 1000.0:.0f} ms")
            return True
        self._clear_sticky("ball_state stale", now)

        msg = self._last_ball_msg
        if msg is None:
            return True

        if msg.markers_found < self._cfg.markers_min_visible:
            self._warn_sticky(
                "low markers",
                now,
                lambda: f"{msg.markers_found}/{self._cfg.markers_min_visible} required",
            )
            return True
        self._clear_sticky("low markers", now)

        if self._last_tracking_valid_time is None:
            tracking_age = msg_age
        else:
            tracking_age = now - self._last_tracking_valid_time
        if tracking_age > self._cfg.missing_ball_grace_s:
            self._warn_sticky(
                "ball not visible",
                now,
                lambda: f"tracking_valid stale {tracking_age * 1000.0:.0f} ms",
            )
            return True
        self._clear_sticky("ball not visible", now)

        return False

    def _project_to_omega(self, ux: float, uy: float, plate_mat: np.ndarray) -> np.ndarray:
        x_hat = plate_mat[:, 0]
        y_hat = plate_mat[:, 1]
        omega = -(ux * y_hat - uy * x_hat)

        max_tilt = self._cfg.max_tilt_rad
        roll, pitch = self._plate_roll_pitch(plate_mat)
        if (roll >= max_tilt and omega[0] > 0.0) or (roll <= -max_tilt and omega[0] < 0.0):
            omega[0] = 0.0
        if (pitch >= max_tilt and omega[1] > 0.0) or (pitch <= -max_tilt and omega[1] < 0.0):
            omega[1] = 0.0
        return omega

    @staticmethod
    def _plate_roll_pitch(plate_mat: np.ndarray) -> tuple[float, float]:
        roll = float(np.arctan2(plate_mat[2, 1], plate_mat[2, 2]))
        pitch = float(np.arctan2(-plate_mat[2, 0], np.hypot(plate_mat[2, 1], plate_mat[2, 2])))
        return roll, pitch

    def _omega_to_joint_delta(self, omega_world: np.ndarray, q_ros: np.ndarray) -> np.ndarray:
        j_rot = self._jacobian.rotational(q_ros)
        n = j_rot.shape[1]
        dq = np.zeros(n)
        if self._free_cols.size:
            dq[self._free_cols], *_ = np.linalg.lstsq(
                j_rot[:, self._free_cols], omega_world, rcond=None
            )
        return dq

    def _publish_reference_target(self, target_pos: tuple[float, float]) -> None:
        """Publish the live reference point for the visualizer (plate frame)."""
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._cfg.plate_frame
        msg.point.x = float(target_pos[0])
        msg.point.y = float(target_pos[1])
        msg.point.z = 0.0
        self._pub_reference_target.publish(msg)

    def _publish_plate_cmd(self, omega_des: np.ndarray) -> None:
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._cfg.plate_frame
        msg.vector.x = float(omega_des[0])
        msg.vector.y = float(omega_des[1])
        msg.vector.z = 0.0
        self._pub_plate_cmd.publish(msg)

    def _return_to_home(self) -> None:
        """Per-tick exponential decay of dq toward zero. Fast path so a
        transient safety condition doesn't latch ``is_busy()``.
        """
        self._mpc.reset()
        self._dq_filtered = (1.0 - self._cfg.return_to_base_rate) * self._dq_filtered
        if self._locked_joints.size:
            self._dq_filtered[self._locked_joints] = 0.0
        q_target = self._home_joints + self._dq_filtered
        self._traj_client.send(q_target.tolist(), trajectory_time_s=0.0)


# ============================================================ entry point --


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = BallBalanceMpcController()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        try:
            executor.spin()
        finally:
            executor.shutdown()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

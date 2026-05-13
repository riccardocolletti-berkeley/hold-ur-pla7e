"""ROS 2 ball-on-plate PID node for the UR arm.

Pipeline (matches the MPC node from omega_des down)::

    /ball_state    ->  PID (ballplate)  ->  (ux, uy)
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

Safety guards run before the PID and short-circuit to ``_return_to_home``:

* stale tracker         (>``stale_ball_state_s`` without /ball_state)
* insufficient markers  (<``markers_min_visible`` ArUco markers visible)
* ball lost             (>``missing_ball_grace_s`` without a fresh detection)

Live YAML reload lives in :mod:`controller_live_reload`; the dataclass
that snapshots every parameter lives in :mod:`controller_config`. Only
ROS plumbing and ikpy live in this file.
"""

from __future__ import annotations

import time
from dataclasses import fields
from typing import Optional

import numpy as np
import rclpy
from ball_tracker_msgs.msg import BallState as BallStateMsg
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data

from ball_balance_controller.controller_config import ControllerConfig
from ball_balance_controller.controller_live_reload import (
    PARAMS_FILE_PATH_PARAM,
    REFERENCE_FILE_PATH_PARAM,
    LiveReloadMixin,
)
from ball_balance_controller.jacobian import JacobianProvider
from ball_balance_controller.joint_state_reader import DEFAULT_JOINT_NAMES, JointStateReader
from ball_balance_controller.tf_plate_reader import PlateTfReader
from ball_balance_controller.trajectory_client import JointTrajectoryClient
from ballplate.control import default_path as control_default_path
from ballplate.control import load as load_control
from ballplate.control import parse_lock_spec
from ballplate.controllers.pid import PidController, PidGains
from ballplate.hardware import HardwareSpec
from ballplate.hardware import default_path as hardware_default_path
from ballplate.hardware import load as load_hardware
from ballplate.safety import deadband, velocity_clip
from ballplate.state import BallState, PlateGeometry
from ballplate.trajectories import Circle, DrawnPath, Figure8, Reference, Stationary

#: Period of sticky-warning re-emission while a fault persists.
_PERSISTENT_LOG_PERIOD_S = 1.0

#: Period of the per-tick INFO summary (ball pose, |omega|, |dq|).
_INFO_LOG_PERIOD_S = 1.0


class BallBalanceController(Node, LiveReloadMixin):
    """Closed-loop ball-on-plate PID controller running on top of the UR arm."""

    def __init__(self) -> None:
        super().__init__("ball_balance_controller")

        self._declare_parameters()
        # Read every parameter, validate, log. The control loop reads
        # ``self._cfg`` only; live edits replace the snapshot atomically.
        self._cfg = self._init_config()

        self._hw: HardwareSpec = load_hardware(hardware_default_path())
        self._plate_geom = PlateGeometry(
            size_x=self._hw.plate.size[0],
            size_y=self._hw.plate.size[1],
            thickness=self._hw.plate.thickness,
        )

        self._init_runtime_state()
        self._init_pid()
        self._init_kinematics()
        self._init_reference()
        self._init_ros_io()
        # Register the param callback only after every actor (PID, reference,
        # kinematics) is built; an early ``ros2 param set`` would otherwise
        # land on a half-constructed node.
        self.init_live_reload()

        self.get_logger().info("BallBalanceController ready.")

    # ============================================================ config --

    def _init_config(self) -> ControllerConfig:
        """Read every parameter at startup, validate, log, and return the snapshot."""
        cfg = ControllerConfig(
            urdf_path=str(self._p("urdf_path")),
            home_joints=tuple(float(v) for v in self._p("home_joints")),
            world_frame=str(self._p("world_frame")),
            plate_frame=str(self._p("plate_frame")),
            locked_joints_spec=str(self._p("locked_joints_spec")),
            kp=float(self._p("kp")),
            ki=float(self._p("ki")),
            kd=float(self._p("kd")),
            kff=float(self._p("kff")),
            integral_limit=float(self._p("integral_limit")),
            error_clip_m=float(self._p("error_clip_m")),
            max_tilt_rad=float(self._p("max_tilt_rad")),
            max_joint_delta_rad=float(self._p("max_joint_delta_rad")),
            deadband_m=float(self._p("deadband_m")),
            velocity_clip_mps=float(self._p("velocity_clip_mps")),
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

    def _log_config(self, cfg: ControllerConfig) -> None:
        """Dump every loaded value in one block at startup."""
        lines = ["Controller config (loaded once at init):"]
        for f in fields(cfg):
            val = getattr(cfg, f.name)
            lines.append(f"  {f.name:24s} = {val!r}")
        self.get_logger().info("\n".join(lines))

    # ============================================================= params --

    def _declare_parameters(self) -> None:
        """Declare every parameter with a type only (no in-code defaults).

        Every value must come from the launch file's ``pid_params.yaml``;
        ``_p`` aborts on missing values rather than letting a stale in-code
        default win.
        """
        # Robot / kinematics
        self.declare_parameter("urdf_path", Parameter.Type.STRING)
        self.declare_parameter("home_joints", Parameter.Type.DOUBLE_ARRAY)
        self.declare_parameter("world_frame", Parameter.Type.STRING)
        self.declare_parameter("plate_frame", Parameter.Type.STRING)

        # PID gains. Inputs: ball position error (m), velocity error (m/s).
        # Output (ux, uy) is a plate-tilt angle in radians, matching
        # sim.adapters.actuation.JointActuator (which also does NOT scale by
        # dt). ``kff`` consumes the target acceleration finite-differenced
        # from the reference velocity.
        for name in ("kp", "ki", "kd", "kff"):
            self.declare_parameter(name, Parameter.Type.DOUBLE)

        # Safety / shaping. ``max_dt_s`` clips a stalled control-loop tick
        # so one slow timer firing cannot blow up the integrator next tick.
        for name in (
            "max_tilt_rad",
            "max_joint_delta_rad",
            "deadband_m",
            "integral_limit",
            "error_clip_m",
            "velocity_clip_mps",
            "max_dt_s",
        ):
            self.declare_parameter(name, Parameter.Type.DOUBLE)

        # Timing / filtering. ``startup_home_time_s`` covers the initial
        # home move dispatched before the PID engages.
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

        # Tracker freshness limits. ``markers_min_visible`` triggers the
        # home-recovery branch as soon as the tracker sees fewer than this
        # many ArUco markers (faster than waiting for ``missing_ball_grace_s``
        # when the homography collapses).
        self.declare_parameter("stale_ball_state_s", Parameter.Type.DOUBLE)
        self.declare_parameter("missing_ball_grace_s", Parameter.Type.DOUBLE)
        self.declare_parameter("markers_min_visible", Parameter.Type.INTEGER)
        # Joint-locking spec, parsed by ballplate.control.parse_lock_spec.
        self.declare_parameter("locked_joints_spec", Parameter.Type.STRING)

        # Paths the live-reload watcher should poll. Set by the launch file
        # to the same YAMLs it loads via ``parameters=[...]``. Empty default
        # disables the watcher; ``ros2 param set`` / ``ros2 param load`` still
        # work.
        self.declare_parameter(PARAMS_FILE_PATH_PARAM, "")
        self.declare_parameter(REFERENCE_FILE_PATH_PARAM, "")

    def _p(self, name: str):
        param = self.get_parameter(name)
        if param.type_ == Parameter.Type.NOT_SET:
            raise SystemExit(
                f"Required parameter '{name}' is not set. The launch file "
                f"must load config/pid_params.yaml (no in-code fallbacks)."
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

        # Latest tracker frame and the monotonic time we received it.
        self._last_ball_msg: BallStateMsg | None = None
        self._last_ball_time: float | None = None

        # Last monotonic time the tracker reported tracking_valid=True. The
        # missing-ball grace check uses this; message age alone never trips
        # because /ball_state keeps publishing at camera rate even with
        # tracking_valid=False.
        self._last_tracking_valid_time: float | None = None

        # Filtered joint offset from current (mirrors JointActuator._dq_filtered).
        self._dq_filtered = np.zeros(len(self._home_joints), dtype=float)
        self._last_step_time = time.monotonic()

        # Reference-velocity finite difference for the kff feed-forward.
        self._last_ref_vel: tuple[float, float] = (0.0, 0.0)
        self._last_ref_time: float | None = None

        # Sticky-warning bookkeeping: name -> (entered_at, last_logged_at).
        self._sticky_warnings: dict[str, tuple[float, float]] = {}

        # Wall-clock throttle for the per-tick INFO summary (not tick-mod:
        # at 500 Hz a tick-mod throttle fires every 60 ms, way too noisy).
        self._info_last_log: float = 0.0

        # Wall-clock at which it is safe to start commanding the arm,
        # set once the startup-home goal has been dispatched.
        self._engage_after: float = float("inf")

    def _init_pid(self) -> None:
        gains = PidGains(
            kp=self._cfg.kp,
            ki=self._cfg.ki,
            kd=self._cfg.kd,
            kff=self._cfg.kff,
            error_clip=self._cfg.error_clip_m,
            windup_limit=self._cfg.integral_limit,
        )
        self._pid = PidController(gains)

    # ====================================================== kinematics deps --

    def _init_kinematics(self) -> None:
        # urdf_path existence was checked by ControllerConfig.validate.
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
        self._reference: Reference = self._build_reference_from(self._cfg)

    @staticmethod
    def _build_reference_from(cfg: ControllerConfig) -> Reference:
        """Build a Reference from a config snapshot.

        Pure: takes its config explicitly so the live-reload path can build
        and validate a candidate reference *before* swapping it in (a bad
        ``drawn_file`` raises here, not after the existing trajectory has
        already been replaced).
        """
        shape = cfg.reference_shape
        if shape == "stationary":
            return Stationary(x=cfg.hold_x_m, y=cfg.hold_y_m)
        if shape == "circle":
            return Circle(radius=cfg.reference_radius_m, period=cfg.reference_period_s)
        if shape == "figure8":
            return Figure8(
                rx=cfg.reference_radius_m,
                ry=cfg.reference_radius_m / 2.0,
                period=cfg.reference_period_s,
            )
        if shape == "drawn":
            return DrawnPath.from_file(cfg.drawn_file)
        # Unreachable: ControllerConfig.problems() rejects unknown shapes.
        raise ValueError(f"Unknown reference_shape: {shape!r}")

    # ============================================================== ROS IO --

    def _init_ros_io(self) -> None:
        # Two callback groups so the control timer is not preempted by the
        # UR /joint_states stream or the TF buffer's listener:
        #   * control + trajectory_client timers -> mutex group
        #     (serialised on one executor thread; no re-entrancy of the step)
        #   * subscribers (ball_state, joint_states, TF, ...)
        #     -> reentrant group (any executor thread may dispatch them)
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
        # Pass the joint reader so the streamer can ramp from the live pose
        # on the very first planned move (startup home), instead of jumping.
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
        # plate frame. The visualizer renders this directly so its target
        # marker can never drift from the controller's actual setpoint.
        self._pub_reference_target = self.create_publisher(PointStamped, "/reference_target", 10)

        # Move the arm to home so the first control step has a sensible pose
        # for the plate-frame TF lookup.
        startup_time = self._cfg.startup_home_time_s
        self.get_logger().info(f"Moving to home pose ({startup_time:.1f}s) ...")
        self._traj_client.send(self._home_joints.tolist(), trajectory_time_s=startup_time)
        # Block PID engagement until the home move has had time to complete.
        self._engage_after = time.monotonic() + startup_time

        self.create_timer(
            1.0 / self._cfg.command_rate_hz,
            self._on_step,
            callback_group=self._cb_control,
        )

    # ====================================================== apply config --

    def _apply_config(
        self,
        new_cfg: ControllerConfig,
        overlay: dict[str, object],
        new_reference: Reference | None,
    ) -> None:
        """Swap config + dependent actors and log the diff. Single writer thread."""
        old_cfg = self._cfg
        self._cfg = new_cfg

        pid_fields = {"kp", "ki", "kd", "kff", "integral_limit", "error_clip_m"}
        if overlay.keys() & pid_fields:
            # Rebind the gains object on the existing PID; the integrator
            # state survives small tweaks (the operator nudging kp mid-run
            # does not want their integral wiped).
            self._pid.gains = PidGains(
                kp=new_cfg.kp,
                ki=new_cfg.ki,
                kd=new_cfg.kd,
                kff=new_cfg.kff,
                error_clip=new_cfg.error_clip_m,
                windup_limit=new_cfg.integral_limit,
            )

        if new_reference is not None:
            self._reference = new_reference
            # The kff finite-difference straddles the boundary between two
            # reference shapes; clear it so the first tick on the new path
            # does not synthesise a bogus target acceleration spike.
            self._last_ref_vel = (0.0, 0.0)
            self._last_ref_time = None

        diff = "\n".join(
            f"  {name:24s} {getattr(old_cfg, name)!r} -> {getattr(new_cfg, name)!r}"
            for name in sorted(overlay)
        )
        self.get_logger().info(f"Live config update:\n{diff}")

    # ============================================================ callbacks --

    def _on_ball_state(self, msg: BallStateMsg) -> None:
        self._last_ball_msg = msg
        self._last_ball_time = time.monotonic()
        if msg.tracking_valid:
            self._last_tracking_valid_time = self._last_ball_time

    # ====================================================== control step --

    def _on_step(self) -> None:
        now = time.monotonic()
        dt = self._tick_dt(now)

        # Hold off PID engagement until the startup home move has had time
        # to complete. ``is_busy()`` is a no-op since the trajectory client
        # switched from action goals to topic streaming, but the call is
        # left in to keep the API symmetric with the action-based version.
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

        # Gate on ``tracking_valid``, NOT ``ball_found``. The latter is true
        # only on a fresh CV detection, which drops out for several frames
        # during motion-blur on fast rolls. ``tracking_valid`` is also true
        # while the tracker's KF is coasting within its window, so the PID
        # keeps acting on a propagated estimate instead of freezing the
        # last commanded tilt right when braking is due. Past the coast
        # window the safety branch above (missing_ball_grace_s) brings us
        # home before stale data can corrupt the integrator.
        if not msg.tracking_valid:
            return

        # KF coast: ball_found=False but tracking_valid=True. One log on
        # entry, one on exit; never per-tick.
        if not msg.ball_found:
            self._warn_sticky("KF coast", now, "no fresh ball detection (motion blur?)")
        else:
            self._clear_sticky("KF coast", now)

        vx_clip = velocity_clip(msg.vx, self._cfg.velocity_clip_mps)
        vy_clip = velocity_clip(msg.vy, self._cfg.velocity_clip_mps)
        # Extrapolate the ball position forward to the current control tick
        # using the last reported velocity. /ball_state runs at camera rate
        # (~30 Hz); without this the PID would react to a position up to one
        # camera period stale (exactly the "slow wrist" the operator sees).
        elapsed_since_msg = max(0.0, now - self._last_ball_time)
        ball = BallState(
            x=msg.x + vx_clip * elapsed_since_msg,
            y=msg.y + vy_clip * elapsed_since_msg,
            vx=vx_clip,
            vy=vy_clip,
            timestamp=now,
            valid=True,
        )

        target_pos, target_vel, target_acc = self._reference_at(now)
        self._publish_reference_target(target_pos)

        ux, uy = self._pid.step(
            ball=ball,
            target_pos=target_pos,
            target_vel=target_vel,
            target_acc=target_acc,
            dt=dt,
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

        ux = deadband(ux, threshold=self._cfg.deadband_m)
        uy = deadband(uy, threshold=self._cfg.deadband_m)
        omega_des = self._project_to_omega(ux, uy, plate_mat)

        # Solve J_rot * dq = omega_des on the free joints. Mirrors
        # JointActuator.apply: the lstsq output is an angle (radians), NOT
        # a rate, so we do not scale by dt.
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
                f"ball=({msg.x:+.3f},{msg.y:+.3f}) "
                f"|omega|={omega_norm:.3f}  |dq|={dq_norm:.3f}  drift={drift:.3f} rad"
            )

        self._traj_client.send(
            q_target.tolist(),
            trajectory_time_s=self._cfg.trajectory_time_s,
        )
        self._publish_plate_cmd(omega_des)

    # ========================================================= internals --

    def _tick_dt(self, now: float) -> float:
        """Step ``dt`` clamped to ``[1e-3, max_dt_s]`` to harden against stalls."""
        raw = now - self._last_step_time
        self._last_step_time = now
        if raw <= 0.0:
            return 1e-3
        # Scalar min/max is ~10x cheaper than np.clip for a Python float.
        return min(max(raw, 1e-3), self._cfg.max_dt_s)

    # ----------------------------------------------------- sticky-log helpers --

    def _warn_sticky(self, name: str, now: float, message) -> None:
        """Edge-triggered + 1 Hz throttled WARNING.

        Call every tick the condition is True. First call after a clear
        (or at startup) logs once at WARN. While the condition persists
        the message re-emits at ``_PERSISTENT_LOG_PERIOD_S`` with an
        "active for Xs" suffix. ``message`` may be a string or a zero-arg
        callable returning a string; the lambda form defers the f-string
        until the throttle decides to emit.
        """
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
        """Log INFO once if a previously-active sticky warning has cleared."""
        state = self._sticky_warnings.pop(name, None)
        if state is not None:
            entered_at, _ = state
            self.get_logger().info(f"{name}: cleared after {now - entered_at:.1f}s")

    def _reference_at(
        self, now: float
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Sample the reference and finite-difference its velocity for kff."""
        x_ref, y_ref, vx_ref, vy_ref = self._reference.evaluate(now)

        if self._last_ref_time is None:
            tax = tay = 0.0
        else:
            dt_ref = max(now - self._last_ref_time, 1e-3)
            tax = (vx_ref - self._last_ref_vel[0]) / dt_ref
            tay = (vy_ref - self._last_ref_vel[1]) / dt_ref

        self._last_ref_vel = (float(vx_ref), float(vy_ref))
        self._last_ref_time = now
        return (
            (float(x_ref), float(y_ref)),
            (float(vx_ref), float(vy_ref)),
            (float(tax), float(tay)),
        )

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

        # Time since the last ``tracking_valid=True`` frame, NOT since the
        # last received message. /ball_state keeps publishing at camera
        # rate with ``tracking_valid=False`` while the ball detector is
        # failing, so a message-age check would never trip and the
        # controller would silently skip every tick. The previous
        # implementation had this bug; the visible symptom was "robot
        # stops actuating for long stretches with no log line".
        if self._last_tracking_valid_time is None:
            tracking_age = msg_age  # never been valid -> treat as message age
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
        """Project a plate-frame command to a world-frame tilt offset.

        Mirrors ``sim.adapters.actuation.JointActuator.apply``: the result
        carries the same units as ``ux`` / ``uy`` (a tilt angle in
        radians). ``max_tilt_rad`` is enforced by zeroing the world-x /
        world-y components that would push the plate further past its
        tilt limit.
        """
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
        """``(roll_x, pitch_y)`` from a world-to-plate rotation matrix."""
        roll = float(np.arctan2(plate_mat[2, 1], plate_mat[2, 2]))
        pitch = float(np.arctan2(-plate_mat[2, 0], np.hypot(plate_mat[2, 1], plate_mat[2, 2])))
        return roll, pitch

    def _omega_to_joint_delta(self, omega_world: np.ndarray, q_ros: np.ndarray) -> np.ndarray:
        """Solve ``J_rot * dq = omega`` on the free joints. Locked entries stay zero."""
        J_rot = self._jacobian.rotational(q_ros)
        n = J_rot.shape[1]
        dq = np.zeros(n)
        if self._free_cols.size:
            dq[self._free_cols], *_ = np.linalg.lstsq(
                J_rot[:, self._free_cols], omega_world, rcond=None
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
        """Publish the commanded plate tilt for downstream consumers (tracker KF).

        The world frame is whatever ``world_frame`` resolves to (typically
        ``base_link``); the tracker's KF interprets the message components
        as tilts in the table frame, which only matches when the world and
        table frames share their x/y axes (true for the current bring-up).
        Components are the small-angle approximation of the desired tilt:
        roll about world-x and pitch about world-y, in radians.
        """
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._cfg.plate_frame
        msg.vector.x = float(omega_des[0])
        msg.vector.y = float(omega_des[1])
        msg.vector.z = 0.0
        self._pub_plate_cmd.publish(msg)

    def _return_to_home(self) -> None:
        """Per-tick exponential decay of dq toward zero.

        Fast path so a transient safety condition doesn't latch
        ``is_busy()`` and gate the PID off after the condition clears.
        """
        self._pid.reset()
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
        node = BallBalanceController()
        # Multi-threaded so the 500 Hz control timer is not serialised
        # behind the UR's 500 Hz /joint_states callback. Three threads
        # cover the control + trajectory_client mutex group, the reentrant
        # subscriber group (ball_state, joint_states), and the node's
        # default mutex group used by the TF listener.
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

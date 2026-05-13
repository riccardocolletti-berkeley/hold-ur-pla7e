"""ROS 2 wrapper around the standalone vision tracker.

Owns the camera, drives the per-frame pipeline at the configured camera
FPS, and publishes the latest ball state on ``/ball_state``. The
controller is expected to publish its commanded plate tilt on
``/plate_cmd`` (``geometry_msgs/Vector3Stamped``); the tilt is fed into
the Kalman filter as a deterministic gravity component so the predictor
stays accurate across short camera dropouts.

Tracking math lives in :mod:`vision`. This module is ROS plumbing only:
parameters, publishers, subscribers, and a thin loop around
``TrackingPipeline``-style components. The building blocks are
instantiated directly here (not via the standalone ``run`` loop) so the
camera read and the publish timer are both driven by the ROS executor.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from ball_tracker_msgs.msg import BallState
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter

from ballplate.hardware import default_path as hardware_default_path
from ballplate.hardware import load as load_hardware
from ballplate.trajectories import (
    Circle,
    DrawnPath,
    Figure8,
    Reference,
    Stationary,
)
from vision.tracker.ball_detector import BallDetector
from vision.tracker.kalman_filter import BallKalmanFilter, tilt_to_accel
from vision.tracker.table_frame import TableFrame

#: Slop allowed past the plate edge before a detection is rejected. Filters
#: out HSV blobs picked up *outside* the plate (e.g. when the camera FOV
#: exposes more than the plate area, as happened when we moved from 4:3
#: 640x480 to 16:9 960x540).
_DETECTION_BOUNDS_MARGIN_M: float = 0.030

#: Frames discarded at startup so the camera's auto-exposure settles.
_CAMERA_WARMUP_FRAMES = 30

#: Allowed reference shapes (used by config validation).
_REFERENCE_SHAPES = ("stationary", "circle", "figure8", "drawn")

#: Hide the moving target marker if no /reference_target message has arrived
#: this recently (e.g. controller is in return-to-home / safety recovery).
_TARGET_STALE_SECONDS = 0.5

#: Reference-trajectory parameters that the camera overlay reads each frame.
#: Edits to these via ``ros2 param set`` or via the YAML watcher take effect
#: on the next debug-frame draw.
_LIVE_TUNABLE_PARAMS = frozenset(
    {
        "reference_shape",
        "reference_radius_m",
        "reference_period_s",
        "hold_x_m",
        "hold_y_m",
        "drawn_file",
        "reference_overlay_samples",
    }
)

#: Camera/calibration/timer fields that are wired into ``cv2.VideoCapture``
#: or ``create_timer`` at construction. Live changes are rejected.
_INIT_ONLY_PARAMS = frozenset(
    {
        "camera_index",
        "camera_width",
        "camera_height",
        "camera_fps",
        "show_debug",
        "calibration_file",
        "plate_cmd_timeout_s",
        "use_plate_cmd",
    }
)

_PARAMS_FILE_PATH_PARAM = "params_file_path"
_REFERENCE_FILE_PATH_PARAM = "reference_file_path"
_PARAM_WATCH_PERIOD_S = 1.0


def _read_node_params(path: str, node_name: str) -> dict[str, object]:
    """Pull the merged ``ros__parameters`` dict for ``node_name`` from a YAML."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, object] = {}
    for key in ("/**", node_name):
        section = data.get(key)
        if not isinstance(section, dict):
            continue
        params = section.get("ros__parameters")
        if isinstance(params, dict):
            out.update(params)
    return out


def _coerce_for_type(value: object, ptype: Parameter.Type) -> object | None:
    try:
        if ptype == Parameter.Type.DOUBLE:
            return float(value)  # type: ignore[arg-type]
        if ptype == Parameter.Type.INTEGER:
            return int(value)  # type: ignore[arg-type]
        if ptype == Parameter.Type.STRING:
            return str(value)
        if ptype == Parameter.Type.BOOL:
            return bool(value)
    except (TypeError, ValueError):
        return None
    return value


@dataclass(frozen=True)
class TrackerConfig:
    """Snapshot of every tracker parameter, frozen at construction.

    Built once at node init, validated, and logged in full so that a stale
    ``install/share`` YAML or a forgotten launch param is visible at startup.
    Mirrors the controller's pattern; in particular the ``reference_*`` block
    must agree with the controller's, otherwise the tracker's KF gravity
    input and debug overlay describe a different trajectory than the one the
    arm is actually tracking.
    """

    camera_index: int
    camera_width: int
    camera_height: int
    camera_fps: int
    show_debug: bool
    calibration_file: str
    plate_cmd_timeout_s: float
    use_plate_cmd: bool

    reference_shape: str
    reference_radius_m: float
    reference_period_s: float
    hold_x_m: float
    hold_y_m: float
    drawn_file: str
    reference_overlay_samples: int

    def problems(self) -> list[str]:
        """Return a list of validation failures (empty == OK)."""
        checks: list[tuple[bool, str]] = [
            (self.camera_index >= 0, f"camera_index must be >= 0 (got {self.camera_index})"),
            (self.camera_width > 0, f"camera_width must be > 0 (got {self.camera_width})"),
            (self.camera_height > 0, f"camera_height must be > 0 (got {self.camera_height})"),
            (self.camera_fps > 0, f"camera_fps must be > 0 (got {self.camera_fps})"),
            (
                self.plate_cmd_timeout_s > 0.0,
                f"plate_cmd_timeout_s must be > 0 (got {self.plate_cmd_timeout_s})",
            ),
            (
                self.reference_shape in _REFERENCE_SHAPES,
                f"reference_shape must be one of {_REFERENCE_SHAPES} (got {self.reference_shape!r})",
            ),
            (
                self.reference_radius_m >= 0.0,
                f"reference_radius_m must be >= 0 (got {self.reference_radius_m})",
            ),
            (
                self.reference_period_s > 0.0,
                f"reference_period_s must be > 0 (got {self.reference_period_s})",
            ),
            (
                self.reference_overlay_samples >= 4,
                f"reference_overlay_samples must be >= 4 (got {self.reference_overlay_samples})",
            ),
        ]
        if self.calibration_file:
            checks.append(
                (
                    Path(self.calibration_file).is_file(),
                    f"calibration_file does not exist: {self.calibration_file!r}",
                )
            )
        if self.reference_shape == "drawn":
            checks.append(
                (bool(self.drawn_file), "reference_shape='drawn' requires non-empty drawn_file")
            )
        return [msg for ok, msg in checks if not ok]

    def validate(self) -> None:
        problems = self.problems()
        if problems:
            raise SystemExit("Tracker config failed validation:\n  - " + "\n  - ".join(problems))


class BallTrackerNode(Node):
    """Publishes ``ball_tracker_msgs/BallState`` at the camera frame rate."""

    def __init__(self) -> None:
        super().__init__("ball_tracker")

        # ---------------------------------------------------------- params --
        # Type-only declarations: no in-code defaults. The launch file MUST
        # provide every value (typically by loading a YAML), or ``_p`` raises
        # at startup.
        self.declare_parameter("camera_index", Parameter.Type.INTEGER)
        self.declare_parameter("camera_width", Parameter.Type.INTEGER)
        self.declare_parameter("camera_height", Parameter.Type.INTEGER)
        self.declare_parameter("camera_fps", Parameter.Type.INTEGER)
        self.declare_parameter("show_debug", Parameter.Type.BOOL)
        self.declare_parameter("calibration_file", Parameter.Type.STRING)
        # If the latest /plate_cmd is older than this, the control input drops
        # to zero and the KF degrades to a constant-velocity predictor.
        self.declare_parameter("plate_cmd_timeout_s", Parameter.Type.DOUBLE)
        # When false, the KF runs as a pure constant-velocity predictor and the
        # /plate_cmd subscription is not created.
        self.declare_parameter("use_plate_cmd", Parameter.Type.BOOL)

        # Reference-trajectory overlay parameters. Mirror the controller's
        # `reference_*` params so the debug window can show the desired trace
        # the controller is actually trying to track.
        self.declare_parameter("reference_shape", Parameter.Type.STRING)
        self.declare_parameter("reference_radius_m", Parameter.Type.DOUBLE)
        self.declare_parameter("reference_period_s", Parameter.Type.DOUBLE)
        self.declare_parameter("hold_x_m", Parameter.Type.DOUBLE)
        self.declare_parameter("hold_y_m", Parameter.Type.DOUBLE)
        self.declare_parameter("drawn_file", Parameter.Type.STRING)
        self.declare_parameter("reference_overlay_samples", Parameter.Type.INTEGER)

        # Optional: paths the live-reload watcher should poll. The launch
        # file sets these to the *source* YAMLs (not install/share copies)
        # so editor saves are reflected within one watcher tick. Defaulted
        # to "" so launch files that omit them still launch.
        self.declare_parameter(_PARAMS_FILE_PATH_PARAM, "")
        self.declare_parameter(_REFERENCE_FILE_PATH_PARAM, "")

        self._cfg = self._init_config()

        # ---------------------------------------------------------- camera --
        self._cap = cv2.VideoCapture(self._cfg.camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.camera_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.camera_fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._cfg.camera_index}")
        for _ in range(_CAMERA_WARMUP_FRAMES):
            self._cap.read()

        # ----------------------------------------------------- calibration --
        self._camera_matrix: np.ndarray | None = None
        self._dist_coeffs: np.ndarray | None = None
        if self._cfg.calibration_file:
            self._load_calibration(Path(self._cfg.calibration_file))

        # ----------------------------------------------------- components --
        self._table = TableFrame()
        self._ball = BallDetector()
        self._kf = BallKalmanFilter(dt=1.0 / self._cfg.camera_fps)
        self._prev_time = time.time()
        self._coast_frames = 0

        # Plate bounds (plate-frame metres) used to reject HSV blobs that
        # map outside the plate. Loaded from the shared hardware spec so
        # any change to plate.size flows through automatically.
        hw = load_hardware(hardware_default_path())
        self._plate_bound_x = hw.plate.size[0] * 0.5 + _DETECTION_BOUNDS_MARGIN_M
        self._plate_bound_y = hw.plate.size[1] * 0.5 + _DETECTION_BOUNDS_MARGIN_M

        # ----------------------------------------------------- reference --
        self._reference: Reference = self._build_reference_from(self._cfg)
        self._ref_samples_table = self._sample_reference_path(
            self._reference, self._cfg.reference_overlay_samples
        )

        # ---------------------------------------------------------- ROS IO --
        self._pub_state = self.create_publisher(BallState, "ball_state", 10)

        # Last commanded plate tilt; cached here and pushed to the KF every
        # frame as a control input. ``stamp`` carries the message timestamp
        # and is compared against the current ROS clock to detect staleness.
        self._plate_angles: tuple[float, float] = (0.0, 0.0)
        self._plate_cmd_stamp = None
        self._sub_plate_cmd = None
        if self._cfg.use_plate_cmd:
            self._sub_plate_cmd = self.create_subscription(
                Vector3Stamped, "/plate_cmd", self._on_plate_cmd, 10
            )

        # Live reference point published by the controller (PID or MPC). The
        # tracker overlays this on the camera image instead of re-evaluating
        # the reference itself: the controller times the path on
        # ``time.monotonic()`` while the tracker would naturally use
        # ``time.time()``, so a locally sampled target sits at an arbitrary
        # fixed phase offset from the controller's actual setpoint.
        self._latest_target_xy: tuple[float, float] | None = None
        self._latest_target_time: float | None = None
        self.create_subscription(
            PointStamped,
            "/reference_target",
            self._on_reference_target,
            10,
        )

        self._timer = self.create_timer(1.0 / self._cfg.camera_fps, self._on_timer)

        self._init_live_reload()

        self.get_logger().info(
            f"Ball tracker started: camera={self._cfg.camera_index} "
            f"{self._cfg.camera_width}x{self._cfg.camera_height}@{self._cfg.camera_fps}fps"
        )

    # ============================================================ config --

    def _p(self, name: str):
        param = self.get_parameter(name)
        if param.type_ == Parameter.Type.NOT_SET:
            raise SystemExit(
                f"Required parameter '{name}' is not set. The launch file "
                f"must load the tracker params YAML (no in-code fallbacks)."
            )
        return param.value

    def _init_config(self) -> TrackerConfig:
        cfg = TrackerConfig(
            camera_index=int(self._p("camera_index")),
            camera_width=int(self._p("camera_width")),
            camera_height=int(self._p("camera_height")),
            camera_fps=int(self._p("camera_fps")),
            show_debug=bool(self._p("show_debug")),
            calibration_file=str(self._p("calibration_file")),
            plate_cmd_timeout_s=float(self._p("plate_cmd_timeout_s")),
            use_plate_cmd=bool(self._p("use_plate_cmd")),
            reference_shape=str(self._p("reference_shape")),
            reference_radius_m=float(self._p("reference_radius_m")),
            reference_period_s=float(self._p("reference_period_s")),
            hold_x_m=float(self._p("hold_x_m")),
            hold_y_m=float(self._p("hold_y_m")),
            drawn_file=str(self._p("drawn_file")),
            reference_overlay_samples=int(self._p("reference_overlay_samples")),
        )
        cfg.validate()
        lines = ["Tracker config (loaded once at init):"]
        for f in fields(cfg):
            lines.append(f"  {f.name:26s} = {getattr(cfg, f.name)!r}")
        self.get_logger().info("\n".join(lines))
        return cfg

    # ========================================================= live reload --

    def _init_live_reload(self) -> None:
        """Wire on-set-parameters callback and (optionally) the YAML watcher.

        Mirrors the controller / visualizer pattern so a save of
        ``reference.yaml`` updates the camera overlay's path within one
        watcher tick. Camera / calibration / timer fields are init-only;
        editing them via ``ros2 param set`` returns a "needs relaunch"
        rejection rather than silently being ignored.
        """
        self.add_on_set_parameters_callback(self._on_set_parameters)

        params_path = str(self.get_parameter(_PARAMS_FILE_PATH_PARAM).value)
        ref_path = str(self.get_parameter(_REFERENCE_FILE_PATH_PARAM).value)
        watched: list[str] = [p for p in (params_path, ref_path) if p]
        if not watched:
            self.get_logger().info(
                "Live YAML watcher disabled (no params_file_path / "
                "reference_file_path set). ros2 param set still works."
            )
            return

        self._param_file_mtimes: dict[str, float] = {p: self._safe_mtime(p) for p in watched}
        self.get_logger().info(
            "Live YAML watcher: polling "
            + ", ".join(repr(p) for p in watched)
            + f" every {_PARAM_WATCH_PERIOD_S:.1f}s"
        )
        self.create_timer(_PARAM_WATCH_PERIOD_S, self._poll_param_files)

    @staticmethod
    def _safe_mtime(path: str) -> float:
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def _poll_param_files(self) -> None:
        changed: list[str] = []
        for path, prev in list(self._param_file_mtimes.items()):
            mtime = self._safe_mtime(path)
            if mtime == 0.0 or mtime == prev:
                continue
            self._param_file_mtimes[path] = mtime
            changed.append(path)
        if not changed:
            return

        merged: dict[str, object] = {}
        for path in changed:
            try:
                merged.update(_read_node_params(path, self.get_name()))
            except yaml.YAMLError as exc:
                self.get_logger().warning(f"YAML watcher: parse error in {path!r}: {exc}")
                return
            except OSError as exc:
                self.get_logger().warning(f"YAML watcher: read failed on {path!r}: {exc}")
                return

        proposed: list[Parameter] = []
        for name, raw in merged.items():
            if name not in _LIVE_TUNABLE_PARAMS:
                continue
            if not self.has_parameter(name):
                continue
            existing = self.get_parameter(name)
            coerced = _coerce_for_type(raw, existing.type_)
            if coerced is None:
                self.get_logger().warning(
                    f"YAML watcher: cannot coerce {name}={raw!r} "
                    f"to declared type {existing.type_}"
                )
                continue
            proposed.append(Parameter(name, existing.type_, coerced))
        if not proposed:
            return

        result = self.set_parameters_atomically(proposed)
        if not result.successful:
            self.get_logger().warning(f"YAML reload from {changed} rejected: {result.reason}")

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        actual_changes = [
            p
            for p in params
            if self.has_parameter(p.name) and self.get_parameter(p.name).value != p.value
        ]
        if not actual_changes:
            return SetParametersResult(successful=True)

        init_only = sorted({p.name for p in actual_changes if p.name in _INIT_ONLY_PARAMS})
        if init_only:
            return SetParametersResult(
                successful=False,
                reason=(
                    f"{init_only} cannot be live-updated "
                    "(camera/timer wired at startup); relaunch the node."
                ),
            )

        config_field_names = {f.name for f in fields(TrackerConfig)}
        overlay = {p.name: p.value for p in actual_changes if p.name in config_field_names}
        if not overlay:
            return SetParametersResult(successful=True)

        kwargs: dict[str, object] = {}
        for f in fields(TrackerConfig):
            kwargs[f.name] = overlay.get(f.name, getattr(self._cfg, f.name))
        kwargs["reference_shape"] = str(kwargs["reference_shape"])
        kwargs["drawn_file"] = str(kwargs["drawn_file"])
        kwargs["reference_overlay_samples"] = int(kwargs["reference_overlay_samples"])  # type: ignore[arg-type]

        try:
            candidate = TrackerConfig(**kwargs)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            return SetParametersResult(
                successful=False,
                reason=f"could not assemble TrackerConfig: {exc}",
            )

        problems = candidate.problems()
        if problems:
            return SetParametersResult(
                successful=False,
                reason="validation failed: " + "; ".join(problems),
            )

        # Build the new reference + overlay points before swapping any state.
        # A bad ``drawn_file`` raises here and the running overlay is preserved.
        try:
            new_reference = self._build_reference_from(candidate)
            new_samples = self._sample_reference_path(
                new_reference,
                candidate.reference_overlay_samples,
            )
        except Exception as exc:
            return SetParametersResult(
                successful=False,
                reason=f"reference build failed: {exc}",
            )

        old_cfg = self._cfg
        self._cfg = candidate
        self._reference = new_reference
        self._ref_samples_table = new_samples

        diff = "\n".join(
            f"  {name:26s} {getattr(old_cfg, name)!r} → {getattr(candidate, name)!r}"
            for name in sorted(overlay)
        )
        self.get_logger().info(f"Live config update:\n{diff}")
        return SetParametersResult(successful=True)

    # ============================================================ callbacks --

    def _on_plate_cmd(self, msg: Vector3Stamped) -> None:
        self._plate_angles = (float(msg.vector.x), float(msg.vector.y))
        self._plate_cmd_stamp = msg.header.stamp

    def _on_reference_target(self, msg: PointStamped) -> None:
        self._latest_target_xy = (float(msg.point.x), float(msg.point.y))
        self._latest_target_time = time.monotonic()

    def _current_control(self) -> tuple[float, float]:
        """Return the in-plane acceleration from the latest fresh tilt command."""
        if not self._cfg.use_plate_cmd or self._plate_cmd_stamp is None:
            return 0.0, 0.0
        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = int(self._plate_cmd_stamp.sec) * 1_000_000_000 + int(
            self._plate_cmd_stamp.nanosec
        )
        age_s = (now_ns - stamp_ns) * 1e-9
        if age_s > self._cfg.plate_cmd_timeout_s:
            return 0.0, 0.0
        return tilt_to_accel(*self._plate_angles)

    def _on_timer(self) -> None:
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warning("Camera read failed.")
            return

        if self._camera_matrix is not None and self._dist_coeffs is not None:
            frame = cv2.undistort(frame, self._camera_matrix, self._dist_coeffs)

        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now

        self._kf.set_control(*self._current_control())

        markers_ok = self._table.update(frame)
        detection = self._ball.detect(frame)

        msg = BallState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "plate_center"
        msg.markers_found = self._table.markers_found
        msg.ball_found = False
        msg.tracking_valid = False
        msg.pixel_u = 0.0
        msg.pixel_v = 0.0

        # Validate the detection: must produce a position within the plate
        # bounds. Otherwise the HSV blob is some off-plate distractor (e.g.
        # something green in the FOV's margins) and we treat it as a missed
        # detection so the KF coasts instead of poisoning the controller.
        ball_position: tuple[float, float] | None = None
        if detection is not None and markers_ok:
            u, v, _ = detection
            msg.pixel_u = u
            msg.pixel_v = v
            table_pos = self._table.pixel_to_table(u, v)
            if (
                table_pos is not None
                and abs(table_pos[0]) <= self._plate_bound_x
                and abs(table_pos[1]) <= self._plate_bound_y
            ):
                ball_position = table_pos

        if ball_position is not None:
            tx, ty = ball_position
            if not self._kf.initialized:
                self._kf.reset(tx, ty)
            else:
                self._kf.predict(dt=dt)
                self._kf.update(tx, ty)
            msg.ball_found = True
            msg.tracking_valid = True
            msg.x, msg.y = self._kf.get_position()
            msg.vx, msg.vy = self._kf.get_velocity()
            self._coast_frames = 0
        elif self._kf.initialized:
            self._coast_frames += 1
            # After ~0.5 s with no detection the KF velocity carries the
            # estimated position far outside the plate, causing spurious
            # edge rescues.  Force re-initialisation on the next real
            # detection instead of letting it coast indefinitely.
            if self._coast_frames > 15:
                self._kf.initialized = False
                self._coast_frames = 0
            else:
                self._kf.predict(dt=dt)
                msg.x, msg.y = self._kf.get_position()
                msg.vx, msg.vy = self._kf.get_velocity()
                # Estimate is from a Kalman coast; usable for control
                # within the 15-frame window, but ``ball_found`` stays
                # false so the visualizer hides the stale marker.
                msg.tracking_valid = True

        self._pub_state.publish(msg)

        if self._cfg.show_debug:
            self._draw_debug(frame, msg)

    # ============================================================== helpers --

    def _draw_reference_overlay(self, frame: np.ndarray) -> None:
        """Overlay the desired-trace path and the current target on the frame."""
        if self._table.homography is None:
            return

        colour = (0, 200, 255)  # amber, distinct from the green ball circle

        # Static path (one full period), skipped for Stationary which is a point.
        if self._ref_samples_table.shape[0] >= 2:
            pts: list[tuple[float, float]] = []
            for x, y in self._ref_samples_table:
                p = self._table.table_to_pixel(float(x), float(y))
                if p is not None:
                    pts.append(p)
            if len(pts) >= 2:
                arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
                # Closed for parametric loops (Circle/Figure8); open otherwise.
                is_closed = isinstance(self._reference, Circle | Figure8)
                cv2.polylines(frame, [arr], is_closed, colour, 2, cv2.LINE_AA)

        # Current desired target (a moving cross). Driven by the controller's
        # /reference_target topic. The tracker must NOT re-evaluate the
        # reference on its own clock (the controller times the path on
        # ``time.monotonic()`` while ``time.time()`` is wall-clock seconds
        # since the epoch, so a locally sampled target sits at an arbitrary
        # fixed phase offset from the controller's actual setpoint).
        target = self._latest_target_xy
        last = self._latest_target_time
        if (
            target is not None
            and last is not None
            and time.monotonic() - last <= _TARGET_STALE_SECONDS
        ):
            target_px = self._table.table_to_pixel(float(target[0]), float(target[1]))
            if target_px is not None:
                u, v = int(target_px[0]), int(target_px[1])
                cv2.drawMarker(frame, (u, v), colour, cv2.MARKER_CROSS, 16, 2)

    @staticmethod
    def _build_reference_from(cfg: TrackerConfig) -> Reference:
        """Pure builder: a Reference for ``cfg`` with no node state touched."""
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
        raise ValueError(f"Unknown reference_shape: {shape!r}")

    @staticmethod
    def _sample_reference_path(reference: Reference, n: int) -> np.ndarray:
        """Pre-sample one period of ``reference`` for the static overlay."""
        if isinstance(reference, Stationary):
            return np.empty((0, 2), dtype=np.float64)
        period = float(reference.period)
        ts = np.linspace(0.0, period, max(n, 2), endpoint=False)
        return np.array(
            [reference.evaluate(float(t))[:2] for t in ts],
            dtype=np.float64,
        )

    def _load_calibration(self, path: Path) -> None:
        try:
            data = np.load(str(path))
            self._camera_matrix = data["camera_matrix"]
            self._dist_coeffs = data["dist_coeffs"]
            self.get_logger().info(f"Loaded calibration from {path}.")
        except Exception as exc:
            self.get_logger().warning(f"Failed to load calibration: {exc}")

    def _draw_debug(self, frame: np.ndarray, msg: BallState) -> None:
        frame = self._table.draw_debug(frame)
        self._draw_reference_overlay(frame)
        if msg.pixel_u != 0.0 or msg.pixel_v != 0.0:
            cv2.circle(frame, (int(msg.pixel_u), int(msg.pixel_v)), 10, (0, 255, 0), 2)

        lines = [f"Markers: {msg.markers_found}"]
        if msg.ball_found or self._kf.initialized:
            lines.append(f"Pos: ({msg.x * 1000.0:+6.1f}, {msg.y * 1000.0:+6.1f}) mm")
            lines.append(f"Vel: ({msg.vx * 1000.0:+6.1f}, {msg.vy * 1000.0:+6.1f}) mm/s")
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.imshow("Ball Tracker", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self._cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BallTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

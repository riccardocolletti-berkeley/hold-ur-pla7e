"""RViz marker publisher for the ball-on-plate stack.

Subscribes to ``/ball_state`` (and ``/reference_target`` from the active
controller) and publishes a ``MarkerArray`` on
``/ball_balance_visualization``. Every marker is in the ``plate_center``
frame so the overlay tracks the wrist as the arm moves.

Markers:

    plate             slab matching ``ballplate.hardware``.
    reference_path    one period of the configured reference, static.
    reference_target  sphere at the point the controller is currently
                      chasing, driven by ``/reference_target`` from the
                      controller (PID or MPC). The node does not
                      re-evaluate the reference at "now" itself: the
                      controller times the path on ``time.monotonic()``
                      while ROS defaults to wall time, so a locally
                      sampled target would sit at an arbitrary fixed
                      phase offset from the controller's setpoint.
    ball              measured ball position; hidden when no fresh detection.
    ball_trail        line strip of recent ball positions, bounded by
                      ``ball_trail_seconds``.

Reference parameters are read from the controller's wildcard (``/**``)
YAML namespace so the static green path overlay matches whatever shape
the controller is tracking. The moving target dot rides on the
controller's published setpoint, not this node's local clock.
"""

from __future__ import annotations

import math
import os
import time
from collections import deque
from dataclasses import dataclass, fields

import numpy as np
import rclpy
import yaml
from ball_tracker_msgs.msg import BallState
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, PointStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from ballplate.hardware import default_path as hardware_default_path
from ballplate.hardware import load as load_hardware
from ballplate.trajectories import Circle, DrawnPath, Figure8, Reference, Stationary

_NS = "ball_balance"
_FRAME = "plate_center"

# Stable marker IDs so RViz reuses entries across publishes instead of
# spawning a new actor every tick.
_ID_PLATE = 0
_ID_BALL = 1
_ID_REFERENCE = 2
_ID_REFERENCE_TARGET = 3
_ID_BALL_TRAIL = 4

_REFERENCE_SHAPES = ("stationary", "circle", "figure8", "drawn")

#: Fields that can be hot-swapped by the on-set callback. ``publish_rate_hz``
#: is excluded because the publish timer is created with that period at init
#: and rclpy timers are not re-tunable.
_LIVE_TUNABLE_PARAMS = frozenset(
    {
        "reference_shape",
        "reference_radius_m",
        "reference_period_s",
        "hold_x_m",
        "hold_y_m",
        "drawn_file",
        "reference_overlay_samples",
        "ball_trail_seconds",
    }
)
_INIT_ONLY_PARAMS = frozenset({"publish_rate_hz"})

_PARAMS_FILE_PATH_PARAM = "params_file_path"
_REFERENCE_FILE_PATH_PARAM = "reference_file_path"
_PARAM_WATCH_PERIOD_S = 1.0

#: Hide the green target sphere if no /reference_target message has arrived
#: this recently. Picks up the controller's return-to-home / safety branches.
_TARGET_STALE_SECONDS = 0.5


def _read_node_params(path: str, node_name: str) -> dict[str, object]:
    """Load the merged ``ros__parameters`` dict for ``node_name`` from a YAML."""
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
    except (TypeError, ValueError):
        return None
    return value


@dataclass(frozen=True)
class VisualizerConfig:
    reference_shape: str
    reference_radius_m: float
    reference_period_s: float
    hold_x_m: float
    hold_y_m: float
    drawn_file: str
    reference_overlay_samples: int
    publish_rate_hz: float
    ball_trail_seconds: float

    def problems(self) -> list[str]:
        out: list[str] = []
        if self.reference_shape not in _REFERENCE_SHAPES:
            out.append(
                f"reference_shape must be one of {_REFERENCE_SHAPES} "
                f"(got {self.reference_shape!r})"
            )
        if self.reference_radius_m < 0.0:
            out.append(f"reference_radius_m must be >= 0 (got {self.reference_radius_m})")
        if self.reference_period_s <= 0.0:
            out.append(f"reference_period_s must be > 0 (got {self.reference_period_s})")
        if self.reference_overlay_samples < 4:
            out.append(
                f"reference_overlay_samples must be >= 4 (got {self.reference_overlay_samples})"
            )
        if self.publish_rate_hz <= 0.0:
            out.append(f"publish_rate_hz must be > 0 (got {self.publish_rate_hz})")
        if self.ball_trail_seconds < 0.0:
            out.append(f"ball_trail_seconds must be >= 0 (got {self.ball_trail_seconds})")
        if self.reference_shape == "drawn" and not self.drawn_file:
            out.append("reference_shape='drawn' requires non-empty drawn_file")
        return out

    def validate(self) -> None:
        problems = self.problems()
        if problems:
            raise SystemExit("Visualizer config failed validation:\n  - " + "\n  - ".join(problems))


class BallBalanceVisualizer(Node):
    """Republishes ``/ball_state`` and the active reference as RViz markers."""

    def __init__(self) -> None:
        super().__init__("ball_balance_visualizer")

        self.declare_parameter("reference_shape", Parameter.Type.STRING)
        self.declare_parameter("reference_radius_m", Parameter.Type.DOUBLE)
        self.declare_parameter("reference_period_s", Parameter.Type.DOUBLE)
        self.declare_parameter("hold_x_m", Parameter.Type.DOUBLE)
        self.declare_parameter("hold_y_m", Parameter.Type.DOUBLE)
        self.declare_parameter("drawn_file", Parameter.Type.STRING)
        self.declare_parameter("reference_overlay_samples", Parameter.Type.INTEGER)
        self.declare_parameter("publish_rate_hz", Parameter.Type.DOUBLE)
        self.declare_parameter("ball_trail_seconds", Parameter.Type.DOUBLE)
        # Optional watcher inputs. Defaulted so launch files that omit them
        # still launch, with the watcher disabled.
        self.declare_parameter(_PARAMS_FILE_PATH_PARAM, "")
        self.declare_parameter(_REFERENCE_FILE_PATH_PARAM, "")

        self._cfg = self._init_config()
        self._hw = load_hardware(hardware_default_path())
        self._reference: Reference = self._build_reference_from(self._cfg)

        # Reference path is time-invariant for every supported shape, so
        # sample it once and reuse the same Point list every publish.
        self._reference_path_points: list[Point] = self._sample_reference_path()

        self._latest_ball: BallState | None = None
        # (monotonic_seconds, x, y); pruned by age in _publish.
        self._trail: deque[tuple[float, float, float]] = deque()

        # Live target published by the controller. The visualizer must NOT
        # re-evaluate the reference itself: the controller times the path on
        # ``time.monotonic()`` while ROS clocks default to wall time, so a
        # locally-resampled target sits at an arbitrary fixed phase offset
        # from the point the controller is actually chasing.
        self._latest_target_xy: tuple[float, float] | None = None
        self._latest_target_time: float | None = None

        # DELETEALL the namespace once on start-up so a leftover marker from
        # a previous run cannot linger in RViz.
        self._cleared_initial_markers = False

        marker_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._pub = self.create_publisher(
            MarkerArray,
            "/ball_balance_visualization",
            marker_qos,
        )
        self.create_subscription(
            BallState,
            "/ball_state",
            self._on_ball_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointStamped,
            "/reference_target",
            self._on_reference_target,
            10,
        )
        self.create_timer(1.0 / self._cfg.publish_rate_hz, self._publish)

        self._init_live_reload()

        self.get_logger().info(
            f"Visualizer up: plate={self._hw.plate.size} m, "
            f"ball_radius={self._hw.ball.radius} m, "
            f"reference={self._cfg.reference_shape}"
        )

    # -- config ---------------------------------------------------------------

    def _p(self, name: str):
        param = self.get_parameter(name)
        if param.type_ == Parameter.Type.NOT_SET:
            raise SystemExit(
                f"Required parameter '{name}' is not set. The launch file must "
                f"load the controller params YAML."
            )
        return param.value

    def _init_config(self) -> VisualizerConfig:
        cfg = VisualizerConfig(
            reference_shape=str(self._p("reference_shape")),
            reference_radius_m=float(self._p("reference_radius_m")),
            reference_period_s=float(self._p("reference_period_s")),
            hold_x_m=float(self._p("hold_x_m")),
            hold_y_m=float(self._p("hold_y_m")),
            drawn_file=str(self._p("drawn_file")),
            reference_overlay_samples=int(self._p("reference_overlay_samples")),
            publish_rate_hz=float(self._p("publish_rate_hz")),
            ball_trail_seconds=float(self._p("ball_trail_seconds")),
        )
        cfg.validate()
        lines = ["Visualizer config (loaded once at init):"]
        for f in fields(cfg):
            lines.append(f"  {f.name:26s} = {getattr(cfg, f.name)!r}")
        self.get_logger().info("\n".join(lines))
        return cfg

    @staticmethod
    def _build_reference_from(cfg: VisualizerConfig) -> Reference:
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

    def _sample_reference_path(
        self,
        reference: Reference | None = None,
        n_samples: int | None = None,
    ) -> list[Point]:
        """Resample the polyline that draws the static reference path.

        Defaults to the currently-installed ``self._reference`` and
        ``self._cfg.reference_overlay_samples``; the live-reload path passes
        a fresh reference + sample count so a candidate path can be built
        before any state is swapped.
        """
        ref = reference if reference is not None else self._reference
        if isinstance(ref, Stationary):
            return []
        period = float(ref.period)
        if not math.isfinite(period) or period <= 0.0:
            return []
        n = n_samples if n_samples is not None else self._cfg.reference_overlay_samples
        z = float(self._hw.ball.radius)
        ts = np.linspace(0.0, period, n, endpoint=False)
        pts = [
            Point(x=float(x), y=float(y), z=z)
            for (x, y, _, _) in (ref.evaluate(float(t)) for t in ts)
        ]
        # Close parametric loops so the polyline meets back at its origin.
        if isinstance(ref, Circle | Figure8) and pts:
            pts.append(pts[0])
        return pts

    # -- live reload ----------------------------------------------------------

    def _init_live_reload(self) -> None:
        """Wire ``add_on_set_parameters_callback`` and the YAML mtime watcher.

        Mirrors the controller's flow so editing ``reference.yaml`` updates
        both the controller's tracked path and the green polyline drawn in
        RViz simultaneously. ``publish_rate_hz`` is rejected if changed
        because the publish timer's period is fixed at construction.
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
                    "(timer period is fixed at startup); relaunch the node."
                ),
            )

        config_field_names = {f.name for f in fields(VisualizerConfig)}
        overlay = {p.name: p.value for p in actual_changes if p.name in config_field_names}
        if not overlay:
            return SetParametersResult(successful=True)

        kwargs: dict[str, object] = {}
        for f in fields(VisualizerConfig):
            kwargs[f.name] = overlay.get(f.name, getattr(self._cfg, f.name))
        kwargs["reference_shape"] = str(kwargs["reference_shape"])
        kwargs["drawn_file"] = str(kwargs["drawn_file"])
        kwargs["reference_overlay_samples"] = int(kwargs["reference_overlay_samples"])  # type: ignore[arg-type]

        try:
            candidate = VisualizerConfig(**kwargs)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            return SetParametersResult(
                successful=False,
                reason=f"could not assemble VisualizerConfig: {exc}",
            )

        problems = candidate.problems()
        if problems:
            return SetParametersResult(
                successful=False,
                reason="validation failed: " + "; ".join(problems),
            )

        path_fields = {
            "reference_shape",
            "reference_radius_m",
            "reference_period_s",
            "hold_x_m",
            "hold_y_m",
            "drawn_file",
            "reference_overlay_samples",
        }
        rebuild_path = bool(overlay.keys() & path_fields)
        new_reference: Reference | None = None
        new_path_points: list[Point] | None = None
        if rebuild_path:
            try:
                new_reference = self._build_reference_from(candidate)
                new_path_points = self._sample_reference_path(
                    reference=new_reference,
                    n_samples=candidate.reference_overlay_samples,
                )
            except Exception as exc:
                # See controller._on_set_parameters: DrawnPath.from_file is
                # the broad-failure surface and we forward whatever it raised.
                return SetParametersResult(
                    successful=False,
                    reason=f"reference build failed: {exc}",
                )

        old_cfg = self._cfg
        self._cfg = candidate
        if new_reference is not None:
            self._reference = new_reference
        if new_path_points is not None:
            self._reference_path_points = new_path_points
            # The next ``_publish`` carries an implicit DELETEALL only at
            # startup; if the polyline shrinks and we don't clear stale
            # points, RViz will keep the old vertices. Setting the marker's
            # ``points`` list each tick already overwrites them, so no
            # extra cleanup is needed beyond the assignment above.

        diff = "\n".join(
            f"  {name:26s} {getattr(old_cfg, name)!r} → {getattr(candidate, name)!r}"
            for name in sorted(overlay)
        )
        self.get_logger().info(f"Live config update:\n{diff}")
        return SetParametersResult(successful=True)

    # -- callbacks ------------------------------------------------------------

    def _on_ball_state(self, msg: BallState) -> None:
        self._latest_ball = msg
        if msg.ball_found and self._cfg.ball_trail_seconds > 0.0:
            self._trail.append((time.monotonic(), float(msg.x), float(msg.y)))

    def _on_reference_target(self, msg: PointStamped) -> None:
        self._latest_target_xy = (float(msg.point.x), float(msg.point.y))
        self._latest_target_time = time.monotonic()

    def _publish(self) -> None:
        # Zero-stamped markers render against the latest TF, which avoids
        # transient drops when robot TF arrives slightly behind this timer.
        stamp = Time()
        markers = MarkerArray()

        if not self._cleared_initial_markers:
            markers.markers.append(self._delete_all_marker(stamp))
            self._cleared_initial_markers = True

        markers.markers.append(self._plate_marker(stamp))
        markers.markers.append(self._reference_path_marker(stamp))
        markers.markers.append(self._reference_target_marker(stamp))
        markers.markers.append(self._ball_marker(stamp))
        markers.markers.append(self._ball_trail_marker(stamp))
        self._pub.publish(markers)

    # -- markers --------------------------------------------------------------

    def _delete_all_marker(self, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = _FRAME
        m.header.stamp = stamp
        m.ns = _NS
        m.action = Marker.DELETEALL
        return m

    def _plate_marker(self, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = _FRAME
        m.header.stamp = stamp
        m.ns = _NS
        m.id = _ID_PLATE
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.frame_locked = True
        # plate_center is the top face of the plate; lower the slab so the
        # cube's top sits flush with z=0.
        m.pose.position.z = -self._hw.plate.thickness / 2.0
        m.pose.orientation.w = 1.0
        m.scale.x = float(self._hw.plate.size[0])
        m.scale.y = float(self._hw.plate.size[1])
        m.scale.z = float(self._hw.plate.thickness)
        m.color = ColorRGBA(r=0.18, g=0.42, b=0.88, a=1.0)
        return m

    def _reference_path_marker(self, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = _FRAME
        m.header.stamp = stamp
        m.ns = _NS
        m.id = _ID_REFERENCE
        m.type = Marker.LINE_STRIP
        m.frame_locked = True
        m.pose.orientation.w = 1.0
        m.scale.x = 0.003
        m.color = ColorRGBA(r=0.0, g=0.85, b=0.55, a=0.9)
        if not self._reference_path_points:
            m.action = Marker.DELETE
            return m
        m.action = Marker.ADD
        m.points = list(self._reference_path_points)
        return m

    def _reference_target_marker(self, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = _FRAME
        m.header.stamp = stamp
        m.ns = _NS
        m.id = _ID_REFERENCE_TARGET
        m.type = Marker.SPHERE
        m.frame_locked = True
        # Hide if the controller has not published a target yet, or has gone
        # quiet (e.g. it's in return-to-home / safety recovery and isn't
        # actively chasing anything). Half a second covers a few skipped
        # control ticks without flickering during normal operation.
        target = self._latest_target_xy
        last = self._latest_target_time
        if target is None or last is None or time.monotonic() - last > _TARGET_STALE_SECONDS:
            m.action = Marker.DELETE
            return m
        m.action = Marker.ADD
        m.pose.position.x = target[0]
        m.pose.position.y = target[1]
        m.pose.position.z = float(self._hw.ball.radius)
        m.pose.orientation.w = 1.0
        # ~70% of the ball diameter; visibly smaller than the ball so the
        # operator can read at a glance which sphere is which.
        d = 1.4 * float(self._hw.ball.radius)
        m.scale.x = d
        m.scale.y = d
        m.scale.z = d
        m.color = ColorRGBA(r=0.0, g=0.85, b=0.55, a=0.9)
        return m

    def _ball_marker(self, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = _FRAME
        m.header.stamp = stamp
        m.ns = _NS
        m.id = _ID_BALL
        m.type = Marker.SPHERE
        m.frame_locked = True
        if self._latest_ball is None or not self._latest_ball.ball_found:
            m.action = Marker.DELETE
            return m
        m.action = Marker.ADD
        m.pose.position.x = float(self._latest_ball.x)
        m.pose.position.y = float(self._latest_ball.y)
        m.pose.position.z = float(self._hw.ball.radius)
        m.pose.orientation.w = 1.0
        d = 2.0 * self._hw.ball.radius
        m.scale.x = d
        m.scale.y = d
        m.scale.z = d
        m.color = ColorRGBA(r=0.95, g=0.55, b=0.10, a=1.0)
        return m

    def _ball_trail_marker(self, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = _FRAME
        m.header.stamp = stamp
        m.ns = _NS
        m.id = _ID_BALL_TRAIL
        m.type = Marker.LINE_STRIP
        m.frame_locked = True
        m.pose.orientation.w = 1.0
        m.scale.x = 0.002
        m.color = ColorRGBA(r=0.95, g=0.55, b=0.10, a=0.45)

        cutoff = time.monotonic() - self._cfg.ball_trail_seconds
        while self._trail and self._trail[0][0] < cutoff:
            self._trail.popleft()

        if len(self._trail) < 2:
            m.action = Marker.DELETE
            return m
        m.action = Marker.ADD
        z = float(self._hw.ball.radius)
        m.points = [Point(x=x, y=y, z=z) for _, x, y in self._trail]
        return m


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BallBalanceVisualizer()
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

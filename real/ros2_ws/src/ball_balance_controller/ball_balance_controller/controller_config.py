"""Frozen ``ControllerConfig`` snapshot and YAML parsing helpers.

Extracted from :mod:`controller` so the live-reload code and the node
class can both depend on the dataclass without sharing the rest of the
control-loop surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from rclpy.parameter import Parameter

#: Allowed reference shapes (used by config validation).
REFERENCE_SHAPES = ("stationary", "circle", "figure8", "drawn")


def read_node_params(path: str, node_name: str) -> dict[str, object]:
    """Pull the flat ``ros__parameters`` dict for ``node_name`` from a YAML file.

    Merges the wildcard ``/**`` block (if any) under the node-specific
    block, matching ``rcl_yaml_param_parser``'s resolution order. Returns
    ``{}`` when the file has no relevant section.
    """
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


def coerce_for_type(value: object, ptype: Parameter.Type) -> object | None:
    """Coerce a YAML-loaded value to match a declared parameter type.

    Returns ``None`` when the coercion is impossible (e.g. a list where a
    scalar is required) so the caller can drop the entry instead of
    pushing a type-mismatched ``Parameter`` that rclpy would reject.
    """
    try:
        if ptype == Parameter.Type.DOUBLE:
            return float(value)  # type: ignore[arg-type]
        if ptype == Parameter.Type.INTEGER:
            return int(value)  # type: ignore[arg-type]
        if ptype == Parameter.Type.STRING:
            return str(value)
        if ptype == Parameter.Type.BOOL:
            return bool(value)
        if ptype == Parameter.Type.DOUBLE_ARRAY:
            return [float(v) for v in value]  # type: ignore[union-attr]
        if ptype == Parameter.Type.INTEGER_ARRAY:
            return [int(v) for v in value]  # type: ignore[union-attr]
        if ptype == Parameter.Type.STRING_ARRAY:
            return [str(v) for v in value]  # type: ignore[union-attr]
    except (TypeError, ValueError):
        return None
    return value


@dataclass(frozen=True)
class ControllerConfig:
    """Snapshot of every controller parameter.

    Built at node init from the ROS parameter server and rebuilt atomically
    on every accepted live edit. The control loop reads ``self._cfg`` only;
    publication is a single atomic assignment.
    """

    # Robot / kinematics
    urdf_path: str
    home_joints: tuple[float, ...]
    world_frame: str
    plate_frame: str
    locked_joints_spec: str

    # PID gains
    kp: float
    ki: float
    kd: float
    kff: float
    integral_limit: float
    error_clip_m: float

    # Safety / shaping
    max_tilt_rad: float
    max_joint_delta_rad: float
    deadband_m: float
    velocity_clip_mps: float
    max_dt_s: float

    # Timing / filtering
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

    def problems(self) -> list[str]:
        """List of validation failures (``[]`` == OK).

        Used both at init (a non-empty list aborts the node via
        :meth:`validate`) and during live reload (a non-empty list
        rejects the proposed parameter change without taking the node
        down).
        """
        checks: list[tuple[bool, str]] = [
            (Path(self.urdf_path).is_file(), f"urdf_path does not exist: {self.urdf_path!r}"),
            (
                len(self.home_joints) == 6,
                f"home_joints must have 6 entries, got {len(self.home_joints)}",
            ),
            (self.kp >= 0.0, f"kp must be >= 0 (got {self.kp})"),
            (self.ki >= 0.0, f"ki must be >= 0 (got {self.ki})"),
            (self.kd >= 0.0, f"kd must be >= 0 (got {self.kd})"),
            (self.integral_limit > 0.0, f"integral_limit must be > 0 (got {self.integral_limit})"),
            (self.error_clip_m > 0.0, f"error_clip_m must be > 0 (got {self.error_clip_m})"),
            (self.max_tilt_rad > 0.0, f"max_tilt_rad must be > 0 (got {self.max_tilt_rad})"),
            (
                self.max_joint_delta_rad > 0.0,
                f"max_joint_delta_rad must be > 0 (got {self.max_joint_delta_rad})",
            ),
            (self.deadband_m >= 0.0, f"deadband_m must be >= 0 (got {self.deadband_m})"),
            (
                self.velocity_clip_mps > 0.0,
                f"velocity_clip_mps must be > 0 (got {self.velocity_clip_mps})",
            ),
            (self.max_dt_s > 0.0, f"max_dt_s must be > 0 (got {self.max_dt_s})"),
            (
                self.command_rate_hz > 0.0,
                f"command_rate_hz must be > 0 (got {self.command_rate_hz})",
            ),
            (
                0.0 <= self.command_alpha <= 1.0,
                f"command_alpha must be in [0,1] (got {self.command_alpha})",
            ),
            (
                self.trajectory_time_s >= 0.0,
                f"trajectory_time_s must be >= 0 (got {self.trajectory_time_s})",
            ),
            (
                self.startup_home_time_s > 0.0,
                f"startup_home_time_s must be > 0 (got {self.startup_home_time_s})",
            ),
            (
                0.0 <= self.return_to_base_rate <= 1.0,
                f"return_to_base_rate must be in [0,1] (got {self.return_to_base_rate})",
            ),
            (
                self.reference_shape in REFERENCE_SHAPES,
                f"reference_shape must be one of {REFERENCE_SHAPES} (got {self.reference_shape!r})",
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
                self.stale_ball_state_s > 0.0,
                f"stale_ball_state_s must be > 0 (got {self.stale_ball_state_s})",
            ),
            (
                self.missing_ball_grace_s > 0.0,
                f"missing_ball_grace_s must be > 0 (got {self.missing_ball_grace_s})",
            ),
            (
                self.markers_min_visible >= 0,
                f"markers_min_visible must be >= 0 (got {self.markers_min_visible})",
            ),
        ]
        if self.reference_shape == "drawn":
            checks.append(
                (bool(self.drawn_file), "reference_shape='drawn' requires non-empty drawn_file")
            )
        return [msg for ok, msg in checks if not ok]

    def validate(self) -> None:
        """Abort the process on the first bad config (init-time only)."""
        problems = self.problems()
        if problems:
            raise SystemExit("Controller config failed validation:\n  - " + "\n  - ".join(problems))

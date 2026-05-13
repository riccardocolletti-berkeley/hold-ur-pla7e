"""Live YAML reload + on-set-parameters callback for :class:`BallBalanceController`.

Mixed into the node class so the param watcher and the validation pipeline
live in their own file. The mixin expects the host to provide:

* ``self._cfg`` of type :class:`ControllerConfig`
* ``self._cb_subs`` callback group for the watch timer
* ``self._apply_config(new_cfg, overlay, new_reference)``: swap config and
  rebind dependent actors (PID gains, reference trajectory).
* ``self._build_reference_from(cfg)``: build a :class:`Reference` from a
  config snapshot (raises ``OSError`` / ``ValueError`` on bad ``drawn_file``).

The on-set-parameters callback is registered exactly once, by
:meth:`init_live_reload`, after the host has finished building all the
actors the callback might want to rebind.
"""

from __future__ import annotations

import os
from dataclasses import fields

import yaml
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter

from ball_balance_controller.controller_config import (
    ControllerConfig,
    coerce_for_type,
    read_node_params,
)

#: Parameters the closed loop reads every tick, or that can be hot-swapped
#: into existing actor objects (PID gains, reference trajectory). Edits via
#: ``ros2 param set`` or via the YAML watcher take effect on the next tick.
LIVE_TUNABLE_PARAMS = frozenset(
    {
        "kp",
        "ki",
        "kd",
        "kff",
        "integral_limit",
        "error_clip_m",
        "max_tilt_rad",
        "max_joint_delta_rad",
        "deadband_m",
        "velocity_clip_mps",
        "max_dt_s",
        "command_alpha",
        "trajectory_time_s",
        "return_to_base_rate",
        "stale_ball_state_s",
        "missing_ball_grace_s",
        "markers_min_visible",
        "reference_shape",
        "reference_radius_m",
        "reference_period_s",
        "hold_x_m",
        "hold_y_m",
        "drawn_file",
    }
)

#: Parameters wired into objects built once at startup (URDF chain, joint
#: locking, control timer period, startup-home dwell). Live changes are
#: rejected with a "needs relaunch" reason.
INIT_ONLY_PARAMS = frozenset(
    {
        "urdf_path",
        "home_joints",
        "world_frame",
        "plate_frame",
        "locked_joints_spec",
        "command_rate_hz",
        "startup_home_time_s",
    }
)

#: Parameter names used by the YAML mtime watcher itself.
PARAMS_FILE_PATH_PARAM = "params_file_path"
REFERENCE_FILE_PATH_PARAM = "reference_file_path"

#: How often the watcher stats the YAML files for mtime changes.
PARAM_WATCH_PERIOD_S = 1.0

#: Reference-related fields; a change to any of these rebuilds the trajectory
#: before swapping it in.
_REFERENCE_FIELDS = frozenset(
    {
        "reference_shape",
        "reference_radius_m",
        "reference_period_s",
        "hold_x_m",
        "hold_y_m",
        "drawn_file",
    }
)


class LiveReloadMixin:
    """Provides the YAML watcher and on-set-parameters callback.

    See module docstring for the host-class contract.
    """

    # ====================================================== entry point --

    def init_live_reload(self) -> None:
        """Register the on-set callback and start the YAML mtime watcher.

        Three paths converge on :meth:`_on_set_parameters`:
        ``ros2 param set``, ``ros2 param load``, and the 1 Hz watcher
        below. All three swap a fully-validated :class:`ControllerConfig`
        in or leave the running config untouched.
        """
        self.add_on_set_parameters_callback(self._on_set_parameters)

        params_path = str(self._p_optional(PARAMS_FILE_PATH_PARAM))
        ref_path = str(self._p_optional(REFERENCE_FILE_PATH_PARAM))
        watched: list[str] = [p for p in (params_path, ref_path) if p]
        if not watched:
            self.get_logger().info(
                "Live YAML watcher disabled (no params_file_path / "
                "reference_file_path set). ros2 param set still works."
            )
            return

        # Snapshot mtimes now so the first poll does not re-fire on values
        # the launch file already loaded into the parameter server.
        self._param_file_mtimes: dict[str, float] = {}
        for path in watched:
            self._param_file_mtimes[path] = self._safe_mtime(path)

        self.get_logger().info(
            "Live YAML watcher: polling "
            + ", ".join(repr(p) for p in watched)
            + f" every {PARAM_WATCH_PERIOD_S:.1f}s"
        )
        # Reentrant group: the watcher must not block the control timer.
        self.create_timer(
            PARAM_WATCH_PERIOD_S,
            self._poll_param_files,
            callback_group=self._cb_subs,
        )

    # ========================================================= helpers --

    def _p_optional(self, name: str) -> object:
        """Read a parameter without raising if it was declared with a default."""
        return self.get_parameter(name).value

    @staticmethod
    def _safe_mtime(path: str) -> float:
        """Mtime in seconds, or 0.0 if the file is missing/unreadable."""
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    # ===================================================== mtime watcher --

    def _poll_param_files(self) -> None:
        """Re-push params from any YAML whose mtime has advanced."""
        changed: list[str] = []
        for path, prev in list(self._param_file_mtimes.items()):
            mtime = self._safe_mtime(path)
            if mtime == 0.0 or mtime == prev:
                continue
            self._param_file_mtimes[path] = mtime
            changed.append(path)
        if not changed:
            return

        proposed = self._collect_proposed_params(changed)
        if not proposed:
            return

        result = self.set_parameters_atomically(proposed)
        if not result.successful:
            # ``_on_set_parameters`` already logged the rejection reason;
            # name the source file so the operator knows where to look.
            self.get_logger().warning(f"YAML reload from {changed} rejected: {result.reason}")

    def _collect_proposed_params(self, paths: list[str]) -> list[Parameter]:
        """Build ``Parameter`` objects for live-tunable fields found in ``paths``.

        Init-only and unknown keys are filtered silently; the watcher pushes
        tuning edits, it does not police the YAML schema.
        """
        merged: dict[str, object] = {}
        for path in paths:
            try:
                merged.update(read_node_params(path, self.get_name()))
            except yaml.YAMLError as exc:
                # Editor saved half a file; retry on the next mtime bump.
                self.get_logger().warning(f"YAML watcher: parse error in {path!r}: {exc}")
                return []
            except OSError as exc:
                self.get_logger().warning(f"YAML watcher: read failed on {path!r}: {exc}")
                return []

        out: list[Parameter] = []
        for name, raw_value in merged.items():
            if name not in LIVE_TUNABLE_PARAMS:
                continue
            if not self.has_parameter(name):
                continue
            existing = self.get_parameter(name)
            coerced = coerce_for_type(raw_value, existing.type_)
            if coerced is None:
                self.get_logger().warning(
                    f"YAML watcher: cannot coerce {name}={raw_value!r} "
                    f"to declared type {existing.type_}"
                )
                continue
            out.append(Parameter(name, existing.type_, coerced))
        return out

    # ================================================ on-set parameters --

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        """Validate and apply a parameter batch atomically.

        The whole batch either applies or none of it does (no half-updated
        config). Called by rclpy for every ``set_parameters`` /
        ``set_parameters_atomically``, whatever the source.
        """
        # Drop entries whose proposed value matches the current one;
        # ``ros2 param load`` re-pushes every key in the YAML.
        actual_changes = [p for p in params if not self._is_unchanged(p)]
        if not actual_changes:
            return SetParametersResult(successful=True)

        init_only = sorted({p.name for p in actual_changes if p.name in INIT_ONLY_PARAMS})
        if init_only:
            return SetParametersResult(
                successful=False,
                reason=(
                    f"{init_only} cannot be live-updated "
                    "(wired into kinematics / timer at startup); relaunch the node."
                ),
            )

        config_field_names = {f.name for f in fields(ControllerConfig)}
        overlay = {p.name: p.value for p in actual_changes if p.name in config_field_names}
        if not overlay:
            # The batch is non-empty but touches only watcher-internal
            # params (e.g. params_file_path). Nothing to rebuild.
            return SetParametersResult(successful=True)

        try:
            candidate = self._build_candidate_config(overlay)
        except (TypeError, ValueError) as exc:
            return SetParametersResult(
                successful=False,
                reason=f"could not assemble ControllerConfig: {exc}",
            )

        problems = candidate.problems()
        if problems:
            return SetParametersResult(
                successful=False,
                reason="validation failed: " + "; ".join(problems),
            )

        # Build the candidate reference up-front; a bad ``drawn_file`` must
        # not corrupt the running trajectory.
        rebuild_reference = bool(overlay.keys() & _REFERENCE_FIELDS)
        new_reference = None
        if rebuild_reference:
            try:
                new_reference = self._build_reference_from(candidate)
            except Exception as exc:
                # Broad catch on purpose: DrawnPath.from_file can raise
                # OSError / yaml errors / ValueError.
                return SetParametersResult(
                    successful=False,
                    reason=f"reference build failed: {exc}",
                )

        self._apply_config(candidate, overlay, new_reference)
        return SetParametersResult(successful=True)

    def _is_unchanged(self, p: Parameter) -> bool:
        """True iff ``p`` matches the value already on the parameter server."""
        if not self.has_parameter(p.name):
            return False
        current = self.get_parameter(p.name).value
        return current == p.value

    def _build_candidate_config(self, overlay: dict[str, object]) -> ControllerConfig:
        """Construct a fresh ControllerConfig: current values + proposed overlay."""
        kwargs: dict[str, object] = {}
        for f in fields(ControllerConfig):
            if f.name in overlay:
                kwargs[f.name] = overlay[f.name]
            else:
                kwargs[f.name] = getattr(self._cfg, f.name)
        # ``home_joints`` is exposed as a tuple but rclpy hands us list-like values.
        kwargs["home_joints"] = tuple(float(v) for v in kwargs["home_joints"])  # type: ignore[arg-type]
        # String-typed fields can come back as non-str when set via the CLI
        # from a YAML literal; coerce defensively.
        kwargs["reference_shape"] = str(kwargs["reference_shape"])
        kwargs["drawn_file"] = str(kwargs["drawn_file"])
        kwargs["urdf_path"] = str(kwargs["urdf_path"])
        kwargs["world_frame"] = str(kwargs["world_frame"])
        kwargs["plate_frame"] = str(kwargs["plate_frame"])
        kwargs["locked_joints_spec"] = str(kwargs["locked_joints_spec"])
        kwargs["markers_min_visible"] = int(kwargs["markers_min_visible"])  # type: ignore[arg-type]
        return ControllerConfig(**kwargs)  # type: ignore[arg-type]

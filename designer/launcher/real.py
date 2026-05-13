"""Real-robot backend: stage the trajectory, return the ``ros2 launch`` line.

The designer runs on a developer machine; the UR arm and the ROS 2 stack
live on the lab machine. This backend does not spawn a subprocess. It (1)
copies the drawn JSON (or selects a preset) into the controller's runtime
directory and (2) returns the exact ``ros2 launch ...`` command the user
runs on the lab. The only state held here is metadata about the most
recent ``start_*`` call so the UI can display the staged path and command.

The launch command depends on the selected controller (e.g. PID picks
``balance.launch.py``, MPC picks ``mpc_balance.launch.py``); the mapping
is supplied by ``designer.yaml::real.ros_launch_commands``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Monorepo root from ``designer/launcher/real.py``: parents[2].
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_CONTROLLER = "pid"


class RealLauncher:
    """Real-robot backend that stages drawings and returns a launch command."""

    backend_id = "real"
    label = "Real Robot"

    def __init__(
        self,
        drawing_target_dir: str,
        ros_launch_commands: dict[str, str],
    ):
        self._target_dir = (_PROJECT_ROOT / drawing_target_dir).resolve()
        self._target_dir.mkdir(parents=True, exist_ok=True)
        if not ros_launch_commands:
            raise ValueError(
                "ros_launch_commands must contain at least one entry "
                "(e.g. 'pid': 'ros2 launch ball_balance_controller balance.launch.py')"
            )
        self._ros_launch_commands = dict(ros_launch_commands)

        self._last_started: dict | None = None

    # ----------------------------------------------------------- lifecycle --

    def is_running(self) -> bool:
        # No local subprocess to track; "running" lives on the lab machine.
        return False

    def stop(self) -> None:
        # Nothing to kill locally; clear the cached metadata so the UI
        # stops advertising the staged command.
        self._last_started = None

    # --------------------------------------------------------------- start --

    def start_drawn(self, drawn_file: Path, *, controller: str | None = None) -> dict:
        """Stage the drawing into the runtime dir and return the launch command."""
        ctrl = self._resolve_controller(controller)
        target = self._target_dir / drawn_file.name
        shutil.copyfile(drawn_file, target)

        params = {
            "reference_shape": "drawn",
            "drawn_file": str(target),
        }
        return self._record(self._build_command(ctrl, params), staged=str(target))

    def start_preset(
        self,
        shape: str,
        *,
        radius: float | None = None,
        hold_x: float | None = None,
        hold_y: float | None = None,
        controller: str | None = None,
    ) -> dict:
        """Return the launch command pre-filled with the chosen preset."""
        ctrl = self._resolve_controller(controller)
        params: dict = {"reference_shape": shape}
        if radius is not None:
            params["reference_radius_m"] = radius
        if hold_x is not None:
            params["hold_x_m"] = hold_x
        if hold_y is not None:
            params["hold_y_m"] = hold_y
        return self._record(self._build_command(ctrl, params))

    # ------------------------------------------------------------ internals --

    def _resolve_controller(self, controller: str | None) -> str:
        """Pick the controller, falling back to ``pid`` when unset."""
        ctrl = controller or _DEFAULT_CONTROLLER
        if ctrl not in self._ros_launch_commands:
            raise ValueError(
                f"real backend has no launch command for controller={ctrl!r}; "
                f"available: {sorted(self._ros_launch_commands)}"
            )
        return ctrl

    def _build_command(self, controller: str, params: dict) -> str:
        """Append ``key:=value`` pairs to the controller's launch command."""
        parts = [self._ros_launch_commands[controller]]
        for key, value in params.items():
            parts.append(f"{key}:={_format(value)}")
        return " ".join(parts)

    def _record(self, command: str, staged: str | None = None) -> dict:
        out: dict = {
            "backend": self.backend_id,
            "command": command,
            "instructions": (
                "Run the printed command on the lab machine where the UR "
                "driver and the ROS 2 workspace are sourced. The drawing "
                "file (when present) is staged at the path shown below."
            ),
        }
        if staged is not None:
            out["staged_file"] = staged
        self._last_started = out
        return out

    def last_started(self) -> dict | None:
        """Return the most recent staged-launch payload (or ``None``)."""
        return self._last_started


def _format(value) -> str:
    """Stringify a launch argument value the way ``ros2 launch`` expects."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)

"""Simulator backend: spawn ``mjpython -m sim.runner`` as a child process.

Owns at most one running subprocess. Each ``start_*`` first stops the
previous run, then spawns the new one in its own process group so a stop
takes down the whole tree via SIGTERM, falling back to SIGKILL.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

# Monorepo root from ``designer/launcher/sim.py``: parents[2].
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_mjpython() -> str:
    """Locate the ``mjpython`` binary the running interpreter ships next to.

    The MuJoCo wheel installs ``mjpython`` alongside the Python it was
    installed into; falling back to ``shutil.which`` only succeeds when
    that bin dir is on PATH. Resolving the sibling first keeps the
    designer working regardless of the user's PATH.
    """
    sibling = Path(sys.executable).with_name("mjpython")
    if sibling.exists():
        return str(sibling)
    on_path = shutil.which("mjpython")
    if on_path is not None:
        return on_path
    raise RuntimeError(
        "mjpython not found. Install mujoco into the same Python the "
        "designer runs in (`pip install mujoco`) or put the binary on PATH."
    )


class SimLauncher:
    """MuJoCo viewer backend driven by ``sim.runner``."""

    backend_id = "sim"
    label = "Simulator"

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None

    # ----------------------------------------------------------- lifecycle --

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if not self.is_running():
            self.proc = None
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            self.proc.wait(timeout=3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        self.proc = None

    # --------------------------------------------------------------- start --

    def start_drawn(self, drawn_file: Path, *, controller: str | None = None) -> dict:
        cmd = self._base_cmd(controller)
        cmd += ["--drawn-file", str(drawn_file)]
        return self._spawn(cmd)

    def start_preset(
        self,
        shape: str,
        *,
        radius: float | None = None,
        hold_x: float | None = None,
        hold_y: float | None = None,
        controller: str | None = None,
    ) -> dict:
        cmd = self._base_cmd(controller)
        cmd += ["--shape", shape]
        if radius is not None:
            cmd += ["--radius", str(radius)]
        if hold_x is not None:
            cmd += ["--hold-x", str(hold_x)]
        if hold_y is not None:
            cmd += ["--hold-y", str(hold_y)]
        return self._spawn(cmd)

    # ------------------------------------------------------------ internals --

    def _base_cmd(self, controller: str | None) -> list[str]:
        cmd: list[str] = [
            _resolve_mjpython(),
            "-m",
            "sim.runner",
            "--config",
            "config/sim.yaml",
        ]
        if controller in {"pid", "mpc", "rl"}:
            cmd += ["--controller", controller]
            if controller == "rl":
                # The RL policy is trained with the arm locked at home so the
                # deploy launch must mirror that to stay in distribution.
                cmd += ["--lock-joints", "all", "--settle-time", "0"]
        return cmd

    def _spawn(self, cmd: list[str]) -> dict:
        self.stop()
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(_PROJECT_ROOT / "sim"),
            preexec_fn=os.setsid,
        )
        return {"backend": self.backend_id, "pid": self.proc.pid, "cmd": " ".join(cmd)}

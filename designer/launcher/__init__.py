"""Backend launchers for the designer.

``SimLauncher`` and ``RealLauncher`` implement :class:`LauncherBase` so the
Flask server can dispatch trajectories interchangeably; the per-backend
work happens inside ``start_drawn`` / ``start_preset``.
"""

from designer.launcher.base import LauncherBase
from designer.launcher.real import RealLauncher
from designer.launcher.sim import SimLauncher

__all__ = ["LauncherBase", "RealLauncher", "SimLauncher"]

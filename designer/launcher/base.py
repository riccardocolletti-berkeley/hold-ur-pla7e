"""Protocol every backend launcher implements so the Flask request handlers
can route a Run click without branching on the backend type.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class LauncherBase(Protocol):
    """Common launcher surface for ``SimLauncher`` and ``RealLauncher``."""

    backend_id: str  # short id exposed to the UI ("sim" / "real")
    label: str  # human-readable label

    def start_drawn(self, drawn_file: Path, *, controller: str | None = None) -> dict:
        """Run the backend on a drawn JSON trajectory."""
        ...

    def start_preset(
        self,
        shape: str,
        *,
        radius: float | None = None,
        hold_x: float | None = None,
        hold_y: float | None = None,
        controller: str | None = None,
    ) -> dict:
        """Run the backend on a built-in shape.

        ``radius`` is used by ``circle`` / ``figure8``; ``hold_x`` / ``hold_y``
        are used by ``stationary``. Unused parameters are ignored.
        """
        ...

    def stop(self) -> None:
        """Cancel the running invocation, if any."""
        ...

    def is_running(self) -> bool:
        """``True`` while a backend invocation is active."""
        ...

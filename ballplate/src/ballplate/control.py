"""Typed loader for ``config/control.yaml`` plus joint-lock helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class ControlSpec:
    """Cross-backend control settings (currently just the joint-lock list)."""

    locked_joints: tuple[int, ...]


def load(path: Path | str) -> ControlSpec:
    """Parse the control YAML at ``path`` into a ``ControlSpec``."""
    data = yaml.safe_load(Path(path).read_text())
    return ControlSpec(
        locked_joints=tuple(int(i) for i in data.get("locked_joints", [])),
    )


def default_path() -> Path:
    """Path to the monorepo's ``config/control.yaml``, CWD-independent."""
    return Path(__file__).resolve().parents[3] / "config" / "control.yaml"


def parse_lock_spec(
    spec: str | None,
    n_joints: int = 6,
    config_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Resolve a CLI/ROS lock-spec string to an array of joint indices.

    Accepted: ``None`` / ``""`` / ``"none"`` (lock nothing), ``"all"``,
    ``"config"`` (delegates to ``config_indices``), or a comma-separated
    list of indices like ``"0,2,5"``. Out-of-range indices raise
    ``ValueError``.
    """
    if spec is None or spec == "" or spec == "none":
        return np.array([], dtype=int)
    if spec == "config":
        if config_indices is None:
            raise ValueError('parse_lock_spec("config", ...) requires config_indices to be passed')
        return np.asarray(list(config_indices), dtype=int)
    if spec == "all":
        return np.arange(n_joints, dtype=int)

    indices = np.array([int(x) for x in spec.split(",")], dtype=int)
    if (indices < 0).any() or (indices >= n_joints).any():
        raise ValueError(f"Lock indices out of range for {n_joints} joints: {indices.tolist()}")
    return indices


def freeze(
    waypoint: np.ndarray,
    home: Sequence[float],
    locked: np.ndarray,
) -> np.ndarray:
    """Overwrite ``waypoint[i] = home[i]`` for every ``i`` in ``locked``.

    Mutates ``waypoint`` in place and returns it for chaining.
    """
    if len(locked):
        waypoint[locked] = np.asarray(home)[locked]
    return waypoint

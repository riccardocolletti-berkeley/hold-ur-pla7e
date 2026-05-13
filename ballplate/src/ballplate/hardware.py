"""Typed loader for ``config/hardware.yaml``.

The same plate, ball, physics, and tool0->plate adapter parameters are
consumed by the simulator, the standalone vision pipeline, and the ROS
controllers, so they live in one YAML and are surfaced here as frozen
dataclasses.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PlateSpec:
    """Plate geometry, mass, and sliding friction against the wrist mount."""

    size: tuple[float, float]  # full edge lengths along plate (x, y) [m]
    thickness: float  # full slab height [m]
    mass: float  # [kg]
    friction: float  # sliding-friction coefficient


@dataclass(frozen=True)
class BallSpec:
    """Ball radius, mass, and sliding friction against the plate."""

    radius: float  # [m]
    mass: float  # [kg]
    friction: float  # sliding-friction coefficient


@dataclass(frozen=True)
class PhysicsSpec:
    """Universal physics constants.

    ``rolling_factor`` is 5/7 for a uniform solid sphere rolling without
    slip; 1.0 collapses the dynamics to a sliding point mass.
    """

    gravity: float  # [m/s^2]
    rolling_factor: float


@dataclass(frozen=True)
class AdapterSpec:
    """Static pose of the plate centre relative to ``tool0``.

    Position in metres (xyz); orientation as XYZ-Euler in radians.
    """

    position: tuple[float, float, float]
    orientation: tuple[float, float, float]


@dataclass(frozen=True)
class ArmSpec:
    """UR-driver settings used only by the real-robot launch files."""

    speed_slider_fraction: float  # [0.0, 1.0]


@dataclass(frozen=True)
class HardwareSpec:
    """Bundle of every spec exposed by ``hardware.yaml``."""

    plate: PlateSpec
    ball: BallSpec
    physics: PhysicsSpec
    adapter: AdapterSpec
    arm: ArmSpec


def load(path: Path | str) -> HardwareSpec:
    """Parse the hardware YAML at ``path`` into a ``HardwareSpec``."""
    data = yaml.safe_load(Path(path).read_text())
    return HardwareSpec(
        plate=PlateSpec(
            size=tuple(data["plate"]["size"]),
            thickness=float(data["plate"]["thickness"]),
            mass=float(data["plate"]["mass"]),
            friction=float(data["plate"]["friction"]),
        ),
        ball=BallSpec(
            radius=float(data["ball"]["radius"]),
            mass=float(data["ball"]["mass"]),
            friction=float(data["ball"]["friction"]),
        ),
        physics=PhysicsSpec(
            gravity=float(data["physics"]["gravity"]),
            rolling_factor=float(data["physics"]["rolling_factor"]),
        ),
        adapter=AdapterSpec(
            position=_xyz(data["adapter"]["position"]),
            orientation=_xyz(data["adapter"]["orientation"]),
        ),
        arm=ArmSpec(
            speed_slider_fraction=float(data["arm"]["speed_slider_fraction"]),
        ),
    )


def _xyz(values: Iterable[Any]) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in values)
    return x, y, z


def default_path() -> Path:
    """Path to the monorepo's ``config/hardware.yaml``, CWD-independent.

    Walks up three parents from this file: ``ballplate/src/ballplate/``
    -> repo root -> ``config/hardware.yaml``.
    """
    return Path(__file__).resolve().parents[3] / "config" / "hardware.yaml"

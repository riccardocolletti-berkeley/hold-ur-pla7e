"""Helpers to build :class:`PidGains` from declarative mappings.

Each backend loads its own YAML/JSON preset file and passes the parsed
block through here. ``ballplate`` itself stays config-format agnostic.
"""

from collections.abc import Mapping

from ballplate.controllers.pid.controller import PidGains


def from_dict(data: Mapping[str, float]) -> PidGains:
    """Build a ``PidGains`` from a flat mapping.

    Recognised keys mirror the dataclass fields (``kp``, ``ki``, ``kd``,
    ``kff``, ``error_clip``, ``windup_limit``, ``max_output``). Missing
    keys fall back to dataclass defaults; unknown keys raise ``TypeError``.
    """
    return PidGains(**dict(data))


def from_named_dict(presets: Mapping[str, Mapping[str, float]], name: str) -> PidGains:
    """Look up a named block in a ``{name: {kp, ki, ...}}`` mapping."""
    if name not in presets:
        raise KeyError(f"Preset {name!r} not found; available: {sorted(presets)}")
    return from_dict(presets[name])

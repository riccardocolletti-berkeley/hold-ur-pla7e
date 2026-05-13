"""RL policy wrapper.

``RLPolicy`` is lazy-imported so callers that only need the observation
builder (``build_observation``, ``normalize_observation``) stay on the
lean import path. Resolving ``RLPolicy`` triggers the Stable-Baselines 3
import; install the ``[rl]`` extra first.
"""

from ballplate.controllers.rl.observation import (
    build_observation,
    normalize_observation,
)

__all__ = ["RLPolicy", "build_observation", "normalize_observation"]


def __getattr__(name: str) -> object:
    if name == "RLPolicy":
        from ballplate.controllers.rl.controller import RLPolicy

        return RLPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

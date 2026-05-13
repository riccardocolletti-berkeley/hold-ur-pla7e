"""Pure-Python core for the ball-on-plate stack: state, trajectories, PID."""

from ballplate.controllers import PidController, PidGains
from ballplate.state import BallState, PlateGeometry
from ballplate.trajectories import (
    Circle,
    DrawnPath,
    Figure8,
    Reference,
    SmoothApproach,
    Stationary,
)
from ballplate.trajectories import (
    sample as sample_reference,
)

__version__ = "0.1.0"
__all__ = [
    "BallState",
    "Circle",
    "DrawnPath",
    "Figure8",
    "PidController",
    "PidGains",
    "PlateGeometry",
    "Reference",
    "SmoothApproach",
    "Stationary",
    "sample_reference",
]

# ``RLPolicy`` lives in ``ballplate.controllers.rl`` and needs the optional
# ``[rl]`` extra (stable-baselines3 + torch). It is deliberately not
# re-exported here so PID-only callers don't have to install them.

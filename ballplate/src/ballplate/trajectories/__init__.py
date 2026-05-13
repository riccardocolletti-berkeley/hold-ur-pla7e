"""Reference trajectory primitives.

Public symbols:
    Reference        protocol every concrete trajectory satisfies
    Stationary       fixed target point
    Circle           circular path
    Figure8          Lissajous figure-eight
    SmoothApproach   blends from a start pose into an inner reference
    DrawnPath        polyline reference loaded from / saved to JSON
    sample           random reference factory used by training and benchmarks
"""

from ballplate.trajectories.base import Reference
from ballplate.trajectories.drawn import DrawnPath
from ballplate.trajectories.parametric import Circle, Figure8, Stationary
from ballplate.trajectories.random import sample
from ballplate.trajectories.wrappers import SmoothApproach

__all__ = [
    "Circle",
    "DrawnPath",
    "Figure8",
    "Reference",
    "SmoothApproach",
    "Stationary",
    "sample",
]

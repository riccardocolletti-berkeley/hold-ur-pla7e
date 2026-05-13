"""MuJoCo viewer overlays and offscreen video capture.

Always-available primitives are re-exported here. The video `Recorder` lives
in `sim.viz.recorder` and depends on the optional ``[video]`` extra
(``imageio``); importing it explicitly keeps headless / live runs free of
that dependency::

    from sim.viz import BallTrail, TargetOverlay     # always works
    from sim.viz.recorder import Recorder             # needs sim[video]
"""

from sim.viz.overlay import TargetOverlay
from sim.viz.trail import BallTrail

__all__ = ["BallTrail", "TargetOverlay"]

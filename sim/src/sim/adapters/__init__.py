"""Bridges between MuJoCo's data structures and `ballplate`'s pure types.

Three single-responsibility adapters:
    * `BallStateReader`: turns MuJoCo state into a `BallState` (+ plate frame).
    * `JointActuator`: turns a virtual `(ux, uy)` control into joint torques.
    * `BallDropMonitor`: detects when the ball has fallen off the plate.

Everything that touches `mujoco.MjModel`, `mujoco.MjData`, or any other MuJoCo
API lives here so the rest of `sim/` (controllers, learning, runner) sees only
`ballplate` types.
"""

from sim.adapters.actuation import JointActuator
from sim.adapters.monitor import BallDropMonitor
from sim.adapters.state import BallStateReader

__all__ = ["BallDropMonitor", "BallStateReader", "JointActuator"]

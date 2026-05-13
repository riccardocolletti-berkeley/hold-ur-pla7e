"""Arm-side trajectories and joint-locking helpers (sim-only)."""

from sim.arm.lock import freeze, parse_lock_spec
from sim.arm.waypoints import JointTrajectory

__all__ = ["JointTrajectory", "freeze", "parse_lock_spec"]

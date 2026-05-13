"""Tests for sim.arm.waypoints."""

import math

import numpy as np
import pytest

from sim.arm.waypoints import JointTrajectory


def _two_segment_traj() -> JointTrajectory:
    # 2-joint robot, two segments of 1 s each; joints sweep independently.
    waypoints = [
        np.array([0.0, 0.0]),
        np.array([1.0, -0.5]),
        np.array([0.5, 0.5]),
    ]
    durations = [1.0, 1.0]
    return JointTrajectory(waypoints, durations)


# ------------------------------------------------------------- construction --


def test_construction_rejects_too_few_waypoints():
    with pytest.raises(ValueError):
        JointTrajectory([np.zeros(2)], [])


def test_construction_rejects_duration_mismatch():
    with pytest.raises(ValueError):
        JointTrajectory([np.zeros(2), np.ones(2)], [1.0, 2.0])


def test_total_time_matches_cumulative_durations():
    traj = _two_segment_traj()
    assert traj.total_time == 2.0
    assert traj.n_segments == 2
    assert traj.n_joints == 2


# ----------------------------------------------------------------- evaluate --


def test_evaluate_at_segment_starts_returns_waypoints():
    traj = _two_segment_traj()
    q0, v0, _ = traj.evaluate(0.0)
    assert np.allclose(q0, [0.0, 0.0])
    # Cubic boundary condition: zero velocity at every waypoint.
    assert np.allclose(v0, [0.0, 0.0])

    q1, v1, _ = traj.evaluate(1.0 - 1e-9)
    assert np.allclose(q1, [1.0, -0.5], atol=1e-6)
    assert np.allclose(v1, [0.0, 0.0], atol=1e-6)


def test_evaluate_saturates_after_total_time():
    traj = _two_segment_traj()
    q_end, _, _ = traj.evaluate(traj.total_time + 5.0)
    # Last waypoint position; the planner clamps t to (total_time - 1e-8).
    assert np.allclose(q_end, [0.5, 0.5], atol=1e-3)


def test_velocity_is_continuous_at_segment_boundary():
    # Both segments end with zero velocity, so crossing the boundary keeps it
    # smooth: v approaches 0 from below and stays near 0 just after.
    traj = _two_segment_traj()
    _, v_before, _ = traj.evaluate(1.0 - 5e-4)
    _, v_after, _ = traj.evaluate(1.0 + 5e-4)
    assert math.isclose(v_before[0], 0.0, abs_tol=5e-3)
    assert math.isclose(v_after[0], 0.0, abs_tol=5e-3)

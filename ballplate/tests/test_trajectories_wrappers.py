"""Tests for ``SmoothApproach``: blend-in, zero-velocity start, inner pass-through after blend."""

import math

from ballplate.trajectories import Circle, SmoothApproach


def test_smooth_approach_starts_at_start_pose_with_zero_velocity():
    inner = Circle(radius=0.05, period=2.0)
    w = SmoothApproach(inner=inner, start_x=0.02, start_y=-0.03, blend_time=1.0)
    x, y, vx, vy = w.evaluate(0.0)
    assert math.isclose(x, 0.02)
    assert math.isclose(y, -0.03)
    # Smoothstep derivative is zero at tau = 0, so velocity is zero too.
    assert math.isclose(vx, 0.0)
    assert math.isclose(vy, 0.0)


def test_smooth_approach_matches_inner_after_blend_time():
    inner = Circle(radius=0.05, period=2.0)
    w = SmoothApproach(inner=inner, start_x=0.02, start_y=-0.03, blend_time=1.0)
    assert w.evaluate(1.0) == inner.evaluate(1.0)
    assert w.evaluate(2.5) == inner.evaluate(2.5)


def test_smooth_approach_with_zero_blend_is_passthrough():
    inner = Circle(radius=0.05, period=2.0)
    w = SmoothApproach(inner=inner, start_x=99.0, start_y=99.0, blend_time=0.0)
    for t in (0.0, 0.5, 3.7):
        assert w.evaluate(t) == inner.evaluate(t)


def test_smooth_approach_period_is_inner_period():
    inner = Circle(radius=0.05, period=2.0)
    w = SmoothApproach(inner=inner, start_x=0.0, start_y=0.0, blend_time=1.0)
    assert w.period == inner.period

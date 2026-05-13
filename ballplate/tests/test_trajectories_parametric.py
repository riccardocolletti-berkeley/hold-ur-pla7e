"""Tests for ``Stationary``, ``Circle``, ``Figure8``: positions, velocities, geometric invariants."""

import math

from ballplate.trajectories import Circle, Figure8, Stationary

# ---------------------------------------------------------------- Stationary --


def test_stationary_default_is_origin():
    s = Stationary()
    for t in (0.0, 1.5, 100.0):
        assert s.evaluate(t) == (0.0, 0.0, 0.0, 0.0)


def test_stationary_with_offset_returns_constant_position():
    s = Stationary(x=0.05, y=-0.02)
    assert s.evaluate(0.0) == (0.05, -0.02, 0.0, 0.0)
    assert s.evaluate(10.0) == (0.05, -0.02, 0.0, 0.0)


def test_stationary_period_is_positive():
    assert Stationary().period > 0.0


# -------------------------------------------------------------------- Circle --


def test_circle_at_t_zero_starts_on_plus_x_axis():
    c = Circle(radius=0.05, period=2.0)
    x, y, vx, vy = c.evaluate(0.0)
    assert math.isclose(x, 0.05)
    assert math.isclose(y, 0.0, abs_tol=1e-12)
    # Velocity is tangent: at +x axis it points along +y.
    assert math.isclose(vx, 0.0, abs_tol=1e-12)
    assert vy > 0.0


def test_circle_radius_is_constant_around_path():
    c = Circle(radius=0.075, period=3.0)
    for t in (0.0, 0.5, 1.0, 2.31):
        x, y, _, _ = c.evaluate(t)
        assert math.isclose(math.hypot(x, y), 0.075, abs_tol=1e-12)


def test_circle_velocity_is_tangent():
    c = Circle(radius=0.05, period=4.0)
    for t in (0.1, 1.7, 2.9):
        x, y, vx, vy = c.evaluate(t)
        # On a circle, position and velocity are orthogonal: <p, v> = 0.
        assert math.isclose(x * vx + y * vy, 0.0, abs_tol=1e-12)


# ------------------------------------------------------------------ Figure8 --


def test_figure8_passes_through_origin_at_t_zero():
    f = Figure8(rx=0.04, ry=0.02, period=4.0)
    x, y, vx, _vy = f.evaluate(0.0)
    assert math.isclose(x, 0.0, abs_tol=1e-12)
    assert math.isclose(y, 0.0, abs_tol=1e-12)
    # Velocity at the crossing is non-zero.
    assert vx != 0.0


def test_figure8_amplitudes_are_bounded_by_rx_and_ry():
    f = Figure8(rx=0.08, ry=0.03, period=4.0)
    samples = [f.evaluate(t) for t in [i / 50.0 for i in range(50)]]
    max_x = max(abs(x) for x, _, _, _ in samples)
    max_y = max(abs(y) for _, y, _, _ in samples)
    assert max_x <= 0.08 + 1e-12
    assert max_y <= 0.03 + 1e-12

"""Tests for ``PidController`` and ``PidGains``: per-axis P/I/D, feed-forward, error clip, anti-windup, output clamp."""

import math

from ballplate import BallState
from ballplate.controllers import PidController, PidGains


def _ball(x=0.0, y=0.0, vx=0.0, vy=0.0) -> BallState:
    return BallState(x=x, y=y, vx=vx, vy=vy, timestamp=0.0)


# ----------------------------------------------------------------- basic --


def test_zero_error_zero_output():
    pid = PidController(PidGains(kp=1.0, ki=0.5, kd=0.2))
    ux, uy = pid.step(_ball(), target_pos=(0.0, 0.0), dt=0.01)
    assert ux == 0.0
    assert uy == 0.0


def test_positive_x_error_yields_positive_ux_with_default_signs():
    pid = PidController(PidGains(kp=2.0, ki=0.0, kd=0.0))
    ux, uy = pid.step(_ball(x=0.05), target_pos=(0.0, 0.0))
    assert ux == 2.0 * 0.05
    assert uy == 0.0


def test_velocity_error_contributes_via_kd():
    pid = PidController(PidGains(kp=0.0, ki=0.0, kd=4.0))
    ux, _ = pid.step(_ball(vx=0.1), target_pos=(0.0, 0.0))
    assert math.isclose(ux, 0.4)


def test_target_velocity_subtracts_from_velocity_error():
    pid = PidController(PidGains(kp=0.0, ki=0.0, kd=4.0))
    # ball.vx == target.vx ⇒ no derivative contribution.
    ux, _ = pid.step(_ball(vx=0.1), target_pos=(0.0, 0.0), target_vel=(0.1, 0.0))
    assert ux == 0.0


def test_feed_forward_subtracts_from_output():
    # kff is subtracted (drives the controller to anticipate the curvature).
    pid = PidController(PidGains(kp=0.0, ki=0.0, kd=0.0, kff=2.0))
    ux, _ = pid.step(_ball(), target_pos=(0.0, 0.0), target_acc=(0.5, 0.0))
    assert ux == -1.0


# ---------------------------------------------------------------- integral --


def test_integral_accumulates_with_constant_error():
    pid = PidController(PidGains(kp=0.0, ki=1.0, kd=0.0))
    pid.step(_ball(x=0.1), target_pos=(0.0, 0.0), dt=0.5)
    pid.step(_ball(x=0.1), target_pos=(0.0, 0.0), dt=0.5)
    int_x, _ = pid.integral
    # Two steps of 0.05 each ⇒ 0.1 in the integrator.
    assert math.isclose(int_x, 0.1)


def test_integral_is_clamped_by_windup_limit():
    pid = PidController(PidGains(kp=0.0, ki=1.0, kd=0.0, windup_limit=0.05))
    pid.step(_ball(x=10.0), target_pos=(0.0, 0.0), dt=1.0)
    int_x, _ = pid.integral
    assert int_x == 0.05


def test_reset_clears_integrator():
    pid = PidController(PidGains(kp=0.0, ki=1.0, kd=0.0))
    pid.step(_ball(x=0.1), target_pos=(0.0, 0.0), dt=1.0)
    assert pid.integral != (0.0, 0.0)
    pid.reset()
    assert pid.integral == (0.0, 0.0)


# ------------------------------------------------------------- error clip --


def test_position_error_norm_is_clamped_along_axis():
    # ex = 0.5, ey = 0; norm = 0.5 > 0.1, rescaled to (0.1, 0).
    pid = PidController(PidGains(kp=2.0, ki=0.0, kd=0.0, error_clip=0.1))
    ux, uy = pid.step(_ball(x=0.5, y=0.0), target_pos=(0.0, 0.0))
    assert math.isclose(ux, 2.0 * 0.1)
    assert uy == 0.0


def test_position_error_norm_clip_preserves_direction_diagonally():
    # (ex, ey) = (0.3, 0.4), norm = 0.5; rescale to norm 0.1 ⇒ (0.06, 0.08).
    pid = PidController(PidGains(kp=1.0, ki=0.0, kd=0.0, error_clip=0.1))
    ux, uy = pid.step(_ball(x=0.3, y=0.4), target_pos=(0.0, 0.0))
    assert math.isclose(ux, 0.06)
    assert math.isclose(uy, 0.08)
    assert math.isclose(math.hypot(ux, uy), 0.1)


def test_position_error_below_clip_is_unchanged():
    pid = PidController(PidGains(kp=1.0, ki=0.0, kd=0.0, error_clip=1.0))
    ux, uy = pid.step(_ball(x=0.03, y=-0.04), target_pos=(0.0, 0.0))
    assert math.isclose(ux, 0.03)
    assert math.isclose(uy, -0.04)


def test_position_error_clip_also_caps_integral_growth():
    # x=10, y=0 ⇒ norm=10, rescaled to 0.05; integral accumulates clipped error.
    pid = PidController(PidGains(kp=0.0, ki=1.0, kd=0.0, error_clip=0.05))
    pid.step(_ball(x=10.0), target_pos=(0.0, 0.0), dt=1.0)
    int_x, _ = pid.integral
    assert math.isclose(int_x, 0.05)


def test_velocity_error_is_not_affected_by_error_clip():
    # error_clip is a position-error clamp; D term sees the raw velocity error.
    pid = PidController(PidGains(kp=0.0, ki=0.0, kd=4.0, error_clip=0.001))
    ux, _ = pid.step(_ball(vx=0.1), target_pos=(0.0, 0.0))
    assert math.isclose(ux, 0.4)


# ----------------------------------------------------------- output limits --


def test_output_is_clamped_by_max_output():
    pid = PidController(PidGains(kp=1000.0, ki=0.0, kd=0.0, max_output=2.0))
    ux, uy = pid.step(_ball(x=10.0, y=-10.0), target_pos=(0.0, 0.0))
    assert ux == 2.0
    assert uy == -2.0

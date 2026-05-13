"""Tests for ``MpcController``: sign conventions, output clamp, horizon
anticipation, and parity with the PID surface.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ballplate import BallState
from ballplate.controllers.mpc import MpcController, MpcParams


def _ball(x: float = 0.0, y: float = 0.0, vx: float = 0.0, vy: float = 0.0) -> BallState:
    return BallState(x=x, y=y, vx=vx, vy=vy, timestamp=0.0)


def _stable_params(**overrides) -> MpcParams:
    """Default test params: short horizon, dt=1/60, well-damped costs."""
    base = dict(
        horizon=15, dt=1.0 / 60.0, q_pos=200.0, q_vel=20.0, r_u=1.0, qf_pos=1000.0, qf_vel=100.0
    )
    base.update(overrides)
    return MpcParams(**base)


# ============================================================ params validation --


def test_params_validate_rejects_zero_r_u():
    # r_u = 0 makes the Hessian rank-deficient when q_vel = 0; fail loudly.
    with pytest.raises(ValueError, match="r_u"):
        MpcParams(r_u=0.0).validate()


def test_params_validate_rejects_negative_horizon():
    with pytest.raises(ValueError, match="horizon"):
        MpcParams(horizon=0).validate()


def test_params_validate_rejects_zero_dt():
    with pytest.raises(ValueError, match="dt"):
        MpcParams(dt=0.0).validate()


# ============================================================ basic step API --


def test_zero_state_zero_ref_yields_zero_output():
    mpc = MpcController(_stable_params())
    ux, uy = mpc.step(_ball(), target_pos=(0.0, 0.0))
    assert math.isclose(ux, 0.0, abs_tol=1e-12)
    assert math.isclose(uy, 0.0, abs_tol=1e-12)


def test_positive_x_ball_yields_positive_ux_matching_pid_sign_convention():
    # Ball at +x must give ux > 0 (downstream projection then tilts toward -x).
    # The PID test pins the same convention; MPC must agree for shared wiring.
    mpc = MpcController(_stable_params())
    ux, uy = mpc.step(_ball(x=0.05), target_pos=(0.0, 0.0))
    assert ux > 0.0
    assert math.isclose(uy, 0.0, abs_tol=1e-9)


def test_positive_y_ball_yields_positive_uy():
    mpc = MpcController(_stable_params())
    ux, uy = mpc.step(_ball(y=0.05), target_pos=(0.0, 0.0))
    assert uy > 0.0
    assert math.isclose(ux, 0.0, abs_tol=1e-9)


def test_velocity_alone_drives_braking_command():
    # Ball moving in -x: MPC should brake, i.e. ux < 0 (drives ball in +x).
    mpc = MpcController(_stable_params())
    ux, _ = mpc.step(_ball(vx=-0.3), target_pos=(0.0, 0.0))
    assert ux < 0.0


def test_target_offset_drives_toward_target():
    # Ball at origin, target at +0.05: error -0.05, expect ux < 0 (push +x).
    mpc = MpcController(_stable_params())
    ux, _ = mpc.step(_ball(), target_pos=(0.05, 0.0))
    assert ux < 0.0


# ============================================================ output limits --


def test_output_is_clamped_by_max_output():
    # Huge state with a strong q_pos to force saturation; max_output bounds it.
    mpc = MpcController(_stable_params(q_pos=1e6, max_output=0.20))
    ux, uy = mpc.step(_ball(x=10.0, y=-10.0), target_pos=(0.0, 0.0))
    assert math.isclose(ux, 0.20, abs_tol=1e-12)
    assert math.isclose(uy, -0.20, abs_tol=1e-12)


def test_unconstrained_output_unchanged_when_max_output_loose():
    # A loose ``max_output`` (well above what the optimum would produce)
    # must give the same control as the unconstrained case.
    mpc_loose = MpcController(_stable_params(max_output=10.0))
    ux_loose, _ = mpc_loose.step(_ball(x=0.05), target_pos=(0.0, 0.0))
    mpc_unbounded = MpcController(_stable_params(max_output=math.inf))
    ux_unbounded, _ = mpc_unbounded.step(_ball(x=0.05), target_pos=(0.0, 0.0))
    assert math.isclose(ux_loose, ux_unbounded, rel_tol=1e-12)
    assert abs(ux_loose) < 10.0  # confirm the cap did not bind


# =========================================================== reset / state --


def test_reset_does_not_change_gains():
    # reset() must leave the gain matrices intact.
    mpc = MpcController(_stable_params())
    ux1, _ = mpc.step(_ball(x=0.1), target_pos=(0.0, 0.0))
    mpc.reset()
    ux2, _ = mpc.step(_ball(x=0.1), target_pos=(0.0, 0.0))
    assert math.isclose(ux1, ux2, rel_tol=1e-12)


def test_target_held_steady_state_zero_offset():
    # Ball exactly on target with zero velocity: optimal control is zero.
    mpc = MpcController(_stable_params())
    ux, uy = mpc.step(_ball(x=0.07, y=-0.03), target_pos=(0.07, -0.03))
    assert math.isclose(ux, 0.0, abs_tol=1e-9)
    assert math.isclose(uy, 0.0, abs_tol=1e-9)


# ===================================================== step_with_horizon --


def test_step_with_horizon_rejects_wrong_shape():
    mpc = MpcController(_stable_params(horizon=10))
    with pytest.raises(ValueError, match="ref_horizon must have shape"):
        mpc.step_with_horizon(_ball(), np.zeros((5, 4)))


def test_step_with_horizon_constant_ref_matches_step():
    # Constant reference along the horizon must match the ``step`` shortcut.
    mpc = MpcController(_stable_params(horizon=12))
    target = (0.04, -0.02)
    target_v = (0.0, 0.0)

    ux_step, uy_step = mpc.step(_ball(x=0.0, y=0.0), target_pos=target, target_vel=target_v)

    ref = np.tile(np.array([target[0], target[1], 0.0, 0.0]), (mpc.horizon, 1))
    ux_h, uy_h = mpc.step_with_horizon(_ball(x=0.0, y=0.0), ref)

    assert math.isclose(ux_step, ux_h, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(uy_step, uy_h, rel_tol=1e-12, abs_tol=1e-12)


def test_step_with_reference_matches_manual_horizon_sample():
    # ``step_with_reference`` is sugar over ``step_with_horizon``: sampling
    # the same Reference by hand must give the identical control.
    from ballplate.trajectories import Circle

    mpc = MpcController(_stable_params(horizon=15))
    circle = Circle(radius=0.05, period=2.0)
    now = 0.7

    ux_a, uy_a = mpc.step_with_reference(_ball(x=0.01, vx=0.05), circle, now)

    horizon = np.empty((mpc.horizon, 4))
    for k in range(mpc.horizon):
        horizon[k] = circle.evaluate(now + (k + 1) * mpc.dt)
    ux_b, uy_b = mpc.step_with_horizon(_ball(x=0.01, vx=0.05), horizon)

    assert math.isclose(ux_a, ux_b, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(uy_a, uy_b, rel_tol=1e-12, abs_tol=1e-12)


def test_step_with_horizon_anticipates_moving_target():
    # A reference moving in +x must pull the control toward +x even when
    # the ball is at the current target; the constant-ref call would see
    # no error and return zero. This is the point of feeding the horizon.
    mpc = MpcController(_stable_params(horizon=20))
    # Ball sitting at origin; reference ramps in +x at 0.2 m/s.
    dt = mpc.dt
    ref = np.zeros((mpc.horizon, 4))
    for k in range(mpc.horizon):
        ref[k, 0] = (k + 1) * dt * 0.2  # x_ref grows
        ref[k, 2] = 0.2  # vx_ref constant
    ux_h, _ = mpc.step_with_horizon(_ball(), ref)
    # ux < 0 drives the ball in +x (sign convention check above).
    assert ux_h < 0.0, f"expected anticipatory negative ux, got {ux_h}"

    # Compared to the constant-ref call (which sees no future motion),
    # the horizon-aware command should be more aggressive.
    ux_const, _ = mpc.step(_ball(), target_pos=(0.0, 0.0), target_vel=(0.0, 0.0))
    assert abs(ux_h) > abs(ux_const)

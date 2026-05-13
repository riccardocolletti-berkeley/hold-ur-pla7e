"""Tests for sim.learning.reward."""

import math

from sim.learning.reward import step_reward


def _weights(**overrides):
    base = {
        "w_track": 1.0,
        "w_vel": 0.3,
        "w_prog": 0.5,
        "w_act": 0.01,
        "w_smooth": 0.05,
        "w_drop": 1.0,
        "sigma_pos": 0.05,
        "sigma_vel": 0.2,
    }
    base.update(overrides)
    return base


def test_perfect_tracking_returns_track_plus_vel_bonus():
    r, err = step_reward(
        ball_pos=(0.0, 0.0),
        ball_vel=(0.0, 0.0),
        target_pos=(0.0, 0.0),
        target_vel=(0.0, 0.0),
        action=(0.0, 0.0),
        prev_action=(0.0, 0.0),
        prev_pos_err=None,
        dropped=False,
        steps_remaining_frac=0.0,
        weights=_weights(),
    )
    assert err == 0.0
    # exp(0) = 1, so r = w_track * 1 + w_vel * 1 = 1.3.
    assert math.isclose(r, 1.3)


def test_position_error_decays_exponentially_with_sigma_pos():
    r, err = step_reward(
        ball_pos=(0.05, 0.0),
        ball_vel=(0.0, 0.0),
        target_pos=(0.0, 0.0),
        target_vel=(0.0, 0.0),
        action=(0.0, 0.0),
        prev_action=(0.0, 0.0),
        prev_pos_err=None,
        dropped=False,
        steps_remaining_frac=0.0,
        weights=_weights(),
    )
    assert math.isclose(err, 0.05)
    # err = sigma_pos ⇒ exp(-1) ≈ 0.368 contribution from w_track.
    expected = 1.0 * math.exp(-1.0) + 0.3 * 1.0
    assert math.isclose(r, expected, rel_tol=1e-9)


def test_progress_term_rewards_decreasing_error():
    r, _err = step_reward(
        ball_pos=(0.05, 0.0),
        ball_vel=(0.0, 0.0),
        target_pos=(0.0, 0.0),
        target_vel=(0.0, 0.0),
        action=(0.0, 0.0),
        prev_action=(0.0, 0.0),
        prev_pos_err=0.10,  # was farther; now closer.
        dropped=False,
        steps_remaining_frac=0.0,
        weights=_weights(),
    )
    base, _ = step_reward(
        ball_pos=(0.05, 0.0),
        ball_vel=(0.0, 0.0),
        target_pos=(0.0, 0.0),
        target_vel=(0.0, 0.0),
        action=(0.0, 0.0),
        prev_action=(0.0, 0.0),
        prev_pos_err=None,
        dropped=False,
        steps_remaining_frac=0.0,
        weights=_weights(),
    )
    # Improving by 0.05 m at w_prog=0.5 adds +0.025 over the no-progress baseline.
    assert math.isclose(r - base, 0.5 * (0.10 - 0.05), rel_tol=1e-9)


def test_action_smoothness_penalty_subtracts():
    r, _ = step_reward(
        ball_pos=(0.0, 0.0),
        ball_vel=(0.0, 0.0),
        target_pos=(0.0, 0.0),
        target_vel=(0.0, 0.0),
        action=(1.0, 0.0),
        prev_action=(0.0, 0.0),
        prev_pos_err=None,
        dropped=False,
        steps_remaining_frac=0.0,
        weights=_weights(),
    )
    # action² = 1 ⇒ -0.01; (Δaction)² = 1 ⇒ -0.05; tracking still 1.3.
    assert math.isclose(r, 1.3 - 0.01 - 0.05, rel_tol=1e-9)


def test_drop_penalty_scales_with_remaining_fraction():
    early, _ = step_reward(
        ball_pos=(0.0, 0.0),
        ball_vel=(0.0, 0.0),
        target_pos=(0.0, 0.0),
        target_vel=(0.0, 0.0),
        action=(0.0, 0.0),
        prev_action=(0.0, 0.0),
        prev_pos_err=None,
        dropped=True,
        steps_remaining_frac=0.95,
        weights=_weights(w_drop=2.0),
    )
    late, _ = step_reward(
        ball_pos=(0.0, 0.0),
        ball_vel=(0.0, 0.0),
        target_pos=(0.0, 0.0),
        target_vel=(0.0, 0.0),
        action=(0.0, 0.0),
        prev_action=(0.0, 0.0),
        prev_pos_err=None,
        dropped=True,
        steps_remaining_frac=0.05,
        weights=_weights(w_drop=2.0),
    )
    # Same step otherwise; only the drop penalty differs.
    assert math.isclose(early - late, -2.0 * (0.95 - 0.05), rel_tol=1e-9)

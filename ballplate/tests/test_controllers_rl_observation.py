"""Tests for ``build_observation`` and ``normalize_observation``: layout, dtype, clipping."""

import numpy as np

from ballplate import BallState
from ballplate.controllers.rl import build_observation, normalize_observation
from ballplate.controllers.rl.observation import OBSERVATION_DIM


def _ball() -> BallState:
    return BallState(x=0.05, y=-0.02, vx=0.10, vy=0.0, timestamp=0.0)


def test_observation_dim_is_twelve():
    assert OBSERVATION_DIM == 12


def test_observation_shape_and_dtype():
    obs = build_observation(
        _ball(),
        target_now=(0.0, 0.0, 0.0, 0.0),
        target_lookahead_pos=(0.01, 0.0),
        plate_pitch=0.0,
        plate_roll=0.0,
    )
    assert obs.shape == (OBSERVATION_DIM,)
    assert obs.dtype == np.float32


def test_observation_layout_matches_documented_order():
    obs = build_observation(
        _ball(),
        target_now=(0.10, 0.20, 0.30, 0.40),
        target_lookahead_pos=(0.50, 0.60),
        plate_pitch=0.07,
        plate_roll=-0.03,
    )
    expected = np.array(
        [0.05, -0.02, 0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.07, -0.03],
        dtype=np.float32,
    )
    assert np.allclose(obs, expected)


def test_normalize_observation_zero_mean_unit_std_is_identity():
    obs = np.arange(OBSERVATION_DIM, dtype=np.float32)
    mean = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    std = np.ones(OBSERVATION_DIM, dtype=np.float32)
    # clip > max(obs) so the identity isn't masked by saturation.
    out = normalize_observation(obs, mean, std, clip=100.0)
    assert np.allclose(out, obs)


def test_normalize_observation_clips_outliers():
    obs = np.array([100.0] * OBSERVATION_DIM, dtype=np.float32)
    mean = np.zeros(OBSERVATION_DIM, dtype=np.float32)
    std = np.ones(OBSERVATION_DIM, dtype=np.float32)
    out = normalize_observation(obs, mean, std, clip=5.0)
    assert np.all(out == 5.0)

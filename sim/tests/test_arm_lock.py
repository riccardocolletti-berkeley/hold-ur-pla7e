"""Tests for the shared joint-locking helpers in :mod:`ballplate.control`."""

import numpy as np
import pytest

from sim.arm.lock import freeze, parse_lock_spec

# ------------------------------------------------------------- parse_lock_spec --


def test_empty_and_none_lock_nothing():
    assert parse_lock_spec("").tolist() == []
    assert parse_lock_spec("none").tolist() == []


def test_all_locks_every_joint():
    assert parse_lock_spec("all", n_joints=6).tolist() == [0, 1, 2, 3, 4, 5]


def test_explicit_indices_are_parsed():
    assert parse_lock_spec("0,2,5").tolist() == [0, 2, 5]


def test_indices_out_of_range_raise():
    with pytest.raises(ValueError):
        parse_lock_spec("0,7", n_joints=6)


def test_config_returns_passed_indices():
    out = parse_lock_spec("config", n_joints=6, config_indices=[0, 1, 2])
    assert out.tolist() == [0, 1, 2]


def test_config_without_indices_raises():
    with pytest.raises(ValueError):
        parse_lock_spec("config", n_joints=6)


# ----------------------------------------------------------------- freeze --


def test_freeze_overrides_only_locked_indices():
    home = np.array([0.5, -2.35, 2.65, -1.85, 1.57, 0.0])
    waypoint = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    out = freeze(waypoint, home, locked=np.array([0, 2, 5]))
    # Locked indices match home; the others stay at their original 1.0.
    assert out[0] == 0.5
    assert out[2] == 2.65
    assert out[5] == 0.0
    assert out[1] == 1.0 and out[3] == 1.0 and out[4] == 1.0


def test_freeze_with_no_locked_indices_is_a_noop():
    home = np.zeros(6)
    waypoint = np.arange(6, dtype=float)
    out = freeze(waypoint, home, locked=np.array([], dtype=int))
    assert np.allclose(out, np.arange(6))


def test_freeze_returns_same_array_in_place():
    home = np.zeros(3)
    waypoint = np.array([1.0, 2.0, 3.0])
    out = freeze(waypoint, home, locked=np.array([0]))
    assert out is waypoint

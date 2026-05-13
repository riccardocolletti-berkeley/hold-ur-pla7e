"""Tests for ballplate.state."""

import dataclasses

import pytest

from ballplate import BallState, PlateGeometry


def test_construction_with_required_fields():
    s = BallState(x=0.05, y=-0.02, vx=0.10, vy=0.0, timestamp=1.234)
    assert s.x == 0.05
    assert s.y == -0.02
    assert s.vx == 0.10
    assert s.vy == 0.0
    assert s.timestamp == 1.234
    assert s.valid is True


def test_construction_with_explicit_invalid_flag():
    s = BallState(x=0.0, y=0.0, vx=0.0, vy=0.0, timestamp=0.0, valid=False)
    assert s.valid is False


def test_instance_is_immutable():
    s = BallState(x=0.0, y=0.0, vx=0.0, vy=0.0, timestamp=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.x = 1.0  # type: ignore[misc]


def test_equality_is_value_based():
    a = BallState(x=0.1, y=0.0, vx=0.0, vy=0.0, timestamp=1.0)
    b = BallState(x=0.1, y=0.0, vx=0.0, vy=0.0, timestamp=1.0)
    c = BallState(x=0.1, y=0.0, vx=0.0, vy=0.0, timestamp=2.0)
    assert a == b
    assert a != c


# ------------------------------------------------------------- PlateGeometry --


def test_plate_geometry_half_extents():
    p = PlateGeometry(size_x=0.70, size_y=0.50, thickness=0.005)
    assert p.half_x == 0.35
    assert p.half_y == 0.25


def test_plate_geometry_is_immutable():
    p = PlateGeometry(size_x=0.30, size_y=0.30, thickness=0.005)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.size_x = 1.0  # type: ignore[misc]

"""Tests for ``DrawnPath``: validation, evaluate, end modes, JSON round-trip."""

import json
import math

import pytest

from ballplate.trajectories import DrawnPath


def _square_path() -> DrawnPath:
    # Square (4 cm half-side), traversed in 4 s, looping.
    pts = [
        [0.05, 0.0],
        [0.0, 0.05],
        [-0.05, 0.0],
        [0.0, -0.05],
        [0.05, 0.0],
    ]
    return DrawnPath(points_xy_m=pts, duration_s=4.0, end_mode="loop")


# -------------------------------------------------------------- Construction --


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        DrawnPath(points_xy_m=[[0.0]], duration_s=1.0)


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        DrawnPath(points_xy_m=[[0.0, 0.0]], duration_s=1.0)


def test_non_positive_duration_raises():
    with pytest.raises(ValueError):
        DrawnPath(points_xy_m=[[0.0, 0.0], [0.1, 0.0]], duration_s=0.0)


def test_unknown_end_mode_raises():
    with pytest.raises(ValueError):
        DrawnPath(
            points_xy_m=[[0.0, 0.0], [0.1, 0.0]],
            duration_s=1.0,
            end_mode="bogus",  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------- Evaluate --


def test_evaluate_at_zero_returns_first_point():
    p = _square_path()
    x, y, _, _ = p.evaluate(0.0)
    assert math.isclose(x, 0.05)
    assert math.isclose(y, 0.0)


def test_evaluate_loops_around_period():
    p = _square_path()
    a = p.evaluate(0.5)
    b = p.evaluate(0.5 + p.period)
    assert all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(a, b, strict=True))


def test_stop_mode_holds_last_point_with_zero_velocity():
    pts = [[0.0, 0.0], [0.05, 0.0]]
    p = DrawnPath(points_xy_m=pts, duration_s=1.0, end_mode="stop")
    x, y, vx, vy = p.evaluate(5.0)
    assert math.isclose(x, 0.05)
    assert math.isclose(y, 0.0)
    assert vx == 0.0 and vy == 0.0


def test_pingpong_mode_reverses_velocity_on_return_leg():
    pts = [[0.0, 0.0], [0.10, 0.0]]
    p = DrawnPath(points_xy_m=pts, duration_s=1.0, end_mode="pingpong")
    _, _, vx_fwd, _ = p.evaluate(0.5)  # forward leg
    _, _, vx_back, _ = p.evaluate(1.5)  # return leg
    assert vx_fwd > 0.0 and vx_back < 0.0


# -------------------------------------------------------------- Serialization --


def test_round_trip_through_dict():
    p = _square_path()
    p2 = DrawnPath.from_dict(p.to_dict())
    assert p2.duration_s == p.duration_s
    assert p2.end_mode == p.end_mode
    assert p2.points.tolist() == p.points.tolist()


def test_round_trip_through_file(tmp_path):
    p = _square_path()
    f = tmp_path / "path.json"
    p.to_file(f)

    raw = json.loads(f.read_text())
    assert raw["version"] == 1
    assert raw["duration_s"] == p.duration_s

    p2 = DrawnPath.from_file(f)
    assert p2.points.tolist() == p.points.tolist()

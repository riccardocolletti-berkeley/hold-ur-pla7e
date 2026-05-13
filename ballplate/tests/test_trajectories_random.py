"""Tests for the ``sample`` random-reference factory: each shape stays inside the safe area."""

import math

import numpy as np
import pytest

from ballplate.state import PlateGeometry
from ballplate.trajectories import sample


def _plate() -> PlateGeometry:
    return PlateGeometry(size_x=0.70, size_y=0.50, thickness=0.005)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.mark.parametrize(
    "shape",
    ["stationary", "slow_circle", "circle", "figure8", "random_spline"],
)
def test_each_shape_returns_reference(shape):
    ref = sample(shape, _rng(), _plate(), ball_radius=0.02)
    # Reference protocol: check the two required members.
    assert hasattr(ref, "evaluate")
    assert hasattr(ref, "period")
    assert ref.period > 0.0
    out = ref.evaluate(0.0)
    assert len(out) == 4


def test_unknown_shape_raises():
    with pytest.raises(ValueError):
        sample("does_not_exist", _rng(), _plate(), ball_radius=0.02)  # type: ignore[arg-type]


def test_sampled_circle_fits_inside_safe_area():
    plate = _plate()
    ball_r = 0.02
    margin = 0.02
    safe = min(plate.half_x, plate.half_y) - ball_r - margin
    ref = sample("circle", _rng(123), plate, ball_radius=ball_r, margin=margin)
    # Walk the path and check the radius never exceeds the safe envelope.
    for k in range(200):
        x, y, _, _ = ref.evaluate(k * 0.05)
        assert math.hypot(x, y) <= safe + 1e-9


def test_random_spline_is_periodic():
    ref = sample("random_spline", _rng(7), _plate(), ball_radius=0.02)
    a = ref.evaluate(0.3)
    b = ref.evaluate(0.3 + ref.period)
    for u, v in zip(a, b, strict=True):
        assert math.isclose(u, v, abs_tol=1e-9)

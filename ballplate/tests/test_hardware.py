"""Tests for the ``ballplate.hardware`` YAML loader."""

import textwrap

from ballplate.hardware import (
    AdapterSpec,
    BallSpec,
    HardwareSpec,
    PhysicsSpec,
    PlateSpec,
    default_path,
    load,
)


def _write_yaml(tmp_path, body: str):
    p = tmp_path / "hardware.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_returns_typed_dataclasses(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        plate:
          size:      [0.50, 0.30]
          thickness: 0.004
          mass:      0.15
          friction:  0.40
        ball:
          radius:    0.02
          mass:      0.0027
          friction:  0.50
        physics:
          gravity:        9.81
          rolling_factor: 0.71428571
        adapter:
          position:    [0.01, 0.10, -0.02]
          orientation: [-1.5708, 0.0, 0.5]
        arm:
          speed_slider_fraction: 0.2
    """,
    )
    spec = load(path)
    assert isinstance(spec, HardwareSpec)
    assert isinstance(spec.plate, PlateSpec)
    assert isinstance(spec.ball, BallSpec)
    assert isinstance(spec.physics, PhysicsSpec)
    assert isinstance(spec.adapter, AdapterSpec)


def test_load_preserves_values(tmp_path):
    path = _write_yaml(
        tmp_path,
        """
        plate:
          size:      [0.70, 0.50]
          thickness: 0.005
          mass:      0.2
          friction:  0.5
        ball:
          radius:    0.020
          mass:      0.0027
          friction:  0.5
        physics:
          gravity:        9.81
          rolling_factor: 0.7142857
        adapter:
          position:    [0.0, 0.1, 0.0]
          orientation: [-1.5708, 0.0, 0.0]
        arm:
          speed_slider_fraction: 0.2
    """,
    )
    spec = load(path)
    assert spec.plate.size == (0.70, 0.50)
    assert spec.plate.thickness == 0.005
    assert spec.plate.mass == 0.2
    assert spec.plate.friction == 0.5
    assert spec.ball.radius == 0.020
    assert spec.ball.mass == 0.0027
    assert spec.physics.gravity == 9.81
    assert spec.physics.rolling_factor == 0.7142857
    assert spec.adapter.position == (0.0, 0.1, 0.0)
    assert spec.adapter.orientation == (-1.5708, 0.0, 0.0)


def test_load_coerces_int_to_float(tmp_path):
    # Integer YAML values must come out as floats so downstream consumers
    # can rely on float arithmetic.
    path = _write_yaml(
        tmp_path,
        """
        plate:
          size:      [1, 1]
          thickness: 1
          mass:      1
          friction:  1
        ball:
          radius:    1
          mass:      1
          friction:  1
        physics:
          gravity:        10
          rolling_factor: 1
        adapter:
          position:    [0, 0, 0]
          orientation: [0, 0, 0]
        arm:
          speed_slider_fraction: 1
    """,
    )
    spec = load(path)
    for v in (
        spec.plate.thickness,
        spec.plate.mass,
        spec.plate.friction,
        spec.ball.radius,
        spec.ball.mass,
        spec.ball.friction,
        spec.physics.gravity,
        spec.physics.rolling_factor,
    ):
        assert isinstance(v, float)
    assert all(isinstance(v, float) for v in spec.adapter.position)
    assert all(isinstance(v, float) for v in spec.adapter.orientation)


def test_default_path_resolves_to_monorepo_config():
    p = default_path()
    assert p.name == "hardware.yaml"
    assert p.parent.name == "config"
    assert p.exists()

"""Tests for the ``from_dict`` and ``from_named_dict`` preset helpers."""

import pytest

from ballplate.controllers.pid import presets
from ballplate.controllers.pid.controller import PidGains


def test_from_dict_with_all_fields():
    g = presets.from_dict(
        {"kp": 1.0, "ki": 0.5, "kd": 0.2, "kff": 0.1, "windup_limit": 0.05, "max_output": 2.0}
    )
    assert g == PidGains(kp=1.0, ki=0.5, kd=0.2, kff=0.1, windup_limit=0.05, max_output=2.0)


def test_from_dict_uses_defaults_for_missing_keys():
    g = presets.from_dict({"kp": 1.0, "ki": 0.0, "kd": 0.0})
    assert g.kff == 0.0
    assert g.windup_limit == float("inf")


def test_from_dict_rejects_unknown_keys():
    # Unknown keys are bugs in callers' YAML files; surface them loudly.
    with pytest.raises(TypeError):
        presets.from_dict({"kp": 1.0, "ki": 0.0, "kd": 0.0, "typo": 0.0})


def test_from_named_dict_picks_the_named_block():
    catalog = {
        "soft": {"kp": 0.5, "ki": 0.0, "kd": 0.1},
        "stiff": {"kp": 2.0, "ki": 0.1, "kd": 0.4},
    }
    soft = presets.from_named_dict(catalog, "soft")
    stiff = presets.from_named_dict(catalog, "stiff")
    assert soft.kp == 0.5
    assert stiff.kp == 2.0


def test_from_named_dict_unknown_name_raises_key_error():
    with pytest.raises(KeyError):
        presets.from_named_dict({"a": {"kp": 1.0, "ki": 0.0, "kd": 0.0}}, "b")

"""Tests for the ``deadband`` and ``velocity_clip`` primitives in ``ballplate.safety``."""

from ballplate.safety import deadband, velocity_clip

# --------------------------------------------------------------------- deadband


def test_deadband_zeros_small_errors():
    assert deadband(0.0005, threshold=0.001) == 0.0
    assert deadband(-0.0009, threshold=0.001) == 0.0


def test_deadband_passes_large_errors():
    assert deadband(0.005, threshold=0.001) == 0.005
    assert deadband(-0.005, threshold=0.001) == -0.005


def test_deadband_at_threshold_is_zeroed():
    # Inclusive boundary: a value exactly equal to the threshold falls
    # inside the dead zone, so the output is 0 not the input.
    assert deadband(0.001, threshold=0.001) == 0.0
    assert deadband(-0.001, threshold=0.001) == 0.0


def test_deadband_threshold_sign_does_not_matter():
    # Negative thresholds would be a misuse but should not flip the logic.
    assert deadband(0.0005, threshold=-0.001) == 0.0


# --------------------------------------------------------------- velocity_clip


def test_velocity_clip_passes_values_within_band():
    assert velocity_clip(0.05, limit=0.1) == 0.05
    assert velocity_clip(-0.05, limit=0.1) == -0.05


def test_velocity_clip_caps_above_limit():
    assert velocity_clip(0.5, limit=0.1) == 0.1


def test_velocity_clip_caps_below_negative_limit():
    assert velocity_clip(-0.5, limit=0.1) == -0.1


def test_velocity_clip_limit_sign_does_not_matter():
    assert velocity_clip(0.5, limit=-0.1) == 0.1

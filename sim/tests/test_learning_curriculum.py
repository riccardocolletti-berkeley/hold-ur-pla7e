"""Tests for sim.learning.curriculum."""

from sim.learning.curriculum import shapes_for_step


def _stages():
    return [
        {"until_steps": 100, "shapes": ["stationary"]},
        {"until_steps": 500, "shapes": ["stationary", "circle"]},
        {"until_steps": 1_000_000_000, "shapes": ["circle", "figure8"]},
    ]


def test_picks_first_stage_for_low_step():
    assert shapes_for_step(_stages(), 0) == ["stationary"]
    assert shapes_for_step(_stages(), 99) == ["stationary"]


def test_advances_to_second_stage_at_boundary():
    # `until_steps` is the upper bound of the previous stage (strict <).
    assert shapes_for_step(_stages(), 100) == ["stationary", "circle"]


def test_falls_back_to_last_stage_past_all_bounds():
    # Past every `until_steps` value the sentinel last stage is returned.
    far_future = 10**12
    assert shapes_for_step(_stages(), far_future) == ["circle", "figure8"]


def test_returned_list_is_a_copy():
    # Mutating the returned list must not corrupt the stages.
    stages = _stages()
    out = shapes_for_step(stages, 0)
    out.append("bogus")
    assert "bogus" not in stages[0]["shapes"]

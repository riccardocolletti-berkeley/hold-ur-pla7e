"""Curriculum: pick the active set of trajectory shapes from the global step.

The training config declares an ordered list of stages; each stage names the
shapes to sample from until ``until_steps`` is reached. The last stage in the
list is treated as a sentinel and used past its `until_steps` value so the
training run never runs out of shapes.
"""

from collections.abc import Sequence


def shapes_for_step(stages: Sequence[dict], step: int) -> list[str]:
    """Return the shape list of the first stage whose `until_steps` is not yet exceeded."""
    for stage in stages:
        if step < stage["until_steps"]:
            return list(stage["shapes"])
    return list(stages[-1]["shapes"])

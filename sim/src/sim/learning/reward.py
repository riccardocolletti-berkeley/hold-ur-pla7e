"""Reward function for the ball-on-plate RL task.

Three building blocks (all weights live in ``config/learning/reward.yaml``):

    * Gaussian tracking term: ``exp(-||err|| / sigma)`` is bounded in
      ``[0, 1]`` and provides a smooth gradient at every distance, fixing
      the "always negative" pathology of pure quadratic penalties.

    * Potential-based progress term: rewards reducing the position error
      between consecutive steps. Proven to leave the optimal policy
      unchanged (Ng et al. 1999) while accelerating tracking convergence.

    * Proportional drop penalty: cost ∝ remaining episode fraction so an
      early drop is much worse than a late one. Removes the "drift around
      then drop" local optimum we hit during early experiments.
"""

import math
from collections.abc import Mapping
from pathlib import Path

import yaml


def load_weights(path: Path) -> dict:
    """Load reward weights from a YAML mapping file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def step_reward(
    ball_pos: tuple[float, float],
    ball_vel: tuple[float, float],
    target_pos: tuple[float, float],
    target_vel: tuple[float, float],
    action: tuple[float, float],
    prev_action: tuple[float, float],
    prev_pos_err: float | None,
    dropped: bool,
    steps_remaining_frac: float,
    weights: Mapping[str, float],
) -> tuple[float, float]:
    """Return ``(reward, current_pos_err)``.

    `prev_pos_err` carries forward across steps for the potential-based shaping
    term; pass ``None`` on the first step of an episode. `steps_remaining_frac`
    is consumed only when `dropped` is True, where it scales the drop penalty.
    """
    bx, by = ball_pos
    vx, vy = ball_vel
    tx, ty = target_pos
    tvx, tvy = target_vel

    pos_err = math.hypot(bx - tx, by - ty)
    vel_err = math.hypot(vx - tvx, vy - tvy)

    r_track = math.exp(-pos_err / weights["sigma_pos"])
    r_vel = math.exp(-vel_err / weights["sigma_vel"])

    progress = (prev_pos_err - pos_err) if prev_pos_err is not None else 0.0

    act2 = action[0] ** 2 + action[1] ** 2
    dact2 = (action[0] - prev_action[0]) ** 2 + (action[1] - prev_action[1]) ** 2

    r = (
        weights["w_track"] * r_track
        + weights["w_vel"] * r_vel
        + weights["w_prog"] * progress
        - weights["w_act"] * act2
        - weights["w_smooth"] * dact2
    )

    if dropped:
        r -= weights["w_drop"] * float(steps_remaining_frac)

    return r, pos_err

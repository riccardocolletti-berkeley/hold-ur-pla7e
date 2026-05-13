"""Domain randomization: sample physical parameters and apply them in place.

Each episode draws one set of parameters from the per-key uniform ranges
defined in ``config/learning/randomize.yaml``. The sampled bundle is then
written into the live `MjModel` (mass, friction, gravity); the env keeps the
remaining keys (observation noise, action delay, ball init radius) in
Python-space because they are not stored on the model.
"""

from collections.abc import Mapping
from pathlib import Path

import mujoco
import numpy as np
import yaml


def load_ranges(path: Path) -> dict[str, tuple[float, float]]:
    """Load the per-key ``[min, max]`` mapping from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {k: (float(lo), float(hi)) for k, (lo, hi) in data.items()}


def sample(
    ranges: Mapping[str, tuple[float, float]],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Draw one parameter set; one float per key, uniform inside ``[lo, hi]``."""
    return {key: float(rng.uniform(lo, hi)) for key, (lo, hi) in ranges.items()}


def apply(
    model,
    data,
    params: Mapping[str, float],
    ball_geom_id: int,
    plate_geom_id: int,
    ball_body_id: int,
) -> None:
    """Mutate `model` so the next physics step uses the sampled parameters.

    When mass changes, MuJoCo's pre-computed quantities (body_inertia,
    body_subtree_mass, body_invweight0, dof_M0/dof_invweight0) become stale
    and the rolling-contact solver goes unstable. We rescale body_inertia
    explicitly (sphere with fixed radius: I ∝ m) and call ``mj_setConst``
    to refresh every other constant derived from mass.
    """
    mass_changed = False
    if "ball_mass" in params:
        old_mass = float(model.body_mass[ball_body_id])
        new_mass = float(params["ball_mass"])
        if old_mass > 0.0:
            model.body_inertia[ball_body_id] *= new_mass / old_mass
        model.body_mass[ball_body_id] = new_mass
        mass_changed = True
    if "ball_friction" in params:
        # Sliding-friction component only; rolling / spin keep the scene defaults.
        model.geom_friction[ball_geom_id, 0] = params["ball_friction"]
    if "plate_friction" in params:
        model.geom_friction[plate_geom_id, 0] = params["plate_friction"]
    if "gravity_scale" in params:
        model.opt.gravity[2] = -9.81 * params["gravity_scale"]
    if mass_changed:
        mujoco.mj_setConst(model, data)

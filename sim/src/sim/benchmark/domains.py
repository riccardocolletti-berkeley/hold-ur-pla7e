"""Named environment configurations for the PID vs PID+RL comparison.

Each :class:`Domain` bundles two pieces:

* ``env_cfg``: the patch applied on top of the training env block (matches
  the structure of ``train.yaml::env``). Determines actuator dynamics, the
  command low-pass, the action and velocity clips, etc.
* ``dr_ranges``: a fully-specified DR mapping (every key from
  ``randomize.yaml`` present, even if its lo/hi are pinned to the nominal
  value). Keeping all keys present means the env never touches a missing
  one and bisecting a single dimension is a one-line edit.

Adding a domain is one entry in :data:`DOMAINS`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    """One named comparison surface."""

    name: str
    description: str
    env_cfg: dict
    dr_ranges: dict


# Defaults shared across every domain. Episode length / policy rate / the
# residual action_scale must mirror the values the policy was trained on,
# otherwise the benchmark would compare PPO at a deploy rate it never saw.
_BASE_ENV: dict = {
    "episode_steps": 600,
    "policy_hz": 30,
    "action_scale": 0.3,
    "lookahead_s": 0.2,
}

# Nominal values: DR effectively off (every range collapsed to one point).
_DR_OFF: dict = {
    "ball_mass": [0.0027, 0.0027],
    "ball_friction": [0.50, 0.50],
    "plate_friction": [0.50, 0.50],
    "ball_init_radius": [0.0, 0.0],
    "gravity_scale": [1.0, 1.0],
    "obs_noise_std": [0.0, 0.0],
    "action_delay_ms": [0.0, 0.0],
}

# Full DR: same numbers as ``config/learning/randomize.yaml``.
_DR_FULL: dict = {
    "ball_mass": [0.0020, 0.0040],
    "ball_friction": [0.30, 0.70],
    "plate_friction": [0.30, 0.80],
    "ball_init_radius": [0.0, 0.020],
    "gravity_scale": [0.98, 1.02],
    "obs_noise_std": [0.0, 0.003],
    "action_delay_ms": [30.0, 150.0],
}

# Realistic actuator + safety constraints used at deploy / training.
_REAL_CONSTRAINTS: dict = {
    "command_alpha": 0.70,
    "actuator_tau_s": 0.15,
    "max_tilt_rad": 0.060,
    "omega_clip_rad_s": 1.0,
    "velocity_clip_mps": 0.120,
}

# Permissive constraints matching ``sim.runner --controller pid``.
_LOOSE_CONSTRAINTS: dict = {
    "command_alpha": 1.0,
    "max_tilt_rad": 0.1,
}


DOMAINS: dict[str, Domain] = {
    "standard": Domain(
        name="standard",
        description="Loose env (sim.runner-like): no DR, fast actuator.",
        env_cfg={**_BASE_ENV, **_LOOSE_CONSTRAINTS},
        dr_ranges=_DR_OFF,
    ),
    "slippery": Domain(
        name="slippery",
        description="Real env + low friction (μ=0.20): ice-like contacts.",
        env_cfg={**_BASE_ENV, **_REAL_CONSTRAINTS},
        dr_ranges={
            **_DR_OFF,
            "ball_friction": [0.20, 0.20],
            "plate_friction": [0.20, 0.20],
        },
    ),
    "sticky": Domain(
        name="sticky",
        description="Real env + high friction (μ=0.90): rubber-like contacts.",
        env_cfg={**_BASE_ENV, **_REAL_CONSTRAINTS},
        dr_ranges={
            **_DR_OFF,
            "ball_friction": [0.90, 0.90],
            "plate_friction": [0.90, 0.90],
        },
    ),
    "heavy_ball": Domain(
        name="heavy_ball",
        description="Real env + 5 g ball (1.85× ITTF): more inertia.",
        env_cfg={**_BASE_ENV, **_REAL_CONSTRAINTS},
        dr_ranges={**_DR_OFF, "ball_mass": [0.0050, 0.0050]},
    ),
    "low_gravity": Domain(
        name="low_gravity",
        description="Real env + half g (4.9 m/s²): slower rolling.",
        env_cfg={**_BASE_ENV, **_REAL_CONSTRAINTS},
        dr_ranges={**_DR_OFF, "gravity_scale": [0.50, 0.50]},
    ),
    "high_gravity": Domain(
        name="high_gravity",
        description="Real env + 1.5× g (14.7 m/s²): faster rolling.",
        env_cfg={**_BASE_ENV, **_REAL_CONSTRAINTS},
        dr_ranges={**_DR_OFF, "gravity_scale": [1.50, 1.50]},
    ),
    "noisy_vision": Domain(
        name="noisy_vision",
        description="Real env + 5 mm camera noise: beyond training (3 mm).",
        env_cfg={**_BASE_ENV, **_REAL_CONSTRAINTS},
        dr_ranges={**_DR_OFF, "obs_noise_std": [0.005, 0.005]},
    ),
    "high_latency": Domain(
        name="high_latency",
        description="Real env + 200 ms delay: beyond training (≤150 ms).",
        env_cfg={**_BASE_ENV, **_REAL_CONSTRAINTS},
        dr_ranges={**_DR_OFF, "action_delay_ms": [200.0, 200.0]},
    ),
    "full_dr": Domain(
        name="full_dr",
        description="Training surface: full DR + realistic actuator.",
        env_cfg={**_BASE_ENV, **_REAL_CONSTRAINTS},
        dr_ranges=_DR_FULL,
    ),
    "extreme_dr": Domain(
        name="extreme_dr",
        description="Worst-case combo: max delay + slippery contacts.",
        env_cfg={**_BASE_ENV, **_REAL_CONSTRAINTS},
        dr_ranges={
            **_DR_FULL,
            "action_delay_ms": [120.0, 150.0],
            "ball_friction": [0.30, 0.40],
            "plate_friction": [0.30, 0.40],
        },
    ),
}

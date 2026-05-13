"""Stable-Baselines 3 PPO policy wrapped as a ball-on-plate controller.

Top-level import of ``stable_baselines3``: only loaded under the ``[rl]``
extra. The wrapper takes a pre-built observation vector (see
``ballplate.controllers.rl.observation.build_observation``) and returns
the 2-D virtual control ``(ux, uy)`` in the PID's units, so the same
downstream wiring (Jacobian, joint PD, ROS publishing) carries over.
"""

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from ballplate.controllers.rl.loader import (
    load_meta,
    load_obs_rms,
    obs_rms_path_for_checkpoint,
    pick_checkpoint,
)
from ballplate.controllers.rl.observation import normalize_observation


def _optional_float(meta: dict[str, Any], name: str) -> float | None:
    return None if name not in meta else float(meta[name])


class RLPolicy:
    """Loaded PPO policy plus the observation calibration saved at training."""

    def __init__(self, run_dir: str | Path):
        run_dir = Path(run_dir)
        ckpt = pick_checkpoint(run_dir)
        if ckpt is None:
            raise FileNotFoundError(
                f"No usable checkpoint in {run_dir!s}. "
                "Looked for best_model.zip, ppo_final.zip, ppo_*_steps.zip."
            )
        self._policy = PPO.load(str(ckpt), device="cpu")

        meta = load_meta(run_dir / "env_meta.json")
        self.action_scale: float = float(meta.get("action_scale", 1.0))
        lookahead = meta.get("lookahead_s", 0.2)
        self.lookahead_s: float = float(
            lookahead[-1] if isinstance(lookahead, list | tuple) else lookahead
        )
        self.policy_hz: float = float(meta.get("policy_hz", 50.0))
        self.omega_clip_rad_s = _optional_float(meta, "omega_clip_rad_s")
        self.max_tilt_rad = _optional_float(meta, "max_tilt_rad")
        self.command_alpha: float = float(meta.get("command_alpha", 1.0))
        self.actuator_tau_s = _optional_float(meta, "actuator_tau_s")
        self.velocity_clip_mps = _optional_float(meta, "velocity_clip_mps")

        self._obs_mean, self._obs_std, self._obs_clip = load_obs_rms(
            obs_rms_path_for_checkpoint(ckpt)
        )

    # ----------------------------------------------------------- inference --

    def predict(self, observation: np.ndarray) -> tuple[float, float]:
        """Return ``(ux, uy)`` for a pre-built observation vector.

        Observation-normalisation stats (if saved at training time) are
        applied here. The action is scaled by ``action_scale`` so units
        match what the PID emits.
        """
        obs = observation
        if self._obs_mean is not None and self._obs_std is not None:
            obs = normalize_observation(obs, self._obs_mean, self._obs_std, self._obs_clip)
        action, _ = self._policy.predict(obs, deterministic=True)
        return (
            float(action[0]) * self.action_scale,
            float(action[1]) * self.action_scale,
        )

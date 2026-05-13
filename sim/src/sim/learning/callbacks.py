"""SB3 callbacks: metrics dumping, curriculum, scheduling, and saving.

Everything that the training loop needs to track or persist lives here so
`sim.learning.train` stays a thin orchestration script.
"""

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from sim.learning.curriculum import shapes_for_step


class JSONScalarLogger(BaseCallback):
    """Append a flat JSON record per rollout to ``<run>/metrics.jsonl``.

    The iPad designer dashboard tails this file to chart training progress
    without parsing TensorBoard event files.
    """

    KEYS_OF_INTEREST = (
        "rollout/ep_rew_mean",
        "rollout/ep_len_mean",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/entropy_loss",
        "train/approx_kl",
        "train/learning_rate",
        "time/fps",
    )

    def __init__(self, out_path: Path, every_n_rollouts: int = 1):
        super().__init__()
        self.out_path = Path(out_path)
        self.every_n = max(1, int(every_n_rollouts))
        self._count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._count += 1
        if self._count % self.every_n != 0:
            return

        scalars = dict(self.model.logger.name_to_value)
        ep_buf = self.model.ep_info_buffer or []
        ep_rew = float(np.mean([x["r"] for x in ep_buf])) if ep_buf else None
        ep_len = float(np.mean([x["l"] for x in ep_buf])) if ep_buf else None

        record = {
            "step": int(self.num_timesteps),
            "time": time.time(),
            "ep_rew_mean": ep_rew,
            "ep_len_mean": ep_len,
        }
        for key in self.KEYS_OF_INTEREST:
            value = scalars.get(key)
            if isinstance(value, int | float):
                record[key.split("/")[-1]] = float(value)

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "a") as f:
            f.write(json.dumps(record) + "\n")


class CurriculumCallback(BaseCallback):
    """Push the active shape list into every parallel env at rollout end."""

    def __init__(self, stages: Sequence[dict]):
        super().__init__()
        self.stages = list(stages)
        self._last_shapes = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        shapes = shapes_for_step(self.stages, self.num_timesteps)
        if shapes == self._last_shapes:
            return
        # `env_method` traverses gym wrappers (Monitor) via `__getattr__`;
        # `set_attr` would land on the wrapper rather than the inner env.
        self.training_env.env_method("set_shapes", list(shapes))
        self._last_shapes = list(shapes)
        if self.verbose:
            print(f"[curriculum] step={self.num_timesteps} shapes={shapes}")


class LinearScheduleCallback(BaseCallback):
    """Linearly anneal a scalar attribute on the model across `total_timesteps`.

    Used for `ent_coef` so PPO explores aggressively early and exploits late.
    """

    def __init__(self, attr_name: str, start: float, end: float, total_timesteps: int):
        super().__init__()
        self.attr_name = str(attr_name)
        self.start = float(start)
        self.end = float(end)
        self.total = max(1, int(total_timesteps))

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total)
        value = self.start + (self.end - self.start) * progress
        setattr(self.model, self.attr_name, value)
        return True


class BestModelCallback(BaseCallback):
    """Save the policy plus VecNormalize stats whenever ``ep_rew_mean`` peaks.

    Skips the first few rollouts so the running mean has stabilised before
    "best so far" is meaningful.
    """

    def __init__(
        self,
        save_dir: Path,
        min_episodes: int = 50,
        save_freq_rollouts: int = 1,
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.best = -float("inf")
        self.min_episodes = int(min_episodes)
        self.save_freq = max(1, int(save_freq_rollouts))
        self._count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        self._count += 1
        if self._count % self.save_freq != 0:
            return
        ep_buf = self.model.ep_info_buffer or []
        if len(ep_buf) < self.min_episodes:
            return
        rew = float(np.mean([x["r"] for x in ep_buf]))
        if rew <= self.best:
            return
        self.best = rew

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(self.save_dir / "best_model"))

        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            np.savez(
                self.save_dir / "best_obs_rms.npz",
                mean=venv.obs_rms.mean,
                var=venv.obs_rms.var,
                epsilon=np.float64(venv.epsilon),
                clip_obs=np.float64(venv.clip_obs),
            )
        if self.verbose:
            print(f"[best] new best ep_rew_mean={rew:.2f} at step {self.num_timesteps}")


class CheckpointWithStatsCallback(BaseCallback):
    """Periodic snapshot of the policy paired with the matching `obs_rms.npz`.

    Each saved checkpoint can be deployed standalone because the
    obs-normalisation stats are persisted alongside the SB3 zip file.
    """

    def __init__(self, save_freq: int, save_dir: Path, name_prefix: str = "ppo"):
        super().__init__()
        self.save_freq = max(1, int(save_freq))
        self.save_dir = Path(save_dir)
        self.name_prefix = str(name_prefix)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True
        step = self.num_timesteps
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model.save(str(self.save_dir / f"{self.name_prefix}_{step}_steps"))
        venv = self.model.get_vec_normalize_env()
        if venv is not None:
            np.savez(
                self.save_dir / f"obs_rms_{step}_steps.npz",
                mean=venv.obs_rms.mean,
                var=venv.obs_rms.var,
                epsilon=np.float64(venv.epsilon),
                clip_obs=np.float64(venv.clip_obs),
            )
        return True

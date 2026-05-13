"""PPO training entry point.

Usage::

    python -m sim.learning.train [--config sim/config/learning/train.yaml]

The script wires together the environment, the SB3 callbacks, and an
optional resume path, then drives ``model.learn(...)`` to completion. When
training writes the policy, the matching VecNormalize stats, and the env meta
(action_scale, lookahead_s, policy_hz) under ``sim/policies/<name>/`` so the
deploy-time `RLPolicy` can pick them up.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from sim.learning import curriculum
from sim.learning.callbacks import (
    BestModelCallback,
    CheckpointWithStatsCallback,
    CurriculumCallback,
    JSONScalarLogger,
    LinearScheduleCallback,
)
from sim.learning.env import BallPlateEnv
from sim.learning.randomize import load_ranges as load_randomize_ranges
from sim.learning.reward import load_weights as load_reward_weights

POLICIES_DIR = Path(__file__).resolve().parents[3] / "policies"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _make_env(sim_cfg, train_cfg, dr_ranges, reward_weights, shapes, seed):
    """Build a thunk that constructs one BallPlateEnv wrapped by `Monitor`.

    The Monitor wrapper populates ``ep_info_buffer`` so SB3 can log
    ``rollout/ep_rew_mean`` and ``rollout/ep_len_mean`` on the parent process.
    """

    def _thunk():
        env = BallPlateEnv(
            sim_cfg=sim_cfg,
            env_cfg=train_cfg["env"],
            dr_ranges=dr_ranges,
            reward_weights=reward_weights,
            shapes=shapes,
            seed=seed,
        )
        return Monitor(env)

    return _thunk


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sim.learning.train")
    parser.add_argument("--config", default="sim/config/learning/train.yaml")
    parser.add_argument("--sim-config", default="sim/config/sim.yaml")
    parser.add_argument("--randomize-config", default="sim/config/learning/randomize.yaml")
    parser.add_argument("--reward-config", default="sim/config/learning/reward.yaml")
    parser.add_argument("--name", default="ppo_run")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "auto"],
        help="Torch device for the policy. CPU is fastest for small MLPs since "
        "MuJoCo is the real bottleneck.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed; each parallel env gets seed * 1000 + i.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from <run>/ppo_final.zip + vecnormalize.pkl. "
        "`total_timesteps` in the config becomes the EXTRA budget.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Specific checkpoint filename inside <run> (e.g. best_model.zip).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_cfg = _load_yaml(Path(args.config))
    sim_cfg = _load_yaml(Path(args.sim_config))
    dr_ranges = load_randomize_ranges(Path(args.randomize_config))
    reward_weights = load_reward_weights(Path(args.reward_config))

    POLICIES_DIR.mkdir(exist_ok=True)
    run_dir = POLICIES_DIR / args.name
    resuming = args.resume or args.resume_from is not None
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save this before learning so interrupted runs and live checkpoints are
    # deployable without waiting for the final save path.
    with open(run_dir / "env_meta.json", "w") as f:
        json.dump(train_cfg["env"], f, indent=2)

    # Initial shape list comes from the curriculum's first stage; subsequent
    # rollouts are advanced by `CurriculumCallback`.
    shapes = curriculum.shapes_for_step(train_cfg["curriculum"]["stages"], 0)
    env_fns = [
        _make_env(
            sim_cfg=sim_cfg,
            train_cfg=train_cfg,
            dr_ranges=dr_ranges,
            reward_weights=reward_weights,
            shapes=shapes,
            seed=args.seed * 1000 + i,
        )
        for i in range(train_cfg["n_envs"])
    ]
    base_venv = SubprocVecEnv(env_fns)

    # When resuming, reuse the VecNormalize statistics from the previous run.
    vn_path = run_dir / "vecnormalize.pkl"
    if resuming and vn_path.exists():
        venv = VecNormalize.load(str(vn_path), base_venv)
        # SB3 sets training=False on load; re-enable so stats keep updating.
        venv.training = True
    else:
        venv = VecNormalize(
            base_venv,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=train_cfg["ppo"]["gamma"],
        )

    sched_cfg = train_cfg.get("schedule", {})
    callbacks = CallbackList(
        [
            CheckpointWithStatsCallback(
                save_freq=max(
                    1,
                    train_cfg.get("save_every_steps", 200_000) // train_cfg["n_envs"],
                ),
                save_dir=run_dir,
                name_prefix="ppo",
            ),
            JSONScalarLogger(out_path=run_dir / "metrics.jsonl"),
            CurriculumCallback(stages=train_cfg["curriculum"]["stages"]),
            BestModelCallback(save_dir=run_dir, min_episodes=80),
            LinearScheduleCallback(
                attr_name="ent_coef",
                start=sched_cfg.get("ent_coef_start", train_cfg["ppo"]["ent_coef"]),
                end=sched_cfg.get("ent_coef_end", train_cfg["ppo"]["ent_coef"]),
                total_timesteps=train_cfg["total_timesteps"],
            ),
        ]
    )

    # Learning-rate decay via SB3's native schedule. `progress_remaining`
    # starts at 1.0 and ends at 0.0 over `total_timesteps`.
    lr_start = float(sched_cfg.get("lr_start", train_cfg["ppo"]["learning_rate"]))
    lr_end = float(sched_cfg.get("lr_end", lr_start))
    if lr_end != lr_start:

        def lr_schedule(progress_remaining: float) -> float:
            return lr_end + (lr_start - lr_end) * progress_remaining

        train_cfg["ppo"]["learning_rate"] = lr_schedule

    device = args.device
    if device == "auto":
        import torch

        device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[train] device = {device}, n_envs = {train_cfg['n_envs']}")

    pol_kwargs = dict(net_arch=train_cfg["policy_kwargs"]["net_arch"])
    activation_name = train_cfg["policy_kwargs"].get("activation_fn")
    if activation_name:
        import torch.nn as nn

        activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}
        if activation_name not in activations:
            raise SystemExit(
                f"Unsupported activation_fn {activation_name!r}; "
                f"choose one of {sorted(activations)}"
            )
        pol_kwargs["activation_fn"] = activations[activation_name]
    if resuming:
        ckpt = run_dir / (args.resume_from or "ppo_final.zip")
        if not ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
        model = PPO.load(str(ckpt), env=venv, device=device)
        print(f"[train] resumed from {ckpt} at step {model.num_timesteps}")
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            device=device,
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(run_dir / "tb"),
            policy_kwargs=pol_kwargs,
            **train_cfg["ppo"],
        )

    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=callbacks,
        reset_num_timesteps=not resuming,
    )
    model.save(str(run_dir / "ppo_final"))

    # Persist obs-normalization stats next to the policy so the deploy-time
    # `RLPolicy` can rebuild the same input pipeline.
    np.savez(
        run_dir / "obs_rms.npz",
        mean=venv.obs_rms.mean,
        var=venv.obs_rms.var,
        epsilon=np.float64(venv.epsilon),
        clip_obs=np.float64(venv.clip_obs),
    )
    venv.save(str(run_dir / "vecnormalize.pkl"))


if __name__ == "__main__":
    main()

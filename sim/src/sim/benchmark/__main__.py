"""Benchmark entry point: PID vs PID+RL across named domains.

Usage::

    uv run python -m sim.benchmark --rl-run-dir sim/policies/ppo_residual_v2

For each ``(domain, policy, seed)`` it constructs a fresh ``BallPlateEnv``
with the domain's env / DR overrides, rolls out one episode, and aggregates
the per-trace metrics. The output dir holds ``summary.csv`` plus one PNG
per metric and per-domain trajectory overlay, ready for the report.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO

from sim.benchmark import metrics as M
from sim.benchmark.domains import DOMAINS, Domain
from sim.benchmark.plot import plot_metric_bars, plot_trajectory_overlay
from sim.benchmark.rollout import EpisodeTrace, run_episode
from sim.learning.env import BallPlateEnv
from sim.learning.reward import load_weights

#: Repo root, walked from this file so the script is CWD-independent.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _load_policy(run_dir: Path):
    """Return a callable ``predict_fn(obs) -> action`` for a trained PPO run.

    The benchmark talks to the env in raw ``[-1, 1]`` action space because
    ``_apply_action`` multiplies by ``action_scale`` itself; we therefore
    bypass ``ballplate.RLPolicy`` (which would multiply once more) and call
    SB3 directly, applying the saved VecNormalize stats so the network sees
    the same input distribution it trained on.
    """
    model = PPO.load(str(run_dir / "best_model.zip"), device="cpu")
    obs_path = run_dir / "best_obs_rms.npz"
    if not obs_path.is_file():
        obs_path = run_dir / "obs_rms.npz"
    if not obs_path.is_file():
        raise FileNotFoundError(
            f"No obs-norm stats found in {run_dir}. Looked for "
            f"best_obs_rms.npz and obs_rms.npz."
        )
    data = np.load(obs_path)
    mean = data["mean"].astype(np.float32)
    std = np.sqrt(data["var"].astype(np.float32) + float(data["epsilon"]))
    clip = float(data["clip_obs"])

    def predict(obs: np.ndarray) -> np.ndarray:
        norm = np.clip((obs - mean) / std, -clip, clip).astype(np.float32)
        action, _ = model.predict(norm, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    return predict


def _build_env(domain: Domain, sim_cfg: dict, reward_weights: dict, seed: int) -> BallPlateEnv:
    """Construct one env for the given domain + seed; one episode per env."""
    return BallPlateEnv(
        sim_cfg=sim_cfg,
        env_cfg=domain.env_cfg,
        dr_ranges=domain.dr_ranges,
        reward_weights=reward_weights,
        shapes=("circle",),
        seed=seed,
    )


def _zero_action(_obs: np.ndarray) -> np.ndarray:
    """PID-only baseline: residual contribution is zero."""
    return np.zeros(2, dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sim.benchmark")
    parser.add_argument(
        "--rl-run-dir",
        required=True,
        help="Trained PPO run dir (best_model.zip + best_obs_rms.npz).",
    )
    parser.add_argument("--seeds", type=int, default=30, help="Episodes per (domain, policy) pair.")
    parser.add_argument(
        "--domains", default=",".join(DOMAINS), help="Comma-separated subset of named domains."
    )
    parser.add_argument(
        "--out-dir", default=None, help="Output dir; defaults to sim/results/<timestamp>."
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.rl_run_dir).resolve()

    with open(_PROJECT_ROOT / "sim" / "config" / "sim.yaml") as f:
        sim_cfg = yaml.safe_load(f)
    reward_weights = load_weights(_PROJECT_ROOT / "sim" / "config" / "learning" / "reward.yaml")

    requested = [d.strip() for d in args.domains.split(",") if d.strip()]
    unknown = [d for d in requested if d not in DOMAINS]
    if unknown:
        raise SystemExit(f"Unknown domain(s): {unknown}. Available: {sorted(DOMAINS)}")
    selected = [DOMAINS[name] for name in requested]

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else _PROJECT_ROOT / "sim" / "results" / time.strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rl_predict = _load_policy(run_dir)
    policies = {"pid": _zero_action, "residual": rl_predict}

    aggregates: dict[str, dict[str, dict[str, float]]] = {}
    sample_traces: dict[str, dict[str, EpisodeTrace]] = {}

    for domain in selected:
        print(f"== domain: {domain.name}  ({domain.description})")
        agg_per_policy: dict[str, dict[str, float]] = {}
        sample_per_policy: dict[str, EpisodeTrace] = {}
        for label, policy_fn in policies.items():
            traces: list[EpisodeTrace] = []
            for s in range(args.seeds):
                env = _build_env(domain, sim_cfg, reward_weights, seed=s)
                traces.append(
                    run_episode(env, policy_fn, max_steps=int(domain.env_cfg["episode_steps"]))
                )
            agg = M.aggregate(traces)
            agg_per_policy[label] = agg
            sample_per_policy[label] = traces[0]
            print(
                f"  {label:9s} drop={agg['drop_rate']:.2f}"
                f"  surv={agg['survival_steps_mean']:5.0f} steps"
                f"  rmse={agg['rmse_mean'] * 1000.0:5.1f} mm"
            )
        aggregates[domain.name] = agg_per_policy
        sample_traces[domain.name] = sample_per_policy
        plot_trajectory_overlay(
            sample_per_policy,
            out_dir / f"trajectory_{domain.name}.png",
            domain.name,
        )

    plot_metric_bars(
        aggregates, "drop_rate", out_dir / "drop_rate.png", "Drop rate (lower is better)"
    )
    plot_metric_bars(aggregates, "rmse_mean", out_dir / "rmse_mean.png", "Mean tracking RMSE [m]")
    plot_metric_bars(
        aggregates,
        "survival_steps_mean",
        out_dir / "survival_steps.png",
        "Mean episode length [steps]",
    )

    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "domain",
                "policy",
                "n_episodes",
                "drop_rate",
                "survival_steps_mean",
                "rmse_mean",
                "rmse_median",
            ]
        )
        for d_name, by_pol in aggregates.items():
            for pol, agg in by_pol.items():
                w.writerow(
                    [
                        d_name,
                        pol,
                        int(agg["n_episodes"]),
                        f"{agg['drop_rate']:.4f}",
                        f"{agg['survival_steps_mean']:.1f}",
                        f"{agg['rmse_mean']:.6f}",
                        f"{agg['rmse_median']:.6f}",
                    ]
                )

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()

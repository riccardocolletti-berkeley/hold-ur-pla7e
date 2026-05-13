"""Matplotlib outputs: per-metric bar charts and ball-trajectory overlays.

Plots are written as PNG to the benchmark's output directory. We keep the
two policies (``pid``, ``residual``) and the colour palette as small
constants so the legend / theming stays consistent across the figures
the report will reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sim.benchmark.rollout import EpisodeTrace

POLICY_LABELS = {"pid": "PID only", "residual": "PID + RL"}
POLICY_COLORS = {"pid": "#3aa1d6", "residual": "#FDB515"}

#: Two-line tick label per domain: name on top, "what changed" on bottom.
#: Keep both halves short so axis labels stay readable at default size.
DOMAIN_LABELS: dict[str, str] = {
    "standard": "Standard\n(no DR)",
    "slippery": "Slippery\n(μ=0.20)",
    "sticky": "Sticky\n(μ=0.90)",
    "heavy_ball": "Heavy ball\n(5 g)",
    "low_gravity": "Low gravity\n(g/2)",
    "high_gravity": "High gravity\n(1.5×g)",
    "noisy_vision": "Noisy vision\n(5 mm σ)",
    "high_latency": "High latency\n(200 ms)",
    "full_dr": "Full DR\n(training)",
    "extreme_dr": "Extreme DR\n(OOD)",
}

#: One-line caption used as a subtitle on the trajectory overlays.
DOMAIN_CAPTIONS: dict[str, str] = {
    "standard": "no randomization: sanity baseline",
    "slippery": "low-friction surface, ball slides freely",
    "sticky": "high-friction surface, ball grips and resists rolling",
    "heavy_ball": "1.85× ITTF mass: more inertia, slower to react",
    "low_gravity": "half Earth gravity, rolling acceleration halved",
    "high_gravity": "1.5× Earth gravity, rolling accelerates faster",
    "noisy_vision": "5 mm Gaussian noise on position, beyond training (3 mm)",
    "high_latency": "200 ms action delay, beyond training (≤150 ms)",
    "full_dr": "full DR + realistic actuator (matches training)",
    "extreme_dr": "out-of-distribution worst-case: max delay + slippery",
}


def plot_metric_bars(
    aggregates: dict[str, dict[str, dict[str, float]]],
    metric: str,
    out_path: Path,
    title: str,
) -> None:
    """Bar chart with one (PID, residual) pair per domain for a given metric.

    ``aggregates`` is the nested dict produced by the runner:
    ``{domain_name: {policy_name: {metric: value}}}``.
    """
    domains = list(aggregates.keys())
    pid_values = [aggregates[d]["pid"][metric] for d in domains]
    rl_values = [aggregates[d]["residual"][metric] for d in domains]

    x = np.arange(len(domains))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - width / 2, pid_values, width, label=POLICY_LABELS["pid"], color=POLICY_COLORS["pid"])
    ax.bar(
        x + width / 2,
        rl_values,
        width,
        label=POLICY_LABELS["residual"],
        color=POLICY_COLORS["residual"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABELS.get(d, d) for d in domains], fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trajectory_overlay(
    traces: dict[str, EpisodeTrace],
    out_path: Path,
    domain_name: str,
) -> None:
    """One example trajectory per policy, overlaid on the same target.

    Both policies see the same target (we picked the same seed inside the
    runner), so the dashed line is identical for both: what differs is the
    ball trace, which makes the comparison visually unambiguous.
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    any_trace = next(iter(traces.values()))
    ax.plot(
        any_trace.target_xy[:, 0],
        any_trace.target_xy[:, 1],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="target",
    )

    for pol, trace in traces.items():
        ax.plot(
            trace.ball_xy[:, 0],
            trace.ball_xy[:, 1],
            color=POLICY_COLORS[pol],
            linewidth=1.5,
            label=POLICY_LABELS[pol],
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    caption = DOMAIN_CAPTIONS.get(domain_name, "")
    ax.set_title(f"Ball trajectory: {domain_name}\n{caption}", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

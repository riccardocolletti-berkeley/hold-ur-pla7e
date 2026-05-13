"""Pure-numerics summary of episode traces: no env, no plotting.

Every function takes either a single :class:`EpisodeTrace` or a sequence of
them, so the same primitives can be reused across domains, policies, and
filtered subsets (e.g. only the dropped episodes). Aggregates return plain
floats / dicts so the downstream CSV writer can dump them directly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from sim.benchmark.rollout import EpisodeTrace


def rmse(trace: EpisodeTrace) -> float:
    """Root-mean-square ball↔target distance over the episode [m]."""
    if trace.n_steps == 0:
        return float("nan")
    err = np.linalg.norm(trace.ball_xy - trace.target_xy, axis=1)
    return float(np.sqrt(np.mean(err**2)))


def drop_rate(traces: Sequence[EpisodeTrace]) -> float:
    """Fraction of episodes that ended with a drop."""
    if not traces:
        return float("nan")
    return sum(t.dropped for t in traces) / len(traces)


def survival_steps_mean(traces: Sequence[EpisodeTrace]) -> float:
    """Mean episode length in steps (high = controller stays alive)."""
    if not traces:
        return float("nan")
    return float(np.mean([t.n_steps for t in traces]))


def aggregate(traces: Sequence[EpisodeTrace]) -> dict[str, float]:
    """Bundle the per-trace metrics into a single flat dict for CSV/UI."""
    rmses = [rmse(t) for t in traces if t.n_steps > 0]
    return {
        "n_episodes": float(len(traces)),
        "drop_rate": drop_rate(traces),
        "survival_steps_mean": survival_steps_mean(traces),
        "rmse_mean": float(np.mean(rmses)) if rmses else float("nan"),
        "rmse_median": float(np.median(rmses)) if rmses else float("nan"),
    }

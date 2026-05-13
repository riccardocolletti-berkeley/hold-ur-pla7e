"""One-episode rollout: take an env + policy, emit ball/target traces.

Both the PID-only and the PID+RL evaluations go through the same function;
the only difference is the ``policy_fn`` passed in, which keeps the
comparison apples-to-apples (identical env construction, identical seed,
identical DR sample). The trace is plain numpy so downstream metrics and
plots can stay free of env objects.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: A policy is a function that maps a 12-element observation to a 2-element
#: action vector in ``[-1, 1]`` (the env multiplies by ``action_scale``
#: internally, mirroring training). Use ``lambda obs: np.zeros(2)`` for the
#: PID-only baseline.
PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EpisodeTrace:
    """Per-step ball and target positions captured during one rollout."""

    ball_xy: np.ndarray  # shape (n_steps, 2), [m]
    target_xy: np.ndarray  # shape (n_steps, 2), [m]
    dropped: bool  # True if the env terminated (ball fell off)
    n_steps: int  # number of recorded steps (≤ max_steps)


def run_episode(env, policy_fn: PolicyFn, max_steps: int) -> EpisodeTrace:
    """Run one episode, capturing ball and target positions each step.

    Returns an :class:`EpisodeTrace` truncated at the actual termination
    step. The env's reset is implicit on the first call; callers that want
    to reuse a fresh env per seed should construct a new env outside.
    """
    obs, _ = env.reset()
    ball = np.zeros((max_steps, 2), dtype=np.float32)
    target = np.zeros((max_steps, 2), dtype=np.float32)
    n = 0
    dropped = False
    for i in range(max_steps):
        action = policy_fn(obs)
        obs, _, term, trunc, info = env.step(action)
        ball[i] = (obs[0], obs[1])
        target[i] = info["target"]
        n = i + 1
        if term:
            dropped = True
            break
        if trunc:
            break
    return EpisodeTrace(
        ball_xy=ball[:n].copy(),
        target_xy=target[:n].copy(),
        dropped=dropped,
        n_steps=n,
    )

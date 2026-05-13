"""Filesystem helpers for picking and loading saved RL artefacts.

Kept separate from ``controller.py`` so path resolution and stats loading
stay unit-testable without pulling Stable-Baselines 3.

Expected run directory layout::

    run_dir/
      best_model.zip        (or ppo_final.zip; preferred in that order)
      obs_rms.npz           mean / var / epsilon / clip_obs from VecNormalize
      env_meta.json         action_scale, lookahead_s, policy_hz
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

_SNAPSHOT_STEPS = re.compile(r"ppo_(\d+)_steps\.zip$")


def pick_checkpoint(run_dir: Path) -> Path | None:
    """Best checkpoint in ``run_dir``, or ``None``.

    Preference: (1) ``best_model.zip``, (2) ``ppo_final.zip``,
    (3) the newest ``ppo_*_steps.zip`` periodic snapshot.
    """
    if not run_dir.is_dir():
        return None
    for name in ("best_model.zip", "ppo_final.zip"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate

    # Sort by parsed step count so the choice is stable across copies and
    # filesystems with coarse mtime resolution; unparsable names fall first.
    def _steps(path: Path) -> int:
        m = _SNAPSHOT_STEPS.search(path.name)
        return int(m.group(1)) if m else -1

    snapshots = sorted(run_dir.glob("ppo_*_steps.zip"), key=_steps)
    return snapshots[-1] if snapshots else None


def obs_rms_path_for_checkpoint(checkpoint_path: Path) -> Path:
    """Path of the VecNormalize stats matching ``checkpoint_path``.

    Falls back to ``obs_rms.npz`` if the per-checkpoint file isn't present.
    """
    run_dir = checkpoint_path.parent
    if checkpoint_path.name == "best_model.zip":
        best = run_dir / "best_obs_rms.npz"
        return best if best.exists() else run_dir / "obs_rms.npz"

    match = _SNAPSHOT_STEPS.search(checkpoint_path.name)
    if match:
        step_stats = run_dir / f"obs_rms_{match.group(1)}_steps.npz"
        return step_stats if step_stats.exists() else run_dir / "obs_rms.npz"

    return run_dir / "obs_rms.npz"


def load_meta(path: Path) -> dict[str, Any]:
    """Load ``env_meta.json``; return ``{}`` if missing."""
    if not path.is_file():
        return {}
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def load_obs_rms(
    path: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Load VecNormalize stats as ``(mean, std, clip_obs)``.

    Missing file -> ``mean`` and ``std`` are ``None`` and ``clip_obs``
    falls back to a conservative ``10.0``.
    """
    if not path.is_file():
        return None, None, 10.0
    data = np.load(path)
    mean = data["mean"].astype(np.float32)
    std = np.sqrt(data["var"].astype(np.float32) + float(data["epsilon"]))
    clip = float(data["clip_obs"])
    return mean, std, clip

"""Tests for the RL loader helpers (``pick_checkpoint``, ``obs_rms_path_for_checkpoint``,
``load_meta``, ``load_obs_rms``) without pulling Stable-Baselines 3.
"""

import numpy as np

from ballplate.controllers.rl.loader import (
    load_meta,
    load_obs_rms,
    obs_rms_path_for_checkpoint,
    pick_checkpoint,
)

# ------------------------------------------------------------ pick_checkpoint --


def test_pick_checkpoint_returns_none_when_dir_missing(tmp_path):
    assert pick_checkpoint(tmp_path / "nope") is None


def test_pick_checkpoint_prefers_best_over_final(tmp_path):
    (tmp_path / "best_model.zip").touch()
    (tmp_path / "ppo_final.zip").touch()
    (tmp_path / "ppo_100_steps.zip").touch()
    assert pick_checkpoint(tmp_path).name == "best_model.zip"


def test_pick_checkpoint_falls_back_to_final(tmp_path):
    (tmp_path / "ppo_final.zip").touch()
    (tmp_path / "ppo_100_steps.zip").touch()
    assert pick_checkpoint(tmp_path).name == "ppo_final.zip"


def test_pick_checkpoint_falls_back_to_highest_step_snapshot(tmp_path):
    # ``pick_checkpoint`` sorts by step count parsed from the filename,
    # so the choice is stable across copies and filesystems with coarse
    # mtime resolution.
    (tmp_path / "ppo_100_steps.zip").touch()
    newer = tmp_path / "ppo_500_steps.zip"
    newer.touch()
    assert pick_checkpoint(tmp_path) == newer


def test_pick_checkpoint_returns_none_when_empty(tmp_path):
    assert pick_checkpoint(tmp_path) is None


# -------------------------------------------------- obs_rms_path_for_checkpoint --


def test_obs_rms_path_prefers_best_stats_for_best_model(tmp_path):
    ckpt = tmp_path / "best_model.zip"
    stats = tmp_path / "best_obs_rms.npz"
    ckpt.touch()
    stats.touch()
    assert obs_rms_path_for_checkpoint(ckpt) == stats


def test_obs_rms_path_matches_step_snapshot(tmp_path):
    ckpt = tmp_path / "ppo_500_steps.zip"
    stats = tmp_path / "obs_rms_500_steps.npz"
    ckpt.touch()
    stats.touch()
    assert obs_rms_path_for_checkpoint(ckpt) == stats


def test_obs_rms_path_falls_back_to_final_stats(tmp_path):
    assert obs_rms_path_for_checkpoint(tmp_path / "best_model.zip") == tmp_path / "obs_rms.npz"


# ----------------------------------------------------------------- load_meta --


def test_load_meta_missing_file_returns_empty(tmp_path):
    assert load_meta(tmp_path / "missing.json") == {}


def test_load_meta_parses_json(tmp_path):
    p = tmp_path / "env_meta.json"
    p.write_text('{"action_scale": 2.0, "policy_hz": 50}')
    meta = load_meta(p)
    assert meta["action_scale"] == 2.0
    assert meta["policy_hz"] == 50


# -------------------------------------------------------------- load_obs_rms --


def test_load_obs_rms_missing_file_returns_none_triplet(tmp_path):
    mean, std, clip = load_obs_rms(tmp_path / "missing.npz")
    assert mean is None and std is None
    assert clip == 10.0  # conservative default


def test_load_obs_rms_round_trip(tmp_path):
    p = tmp_path / "obs_rms.npz"
    np.savez(
        p,
        mean=np.zeros(12, dtype=np.float32),
        var=np.ones(12, dtype=np.float32),
        epsilon=np.float64(1.0e-8),
        clip_obs=np.float64(8.0),
    )
    mean, std, clip = load_obs_rms(p)
    assert mean.shape == (12,)
    assert np.allclose(std, 1.0)
    assert clip == 8.0

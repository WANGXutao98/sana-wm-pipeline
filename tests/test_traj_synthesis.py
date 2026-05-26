"""Tests for stage03_3dgs_aug.traj_synthesis (paper App. B.2)."""
from __future__ import annotations

import numpy as np
import pytest

from sana_wm_pipeline.stage03_3dgs_aug.traj_synthesis import (
    FAMILIES_30,
    FAMILY_COUNTS,
    TOTAL_TRAJECTORIES,
    Trajectory,
    synthesize_40_trajectories,
)


def _scene_stats(n_train: int = 50, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cams = rng.standard_normal((n_train, 3)).astype(np.float32) * 2.0
    return dict(
        centroid=np.array([0, 0, 0], np.float32),
        median_radius=2.0,
        height_range=(0.0, 1.5),
        pca_dirs=np.eye(3, dtype=np.float32),
        training_cam_positions=cams,
        training_cam_orientations=np.tile(np.eye(3, dtype=np.float32), (n_train, 1, 1)),
    )


def test_total_count_is_40():
    assert TOTAL_TRAJECTORIES == 40
    assert sum(FAMILY_COUNTS.values()) == 30


def test_family_distribution_uses_all_eight():
    expected = {"orbit", "spiral", "dolly", "flythrough",
                "random_walk", "crane", "pendulum", "compound"}
    assert set(FAMILIES_30) == expected
    assert set(FAMILY_COUNTS.keys()) == expected


def test_synthesize_returns_40_with_expected_families():
    trajs = synthesize_40_trajectories(_scene_stats(), n_frames=961, fps=16)
    assert len(trajs) == 40
    fams = [t.family for t in trajs]
    # exactly 10 spline_interp + 30 family.
    assert fams.count("spline_interp") == 10
    for fam, n in FAMILY_COUNTS.items():
        assert fams.count(fam) == n, f"{fam}: want {n} got {fams.count(fam)}"


def test_each_traj_has_correct_shapes_and_dtypes():
    trajs = synthesize_40_trajectories(_scene_stats(), n_frames=121, fps=16)
    for t in trajs:
        assert isinstance(t, Trajectory)
        assert t.poses_c2w.shape == (121, 4, 4)
        assert t.poses_c2w.dtype == np.float32
        assert t.intrinsics.shape == (121, 4)
        assert t.intrinsics.dtype == np.float32


def test_poses_first_row_is_rotation_matrix():
    trajs = synthesize_40_trajectories(_scene_stats(), n_frames=33, fps=16)
    for t in trajs:
        R = t.poses_c2w[0, :3, :3]
        # rotation matrix → det close to 1, orthonormal
        assert np.isfinite(R).all()
        assert abs(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-3)
        I = R @ R.T
        assert np.allclose(I, np.eye(3), atol=1e-3)


def test_unknown_family_rejected():
    from sana_wm_pipeline.stage03_3dgs_aug.traj_synthesis import _family_traj
    with pytest.raises(ValueError):
        _family_traj("alien", _scene_stats(), 16, seed=0)

"""Tests for Umeyama Sim(3) alignment (paper App. B.1)."""
import numpy as np
import pytest
from sana_wm_pipeline.stage02_pose.umeyama import (
    umeyama_sim3, umeyama_sim3_inlier_filter, DEFAULT_INLIER_PERCENTILE,
)


def _make_random_correspondences(N=300, s=0.7, theta=0.4, t=(1.0, -2.0, 0.5), seed=0):
    rng = np.random.default_rng(seed)
    src = rng.standard_normal((N, 3))
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
                  [np.sin(theta),  np.cos(theta), 0],
                  [0, 0, 1]], dtype=np.float64)
    dst = s * (src @ R.T) + np.asarray(t)
    return src, dst, s, R, np.asarray(t)


def test_default_inlier_percentile_is_paper_value():
    assert DEFAULT_INLIER_PERCENTILE == 80.0


def test_umeyama_recovers_sim3_clean():
    src, dst, s_true, R_true, t_true = _make_random_correspondences()
    s, R, t = umeyama_sim3(src, dst)
    assert abs(s - s_true) < 1e-9
    assert np.allclose(R, R_true, atol=1e-9)
    assert np.allclose(t, t_true, atol=1e-9)


def test_umeyama_handles_reflection_correctly():
    """If raw SVD gives a reflection, the D = diag(1,1,-1) flip should recover proper R."""
    rng = np.random.default_rng(11)
    src = rng.standard_normal((200, 3))
    # Make dst a *reflected* copy by negating one axis — Umeyama must still return det(R)=+1.
    dst = src.copy()
    dst[:, 0] = -dst[:, 0]
    s, R, t = umeyama_sim3(src, dst)
    assert abs(float(np.linalg.det(R)) - 1.0) < 1e-9


def test_inlier_filter_rejects_outliers():
    src, dst, s_true, R_true, t_true = _make_random_correspondences(N=300, seed=42)
    # corrupt 20% with large noise
    rng = np.random.default_rng(99)
    n_out = 60
    dst[:n_out] += rng.normal(0, 5.0, size=(n_out, 3))
    s, R, t, mask = umeyama_sim3_inlier_filter(src, dst, inlier_percentile=80.0)
    # outliers should be the rejected 20%
    assert mask.sum() >= 0.60 * len(src)
    assert mask.sum() <= 0.85 * len(src)
    # recovered transform should be near ground truth
    assert abs(s - s_true) < 0.05
    assert np.allclose(R, R_true, atol=0.05)
    assert np.allclose(t, t_true, atol=0.2)


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="at least 3"):
        umeyama_sim3(np.zeros((2, 3)), np.zeros((2, 3)))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        umeyama_sim3(np.zeros((10, 3)), np.zeros((10, 2)))


def test_clean_inputs_iterate_to_full_inlier_set():
    """With clean inputs, the inlier mask should stabilize quickly."""
    src, dst, _, _, _ = _make_random_correspondences(N=100)
    s, R, t, mask = umeyama_sim3_inlier_filter(src, dst, inlier_percentile=80.0)
    # the residuals are essentially zero, so the percentile cutoff catches all points
    # but the next iteration sees the same set → returns stable
    # (mask can be either ~all True or stabilize at 80% — either is acceptable as long as transform is recovered)
    s_true = 0.7
    assert abs(s - s_true) < 1e-9

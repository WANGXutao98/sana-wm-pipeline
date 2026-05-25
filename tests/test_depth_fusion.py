"""Tests for Pi3X+MoGe-2 depth fusion (paper App. B.1)."""
import numpy as np
import pytest
from sana_wm_pipeline.stage02_pose.depth_fusion import (
    fuse_metric_scale, per_frame_scale_ls, DEFAULT_EMA_MOMENTUM, MIN_DEPTH,
)


def test_default_ema_momentum_is_paper_value():
    """Paper App. B.1 explicitly states momentum 0.99."""
    assert DEFAULT_EMA_MOMENTUM == 0.99


def test_single_frame_recovers_known_scale_no_noise():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.5, 10.0, size=200)
    s_true = 0.37
    b = s_true * a
    assert abs(per_frame_scale_ls(a, b) - s_true) < 1e-9


def test_single_frame_recovers_known_scale_with_noise():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.5, 10.0, size=2000)
    s_true = 1.83
    b = s_true * a + rng.normal(0, 0.01, size=a.shape)
    assert abs(per_frame_scale_ls(a, b) - s_true) < 5e-3


def test_multi_frame_no_ema_uses_independent_estimates():
    """When momentum=0, each frame's scale should match the noiseless LS estimate."""
    rng = np.random.default_rng(2)
    T, N = 8, 500
    a = rng.uniform(0.5, 10.0, size=(T, N))
    true_scales = np.linspace(0.4, 2.0, T)
    b = true_scales[:, None] * a
    s = fuse_metric_scale(a, b, momentum=0.0)
    assert np.allclose(s, true_scales, atol=1e-9)


def test_multi_frame_ema_smooths_jumps():
    """A sudden jump in scale should be heavily damped by momentum=0.99."""
    T, N = 5, 1000
    rng = np.random.default_rng(3)
    a = rng.uniform(0.5, 10.0, size=(T, N))
    true_scales = np.array([1.0, 1.0, 5.0, 5.0, 5.0])    # jump at t=2
    b = true_scales[:, None] * a
    s = fuse_metric_scale(a, b, momentum=0.99)
    # After 1 step at 0.99 momentum, scale moves only 1% toward target:
    # s[2] = 0.99*1.0 + 0.01*5.0 = 1.04
    assert abs(s[2] - 1.04) < 1e-6
    # After 3 steps still nowhere near 5.0 (about 1.1188)
    assert s[4] < 1.2


def test_inverse_depth_weight_handles_zero_safely():
    """Points with depth at MIN_DEPTH floor should still produce a finite scale."""
    a = np.array([MIN_DEPTH, 1.0, 2.0, 5.0])
    s_true = 0.5
    b = s_true * a
    s = per_frame_scale_ls(a, b)
    assert np.isfinite(s) and abs(s - s_true) < 1e-9


def test_valid_mask_excludes_outliers():
    a = np.array([1.0, 1.0, 1.0, 1.0])
    b = np.array([0.5, 0.5, 99.0, 0.5])              # one outlier
    mask = np.array([True, True, False, True])
    s = per_frame_scale_ls(a, b, valid_mask=mask)
    assert abs(s - 0.5) < 1e-9


def test_shape_mismatch_raises():
    a = np.zeros((3, 100))
    b = np.zeros((3, 50))
    with pytest.raises(AssertionError):
        fuse_metric_scale(a, b)


def test_t_equals_1_works():
    a = np.array([[1.0, 2.0, 3.0]])
    b = np.array([[0.5, 1.0, 1.5]])
    s = fuse_metric_scale(a, b)
    assert s.shape == (1,) and abs(s[0] - 0.5) < 1e-9


def test_ema_first_frame_no_smoothing():
    """Frame 0 has no s_prev, so its output should equal the raw LS estimate."""
    a = np.array([[1.0, 1.0, 1.0]])
    b = np.array([[0.7, 0.7, 0.7]])
    s = fuse_metric_scale(a, b, momentum=0.99)
    assert abs(s[0] - 0.7) < 1e-9

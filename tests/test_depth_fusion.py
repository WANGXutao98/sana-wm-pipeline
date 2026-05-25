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
    """Points slightly above MIN_DEPTH floor should still produce a finite scale.

    Note: under the validity-mask contract (a > MIN_DEPTH), points exactly at
    the floor are excluded. We use MIN_DEPTH * 2 here to test the boundary case
    where the smallest valid depth is just above the floor.
    """
    a = np.array([MIN_DEPTH * 2, 1.0, 2.0, 5.0])
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


def test_negative_pi3x_excluded_from_estimate():
    """Negative or sub-MIN_DEPTH Pi3X values must not contaminate the LS result."""
    a = np.array([-0.5, 0.0, 1e-5, 1.0, 2.0, 3.0])    # first 3 invalid
    b = 0.5 * np.abs(a) + 1e-9                         # avoid exact zero in b
    s = per_frame_scale_ls(a, b)
    # Should equal LS on the last 3 points which all give s=0.5
    assert abs(s - 0.5) < 1e-6


def test_all_masked_returns_nan_for_single_frame():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.5, 1.0, 1.5])
    mask = np.zeros(3, dtype=bool)
    s = per_frame_scale_ls(a, b, valid_mask=mask)
    assert np.isnan(s)


def test_degenerate_frame_carries_forward_prev_scale():
    """A NaN frame in the middle should carry forward the previous scale,
    not reset it to 0 and pollute subsequent EMA states."""
    T, N = 4, 100
    rng = np.random.default_rng(7)
    a = rng.uniform(0.5, 5.0, size=(T, N))
    b = 0.8 * a
    mask = np.ones((T, N), dtype=bool)
    mask[2, :] = False                  # frame 2 fully masked out
    s = fuse_metric_scale(a, b, valid_mask=mask, momentum=0.99)
    assert np.isfinite(s).all()
    assert abs(s[3] - s[1]) < 0.05      # frame 3 should NOT be dragged toward 0


def test_first_frame_degenerate_returns_nan():
    a = np.array([[1.0, 2.0]])
    b = np.array([[0.5, 1.0]])
    mask = np.zeros((1, 2), dtype=bool)
    s = fuse_metric_scale(a, b, valid_mask=mask, momentum=0.99)
    assert s.shape == (1,) and np.isnan(s[0])

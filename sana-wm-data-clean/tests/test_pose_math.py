"""Tests for the SANA-WM pose-engine math: depth fusion + Sim(3) alignment."""

import numpy as np
import pytest

from sana_wm_data.pose.fusion import solve_frame_scale, fuse_depth_sequence
from sana_wm_data.pose.alignment import umeyama_sim3, recover_metric_scale


def test_solve_frame_scale_recovers_known_scale():
    rng = np.random.default_rng(0)
    d_pi3x = rng.uniform(0.5, 10.0, size=(4, 8))
    true_s = 2.5
    d_moge = true_s * d_pi3x  # perfect scaled match
    s = solve_frame_scale(d_pi3x[0], d_moge[0])
    assert s == pytest.approx(true_s, rel=1e-9)


def test_solve_frame_scale_weighted_least_squares():
    # closed form: s = sum(w a b) / sum(w a^2), w = 1/b (reference depth)
    a = np.array([1.0, 2.0, 4.0])
    b = np.array([2.0, 4.0, 8.0])  # exactly 2x -> s=2 regardless of weights
    assert solve_frame_scale(a, b) == pytest.approx(2.0)


def test_fuse_depth_sequence_ema_smooths():
    # per-frame raw scales jump; EMA(0.99) should lag toward the running mean
    T, HW = 5, 6
    d_pi3x = np.ones((T, HW))
    raw_scales = np.array([1.0, 3.0, 1.0, 3.0, 1.0])
    d_moge = d_pi3x * raw_scales[:, None]
    fused, scales = fuse_depth_sequence(d_pi3x, d_moge, ema_momentum=0.99)
    # first frame seeds EMA exactly
    assert scales[0] == pytest.approx(1.0)
    # EMA heavily damps the jump: frame1 ~ 0.99*1 + 0.01*3 = 1.02
    assert scales[1] == pytest.approx(0.99 * 1.0 + 0.01 * 3.0, rel=1e-6)
    # fused depth = scale * pi3x
    assert np.allclose(fused[1], scales[1] * d_pi3x[1])


def test_umeyama_sim3_recovers_transform():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(20, 3))
    s_true = 3.0
    theta = 0.7
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1.0],
    ])
    t = np.array([1.0, -2.0, 0.5])
    dst = (s_true * (R @ src.T)).T + t
    s, R_est, t_est = umeyama_sim3(src, dst)
    assert s == pytest.approx(s_true, rel=1e-6)
    assert np.allclose(R_est, R, atol=1e-6)
    assert np.allclose(t_est, t, atol=1e-6)


def test_recover_metric_scale_with_outliers():
    rng = np.random.default_rng(2)
    pred = rng.normal(size=(50, 3))
    s_true = 1.7
    gt = s_true * pred
    # corrupt 20% with gross outliers
    gt[:10] += rng.normal(scale=50, size=(10, 3))
    s = recover_metric_scale(pred, gt, inlier_percentile=80)
    assert s == pytest.approx(s_true, rel=0.05)

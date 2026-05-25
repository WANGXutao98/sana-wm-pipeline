"""Tests for paper App. B.3 pose quality filter."""
import numpy as np
import pytest
from sana_wm_pipeline.stage02_pose.pose_quality import (
    FOV_DEG_MIN, FOV_DEG_MAX, FOCAL_DIV_MAX, SCALE_CV_MAX,
    horizontal_fov_deg, vertical_fov_deg, focal_divergence,
    scale_coefficient_of_variation, evaluate_pose_quality,
)


# ---- Paper-constant locks ----
def test_paper_constants():
    assert FOV_DEG_MIN == 25.0
    assert FOV_DEG_MAX == 120.0
    assert FOCAL_DIV_MAX == 0.20
    assert SCALE_CV_MAX == 2.0


# ---- Per-helper math ----
def test_horizontal_fov_known_case():
    # 1280-wide image, fx=640 → θ_x = 2*atan(1) = 90°
    assert abs(float(horizontal_fov_deg(np.array([640.0]), 1280)[0]) - 90.0) < 1e-9


def test_vertical_fov_known_case():
    # 720-tall image, fy=360 → θ_y = 2*atan(1) = 90°
    assert abs(float(vertical_fov_deg(np.array([360.0]), 720)[0]) - 90.0) < 1e-9


def test_focal_divergence_zero_for_square_pixels():
    fx = np.full(5, 700.0); fy = np.full(5, 700.0)
    assert float(focal_divergence(fx, fy).max()) < 1e-9


def test_focal_divergence_matches_paper_formula():
    # |500-1500| / ((500+1500)/2) = 1000 / 1000 = 1.0
    fx = np.array([500.0]); fy = np.array([1500.0])
    assert abs(float(focal_divergence(fx, fy)[0]) - 1.0) < 1e-9


def test_scale_cv_zero_for_constant_series():
    assert scale_coefficient_of_variation(np.full(961, 0.7)) < 1e-9


def test_scale_cv_ignores_nan():
    s = np.array([1.0, 1.0, 1.0, float("nan"), 1.0])
    assert scale_coefficient_of_variation(s) < 1e-9


# ---- Aggregator (paper App. B.3 boundary cases) ----
def test_typical_720p_clip_passes():
    T = 961
    intr = np.tile([[[700, 705, 640, 360]]], (T, 1, 1)).astype(np.float32)
    rng = np.random.default_rng(0)
    scale = np.ones(T) + rng.normal(0, 0.05, T)
    result = evaluate_pose_quality(intr, (1280, 720), scale)
    assert result.passed, result.reasons


def test_too_wide_fov_fails():
    # fx=100 with W=1280 → fov_x ≈ 2*atan(6.4) ≈ 161° > 120°
    T = 5
    intr = np.tile([[[100, 100, 640, 360]]], (T, 1, 1)).astype(np.float32)
    result = evaluate_pose_quality(intr, (1280, 720), np.ones(T))
    assert not result.passed
    assert any("fov_x" in r for r in result.reasons)


def test_focal_divergence_violation_fails():
    T = 3
    intr = np.tile([[[500.0, 1500.0, 640, 360]]], (T, 1, 1)).astype(np.float32)
    # focal_divergence = 1.0 ≫ 0.20
    result = evaluate_pose_quality(intr, (1280, 720), np.ones(T))
    assert not result.passed
    assert any("focal_divergence" in r for r in result.reasons)


def test_scale_cv_violation_fails():
    T = 100
    intr = np.tile([[[700, 700, 640, 360]]], (T, 1, 1)).astype(np.float32)
    # Heavy-tailed (one massive outlier) → CV ≫ 2
    scale = np.full(T, 0.1)
    scale[0] = 1e4
    result = evaluate_pose_quality(intr, (1280, 720), scale)
    assert not result.passed
    assert any("scale_cv" in r for r in result.reasons)


def test_multiple_violations_aggregate_reasons():
    T = 5
    # Bad fx (too small → wide fov) AND large focal divergence
    intr = np.tile([[[100.0, 1000.0, 640, 360]]], (T, 1, 1)).astype(np.float32)
    result = evaluate_pose_quality(intr, (1280, 720), np.ones(T))
    assert not result.passed
    assert len(result.reasons) >= 2


def test_wrong_intrinsics_shape_raises():
    with pytest.raises(ValueError, match="N, V, 4"):
        evaluate_pose_quality(np.zeros((5, 1, 3), dtype=np.float32),
                              (1280, 720), np.ones(5))

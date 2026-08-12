"""Tests for camera-specific filter math (SANA-WM Appendix B.3)."""

import math

import numpy as np
import pytest

from sana_wm_data.filter.camera import (
    fov_degrees,
    focal_divergence,
    scale_cov,
    camera_filter_pass,
)


def test_fov_degrees_known_value():
    # fx such that W/(2 fx) = tan(30deg) => FoV_x = 60deg
    W, H = 1920, 1080
    fx = W / (2 * math.tan(math.radians(30)))
    fy = H / (2 * math.tan(math.radians(20)))  # FoV_y = 40deg
    fov_x, fov_y = fov_degrees(W, H, fx, fy)
    assert fov_x == pytest.approx(60.0, abs=1e-6)
    assert fov_y == pytest.approx(40.0, abs=1e-6)


def test_focal_divergence_symmetric_normalized():
    # |fx-fy| / ((fx+fy)/2)
    assert focal_divergence(100, 100) == pytest.approx(0.0)
    assert focal_divergence(110, 90) == pytest.approx(20 / 100)  # 0.2


def test_scale_cov():
    s = [1.0, 1.0, 1.0]
    assert scale_cov(s) == pytest.approx(0.0)
    s2 = np.array([1.0, 2.0, 3.0])
    expected = np.std(s2) / (np.mean(s2) + 1e-8)
    assert scale_cov(s2) == pytest.approx(expected, rel=1e-6)


def test_camera_filter_pass_accepts_good_clip():
    W, H = 1280, 720
    fx = W / (2 * math.tan(math.radians(30)))  # ~60 deg
    fy = fx  # square pixels -> 0 divergence, FoV_y < FoV_x but still > 25
    fov_y = math.degrees(2 * math.atan(H / (2 * fy)))
    assert fov_y > 25  # sanity
    ok, reasons = camera_filter_pass(
        W, H, fx, fy, scale_factors=[1.0, 1.01, 0.99],
        cfg={"fov_deg": [25, 120], "focal_div_max": 0.20, "scale_cov_max": 2.0},
    )
    assert ok, reasons
    assert reasons == []


def test_camera_filter_rejects_narrow_fov():
    W, H = 1280, 720
    fx = fy = W / (2 * math.tan(math.radians(10)))  # FoV_x ~ 20deg < 25
    ok, reasons = camera_filter_pass(
        W, H, fx, fy, scale_factors=[1.0],
        cfg={"fov_deg": [25, 120], "focal_div_max": 0.20, "scale_cov_max": 2.0},
    )
    assert not ok
    assert any("fov" in r for r in reasons)


def test_camera_filter_rejects_focal_divergence_and_scale_cov():
    W, H = 1000, 1000
    fx, fy = 130.0, 70.0  # divergence = 60/100 = 0.6 > 0.2; FoV also wide
    scales = [1.0, 5.0, 0.2, 4.0]  # high coefficient of variation > 2.0? check
    ok, reasons = camera_filter_pass(
        W, H, fx, fy, scale_factors=scales,
        cfg={"fov_deg": [25, 120], "focal_div_max": 0.20, "scale_cov_max": 2.0},
    )
    assert not ok
    assert any("focal_div" in r for r in reasons)

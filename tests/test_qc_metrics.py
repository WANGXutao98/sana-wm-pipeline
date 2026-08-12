from __future__ import annotations
import numpy as np
import pytest
from sana_wm_pipeline.qc.metrics import (
    check_so3, check_first_frame, check_trajectory,
    check_no_nan_inf, check_caption,
    check_color_saturation, check_caption_camera_words,
    compute_stage1_metrics,
)


def _poses(T: int, step: float = 0.1) -> np.ndarray:
    p = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    p[:, 0, 3] = np.arange(T, dtype=np.float32) * step
    return p


def _intr(T: int, fx: float = 700.0) -> np.ndarray:
    intr = np.zeros((T, 1, 4), dtype=np.float32)
    intr[:, 0, :] = [fx, fx, 640, 360]
    return intr


# --- check_so3 ---
def test_so3_identity():
    det_mean, det_std, orth_err = check_so3(np.tile(np.eye(4, dtype=np.float32), (5, 1, 1)))
    assert abs(det_mean - 1.0) < 1e-6 and orth_err < 1e-6

def test_so3_zero_rotation():
    p = np.tile(np.eye(4, dtype=np.float32), (5, 1, 1))
    p[:, :3, :3] = 0.0
    det_mean, _, _ = check_so3(p)
    assert abs(det_mean) < 1e-6

# --- check_first_frame ---
def test_first_frame_identity():
    ok, dev = check_first_frame(np.tile(np.eye(4, dtype=np.float32), (5, 1, 1)))
    assert ok and dev < 1e-6

def test_first_frame_shifted():
    p = np.tile(np.eye(4, dtype=np.float32), (5, 1, 1))
    p[0, 0, 3] = 1.0
    ok, dev = check_first_frame(p)
    assert not ok

# --- check_trajectory ---
def test_trajectory_linear():
    total, mean, mx, n_jumps = check_trajectory(_poses(100, 0.1), 0.5)
    assert abs(total - 9.9) < 1e-3 and n_jumps == 0

def test_trajectory_counts_jumps():
    # Frame 5 at x=5, all others at x=0.
    # Creates 2 large steps: step 4→5 (distance 5) and step 5→6 (distance 5).
    p = _poses(10, 0.0)
    p[5, 0, 3] = 5.0
    _, _, _, n = check_trajectory(p, 0.5)
    assert n == 2

# --- check_no_nan_inf ---
def test_no_nan_inf_clean():
    ok, reasons = check_no_nan_inf({"a": np.ones((5, 4, 4))})
    assert ok and reasons == []

def test_no_nan_inf_detects_nan():
    a = np.ones((5, 4, 4))
    a[0, 0, 0] = float("nan")
    ok, reasons = check_no_nan_inf({"poses": a})
    assert not ok and any("poses" in r for r in reasons)

# --- check_caption ---
def test_caption_ok():
    ok, length = check_caption("A" * 60)
    assert ok and length >= 50

def test_caption_short():
    ok, _ = check_caption("hi")
    assert not ok

def test_caption_placeholder():
    for s in ["n/a", "N/A", "", "none"]:
        assert not check_caption(s, 50)[0]

# --- check_color_saturation ---
def test_saturation_gray_image():
    gray = np.full((10, 64, 64, 3), 128, dtype=np.uint8)
    s = check_color_saturation(gray)
    assert 0.0 <= s <= 10.0  # gray → near-zero saturation

def test_saturation_red_image():
    red = np.zeros((10, 64, 64, 3), dtype=np.uint8)
    red[..., 0] = 255  # pure red
    s = check_color_saturation(red)
    assert s > 100  # saturated red → high S in HSV

# --- check_caption_camera_words ---
def test_camera_words_none():
    assert check_caption_camera_words("A busy city street at night.") == []

def test_camera_words_detected():
    hits = check_caption_camera_words("The camera pans left across the scene.")
    assert any("pan" in h for h in hits)

def test_camera_words_weak_allowed():
    # "camera stays behind" is a weak framework word, not a strong action word
    hits = check_caption_camera_words("The camera stays behind the character.")
    assert hits == []

def test_camera_words_zoom():
    hits = check_caption_camera_words("zooms in on the building.")
    assert any("zoom" in h for h in hits)

# --- compute_stage1_metrics ---
def test_stage1_metrics_pass():
    T = 100
    result = compute_stage1_metrics(
        poses=_poses(T), intrinsics=_intr(T), scale=np.ones(T, np.float32),
        caption="A " * 60, meta_T=T, image_wh=(1280, 720),
        jump_threshold_m=0.5, min_caption_len=50,
    )
    assert result["t_aligned"] and result["so3_valid"] and result["first_frame_ok"]
    assert result["caption_ok"] and result["n_jumps"] == 0
    assert result["pose_quality_ok"] is True

def test_stage1_metrics_t_mismatch():
    T = 10
    result = compute_stage1_metrics(
        poses=_poses(T), intrinsics=_intr(T), scale=np.ones(T, np.float32),
        caption="A " * 60, meta_T=T + 1, image_wh=(1280, 720),
        jump_threshold_m=0.5, min_caption_len=50,
    )
    assert not result["t_aligned"] and "t_mismatch" in " ".join(result["reasons"])

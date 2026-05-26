"""Tests for stage04_filter.visual_metrics — paper-fixed sampling and DI."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sana_wm_pipeline.stage04_filter.visual_metrics import (
    UNIMATCH_SAMPLE_EVERY_S,
    UNIMATCH_WINDOW_S,
    DOVER_CHUNK_S,
    enumerate_unimatch_pairs,
    dover_chunk_indices,
    mean_saturation,
    frame_diff_motion_proxy,
    unimatch_flow_magnitude,
    dover_score,
    compute_all,
)


# ---- Paper-fixed sampling constants ---------------------------------------
def test_paper_fixed_constants():
    assert UNIMATCH_SAMPLE_EVERY_S == 0.5
    assert UNIMATCH_WINDOW_S == 60
    assert DOVER_CHUNK_S == 5


# ---- enumerate_unimatch_pairs ---------------------------------------------
def test_unimatch_pairs_every_half_sec_over_first_60s():
    pairs = enumerate_unimatch_pairs(n_frames=961, fps=16, window_s=60,
                                     sample_every_s=0.5)
    # window = 16*60 = 960 frames; stride = 8 ; pairs at 0..952 -> 119 pairs.
    # 60s / 0.5s = 120 nominal pairs but the LAST one ends at 960
    # which equals n_in_window so it is excluded by range(0, 960-8, 8).
    assert len(pairs) == 119
    assert pairs[0] == (0, 8)
    assert pairs[1] == (8, 16)
    assert pairs[-1] == (944, 952)


def test_unimatch_pairs_short_clip_truncates_window():
    pairs = enumerate_unimatch_pairs(n_frames=80, fps=16, window_s=60,
                                     sample_every_s=0.5)
    # n_in_window=80, stride=8 -> i in [0, 72)
    assert len(pairs) == 9
    assert pairs[-1] == (64, 72)


def test_unimatch_pairs_empty_for_degenerate_input():
    assert enumerate_unimatch_pairs(0, fps=16) == []
    assert enumerate_unimatch_pairs(961, fps=0) == []


# ---- dover_chunk_indices --------------------------------------------------
def test_dover_chunks_5s_nonoverlap_for_961_frames():
    chunks = dover_chunk_indices(n_frames=961, fps=16, chunk_s=5)
    # 961 // 80 = 12 full chunks; the trailing 1 frame is dropped.
    assert len(chunks) == 12
    assert chunks[0] == (0, 80)
    assert chunks[-1] == (880, 960)


def test_dover_chunks_skips_partial_tail():
    chunks = dover_chunk_indices(n_frames=79, fps=16, chunk_s=5)
    assert chunks == []


# ---- mean_saturation -------------------------------------------------------
def test_mean_saturation_grey_is_zero():
    frames = np.full((3, 16, 16, 3), 128, dtype=np.uint8)
    assert mean_saturation(frames) == pytest.approx(0.0)


def test_mean_saturation_pure_red_is_max():
    # Pure red in RGB -> HSV S channel = 255 for cv2's 8-bit pipeline.
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    frames[..., 0] = 255  # R
    s = mean_saturation(frames)
    assert s == pytest.approx(255.0, abs=1.0)


def test_mean_saturation_rejects_bad_shape():
    with pytest.raises(ValueError):
        mean_saturation(np.zeros((10, 10), dtype=np.uint8))


# ---- frame_diff_motion_proxy ----------------------------------------------
def test_frame_diff_motion_proxy_zero_for_static():
    frames = np.full((5, 8, 8, 3), 100, dtype=np.uint8)
    assert frame_diff_motion_proxy(frames) == pytest.approx(0.0)


def test_frame_diff_motion_proxy_positive_for_moving():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(4, 8, 8, 3), dtype=np.uint8)
    assert frame_diff_motion_proxy(frames) > 0


# ---- unimatch_flow_magnitude with injected flow_fn ------------------------
def test_unimatch_flow_magnitude_uses_paper_sampling_and_avg():
    H, W = 4, 4
    # 961 frames so we exercise the full 60 s window (119 pairs).
    frames = np.zeros((961, H, W, 3), dtype=np.uint8)
    calls = {"n": 0}

    def fake_flow(a, b):
        calls["n"] += 1
        # Flow vector of constant magnitude 3 (3-4-5 triangle: (3,4) -> |v|=5)
        return np.tile(np.array([3.0, 4.0], dtype=np.float32), (H, W, 1))

    mag = unimatch_flow_magnitude(frames, fake_flow, fps=16)
    assert calls["n"] == 119
    assert mag == pytest.approx(5.0)


def test_unimatch_flow_magnitude_nan_for_empty():
    frames = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    out = unimatch_flow_magnitude(frames, lambda a, b: np.zeros((4, 4, 2)))
    assert math.isnan(out)


# ---- dover_score with injected dover_fn -----------------------------------
def test_dover_score_averages_over_5s_chunks():
    frames = np.zeros((961, 4, 4, 3), dtype=np.uint8)
    seen = []

    def fake_dover(clip):
        seen.append(clip.shape[0])
        return 0.5 + 0.01 * len(seen)

    score = dover_score(frames, fake_dover, fps=16)
    assert seen == [80] * 12          # 12 non-overlapping 5 s chunks
    expected = np.mean([0.5 + 0.01 * (k + 1) for k in range(12)])
    assert score == pytest.approx(expected)


def test_dover_score_nan_for_short_clip():
    frames = np.zeros((50, 4, 4, 3), dtype=np.uint8)
    out = dover_score(frames, lambda c: 0.5, fps=16)
    assert math.isnan(out)


# ---- compute_all ----------------------------------------------------------
def test_compute_all_returns_nans_when_models_missing():
    frames = np.zeros((100, 4, 4, 3), dtype=np.uint8)
    out = compute_all(frames, video_path=None, flow_fn=None, dover_fn=None)
    assert set(out.keys()) == {"saturation", "vmaf_motion", "unimatch_flow", "dover"}
    # saturation always available
    assert not math.isnan(out["saturation"])
    # remaining three are NaN when their backends are not provided
    for k in ("vmaf_motion", "unimatch_flow", "dover"):
        assert math.isnan(out[k])

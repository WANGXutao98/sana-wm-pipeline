"""Tests for stage03_3dgs_aug.coverage_gate (paper App. B.2)."""
from __future__ import annotations

import numpy as np
import pytest

from sana_wm_pipeline.stage03_3dgs_aug.coverage_gate import (
    FRAME_PASS_FRACTION,
    MIN_SPLATS,
    NEAR_BLACK_FRAME_MAX,
    TILE_EMPTY_MAX,
    passes_coverage,
    passes_post_render,
)
from sana_wm_pipeline.stage03_3dgs_aug.difix3d_refine import (
    DIFIX3D_PARAMS,
    assert_difix3d_params_match_paper,
)


def test_paper_constants():
    assert MIN_SPLATS == 1000
    assert FRAME_PASS_FRACTION == 0.70
    assert TILE_EMPTY_MAX == 0.65
    assert NEAR_BLACK_FRAME_MAX == 0.30


def test_70pct_frame_pass_accepts():
    n = 100
    splats = np.zeros(n, int)
    splats[: int(0.71 * n)] = 5000
    tile_empty = np.full(n, 0.5)
    ok, reason = passes_coverage(splats, tile_empty)
    assert ok, reason


def test_69pct_frame_pass_rejects():
    n = 100
    splats = np.zeros(n, int)
    splats[: int(0.69 * n)] = 5000
    tile_empty = np.full(n, 0.5)
    ok, reason = passes_coverage(splats, tile_empty)
    assert not ok
    assert "frame_pass" in reason


def test_tile_empty_above_65pct_rejects():
    splats = np.full(10, 5000)
    tile_empty = np.full(10, 0.66)
    ok, reason = passes_coverage(splats, tile_empty)
    assert not ok
    assert "tile_empty" in reason


def test_empty_input_rejects():
    ok, reason = passes_coverage(np.array([]), np.array([]))
    assert not ok
    assert "no_sampled_frames" in reason


def test_post_render_near_black_accepts_below_30pct():
    ok, _ = passes_post_render(0.29)
    assert ok


def test_post_render_near_black_above_30pct_rejects():
    ok, reason = passes_post_render(0.31)
    assert not ok
    assert "near_black_frac" in reason


# ---- DiFix3D parameter sanity ---------------------------------------------
def test_difix3d_paper_params_locked():
    assert_difix3d_params_match_paper(DIFIX3D_PARAMS)


def test_difix3d_param_drift_caught():
    bad = dict(DIFIX3D_PARAMS, timestep=200)
    with pytest.raises(AssertionError, match="timestep"):
        assert_difix3d_params_match_paper(bad)

"""Tests for PerFrameIntrinsics (paper App. B.1 (N,V,D) tensor)."""
import numpy as np
import pytest
from sana_wm_pipeline.stage02_pose.per_frame_intrinsics import (
    PerFrameIntrinsics, INTRINSICS_DIM, DEFAULT_VIEWS,
)


def test_intrinsics_dim_is_four():
    assert INTRINSICS_DIM == 4


def test_default_views_is_one():
    assert DEFAULT_VIEWS == 1


def test_from_flat_round_trip():
    N = 961
    fx = np.full(N, 700.0, dtype=np.float32)
    fy = np.full(N, 705.0, dtype=np.float32)
    cx = np.full(N, 640.0, dtype=np.float32)
    cy = np.full(N, 360.0, dtype=np.float32)
    intr = PerFrameIntrinsics.from_flat(fx, fy, cx, cy)
    assert intr.tensor.shape == (N, 1, 4)
    assert intr.tensor.dtype == np.float32
    assert intr.n_frames == N and intr.n_views == 1
    np.testing.assert_array_equal(intr.fx.squeeze(), fx)
    np.testing.assert_array_equal(intr.fy.squeeze(), fy)


def test_to_K_layout():
    N = 5
    intr = PerFrameIntrinsics.from_flat(
        np.full(N, 700, dtype=np.float32),
        np.full(N, 710, dtype=np.float32),
        np.full(N, 640, dtype=np.float32),
        np.full(N, 360, dtype=np.float32),
    )
    K = intr.to_K()
    assert K.shape == (N, 1, 3, 3)
    assert K[0, 0, 0, 0] == 700.0
    assert K[0, 0, 1, 1] == 710.0
    assert K[0, 0, 0, 2] == 640.0
    assert K[0, 0, 1, 2] == 360.0
    assert K[0, 0, 2, 2] == 1.0
    assert K[0, 0, 1, 0] == 0.0   # off-diagonal zero


def test_wrong_dtype_raises():
    with pytest.raises(ValueError, match="dtype"):
        PerFrameIntrinsics(tensor=np.zeros((5, 1, 4), dtype=np.float64))


def test_wrong_shape_raises():
    with pytest.raises(ValueError, match="N, V"):
        PerFrameIntrinsics(tensor=np.zeros((5, 1, 3), dtype=np.float32))

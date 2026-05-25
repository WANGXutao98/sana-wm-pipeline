"""Pi3X + MoGe-2 depth fusion per paper App. B.1.

Implements weighted least-squares per-frame scale recovery:
    s* = argmin_s Σ_i w_i (s · d_i^Pi3X − d_i^MoGe)²,  w_i = 1/d_i^Pi3X

with EMA temporal smoothing at momentum 0.99 (paper App. B.1).

Reference: arXiv:2605.15178v1, page 12, "Depth model upgrade" paragraph.
"""
from __future__ import annotations
import numpy as np

DEFAULT_EMA_MOMENTUM = 0.99   # paper App. B.1
MIN_DEPTH = 1e-3              # numerical floor for weight w_i = 1/d_i


def per_frame_scale_ls(d_pi3x: np.ndarray, d_moge: np.ndarray,
                       valid_mask: np.ndarray | None = None) -> float:
    """Single-frame weighted least-squares scale.

    Solves min_s Σ w_i (s a_i - b_i)²  with  w_i = 1/a_i.
    Closed-form: s* = Σ(w·a·b) / Σ(w·a²)  = Σ(b) / Σ(a)  [since w=1/a].

    But we keep the explicit weighted form to honor paper notation and to
    accept arbitrary external weights later.

    The constant MIN_DEPTH = 1e-3 acts as a numerical floor on a_i so that
    the inverse-depth weight w_i = 1/a_i never blows up when Pi3X reports a
    near-zero depth. Documented in the module-level constant.

    Args:
        d_pi3x: (N,) Pi3X depths at matched correspondence points (arbitrary scale)
        d_moge: (N,) MoGe-2 metric depths at the same points
        valid_mask: optional (N,) boolean mask; if None, all are used

    Returns:
        s* float — the metric scale that aligns Pi3X depths to MoGe-2 metric.
    """
    a = np.asarray(d_pi3x, dtype=np.float64)
    b = np.asarray(d_moge, dtype=np.float64)
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool)
        a, b = a[m], b[m]
    a = np.clip(a, MIN_DEPTH, None)
    w = 1.0 / a                           # inverse-depth weighting (paper)
    num = float((w * a * b).sum())        # Σ w·a·b
    den = float((w * a * a).sum()) + 1e-12
    return num / den


def fuse_metric_scale(d_pi3x_tracks: np.ndarray,
                      d_moge_tracks: np.ndarray,
                      valid_mask: np.ndarray | None = None,
                      momentum: float = DEFAULT_EMA_MOMENTUM) -> np.ndarray:
    """Per-frame metric-scale recovery with EMA smoothing.

    Args:
        d_pi3x_tracks: (T, N) per-frame Pi3X depths at matched track points.
                       T = number of frames, N = number of tracked points per frame.
        d_moge_tracks: (T, N) MoGe-2 metric depths at the same track points.
        valid_mask: (T, N) optional boolean mask of valid points per frame.
        momentum: EMA momentum (paper default 0.99). Pass 0.0 to disable smoothing.

    Returns:
        s_per_frame: (T,) float64 array of metric scale factors.
                     Frame 0 has no prior; uses raw least-squares estimate.
    """
    assert d_pi3x_tracks.shape == d_moge_tracks.shape, \
        f"shape mismatch: {d_pi3x_tracks.shape} vs {d_moge_tracks.shape}"
    if valid_mask is not None:
        assert valid_mask.shape == d_pi3x_tracks.shape, \
            f"valid_mask shape {valid_mask.shape} != {d_pi3x_tracks.shape}"
    T = d_pi3x_tracks.shape[0]
    s_out = np.empty(T, dtype=np.float64)
    s_prev: float | None = None
    for t in range(T):
        mt = valid_mask[t] if valid_mask is not None else None
        s_star = per_frame_scale_ls(d_pi3x_tracks[t], d_moge_tracks[t], mt)
        if s_prev is None or momentum == 0.0:
            s_t = s_star
        else:
            s_t = momentum * s_prev + (1.0 - momentum) * s_star
        s_out[t] = s_t
        s_prev = s_t
    return s_out

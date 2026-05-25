"""Pi3X + MoGe-2 depth fusion per paper App. B.1.

Implements weighted least-squares per-frame scale recovery:
    s* = argmin_s Σ_i w_i (s · d_i^Pi3X − d_i^MoGe)²,  w_i = 1/d_i^Pi3X

with EMA temporal smoothing at momentum 0.99 (paper App. B.1).

Degenerate (no-valid-point) frames return NaN per-frame and carry forward the
prior EMA scale; if the first frame is degenerate the output starts with NaN
until a valid frame appears.

Reference: arXiv:2605.15178v1, page 12, "Depth model upgrade" paragraph.
"""
from __future__ import annotations
import numpy as np

DEFAULT_EMA_MOMENTUM = 0.99   # paper App. B.1
MIN_DEPTH = 1e-3              # validity threshold for d_i (meters); see per_frame_scale_ls


def per_frame_scale_ls(d_pi3x: np.ndarray, d_moge: np.ndarray,
                       valid_mask: np.ndarray | None = None) -> float:
    """Single-frame weighted least-squares scale.

    Solves min_s Σ w_i (s a_i - b_i)²  with  w_i = 1/a_i.
    Closed-form: s* = Σ(w·a·b) / Σ(w·a²)  = Σ(b) / Σ(a)  [since w=1/a].

    But we keep the explicit weighted form to honor paper notation and to
    accept arbitrary external weights later.

    Validity criterion: a point is included in the LS sums iff
        a_i > MIN_DEPTH AND b_i > MIN_DEPTH AND np.isfinite(a_i) AND np.isfinite(b_i)
    (and `valid_mask[i]` is True, if provided). `MIN_DEPTH = 1e-3` (meters)
    is treated as a validity threshold, NOT a clip — sub-floor or non-positive
    depths are unphysical (a metric depth ≤ 1 mm is implausible) and would
    bias the inverse-depth-weighted LS estimate if clipped instead of dropped.
    If zero valid points remain, returns NaN.

    Args:
        d_pi3x: (N,) Pi3X depths at matched correspondence points (arbitrary scale, meters)
        d_moge: (N,) MoGe-2 metric depths at the same points (meters)
        valid_mask: optional (N,) boolean mask; if None, only the validity
                    criterion above is applied.

    Returns:
        s* float — the metric scale that aligns Pi3X depths to MoGe-2 metric,
        or NaN if no valid points remain.
    """
    a = np.asarray(d_pi3x, dtype=np.float64)
    b = np.asarray(d_moge, dtype=np.float64)
    valid = (a > MIN_DEPTH) & (b > MIN_DEPTH) & np.isfinite(a) & np.isfinite(b)
    if valid_mask is not None:
        valid = valid & np.asarray(valid_mask, dtype=bool)
    if valid.sum() == 0:
        return float("nan")
    a = a[valid]
    b = b[valid]
    w = 1.0 / a                           # inverse-depth weighting (paper)
    num = float((w * a * b).sum())        # Σ w·a·b
    den = float((w * a * a).sum())        # Σ w·a² = Σ a   (guaranteed > 0 since valid.sum()>0 and a>MIN_DEPTH)
    return num / den


def fuse_metric_scale(d_pi3x_tracks: np.ndarray,
                      d_moge_tracks: np.ndarray,
                      valid_mask: np.ndarray | None = None,
                      momentum: float = DEFAULT_EMA_MOMENTUM) -> np.ndarray:
    """Per-frame metric-scale recovery with EMA smoothing.

    Carry-forward behavior for degenerate frames: if the per-frame LS estimate
    returns NaN (no valid correspondence points after filtering), this function
    carries forward the previous frame's scale (s_t = s_prev) and does NOT
    advance the EMA state — so the next valid frame blends against the last
    *finite* scale, not against NaN. If the very first frame is degenerate,
    its output is NaN and s_prev remains unset until a finite estimate arrives.

    Args:
        d_pi3x_tracks: (T, N) per-frame Pi3X depths at matched track points.
                       T = number of frames, N = number of tracked points per frame.
        d_moge_tracks: (T, N) MoGe-2 metric depths at the same track points.
        valid_mask: (T, N) optional boolean mask of valid points per frame.
        momentum: EMA momentum (paper default 0.99). Pass 0.0 to disable smoothing.

    Returns:
        s_per_frame: (T,) float64 array of metric scale factors.
                     Frame 0 has no prior; uses raw least-squares estimate.
                     Degenerate frames carry forward the previous finite scale,
                     or NaN if no prior finite scale exists yet.
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
        if np.isnan(s_star):
            # Degenerate frame: carry forward last finite scale; do NOT advance EMA.
            if s_prev is None:
                s_t = float("nan")
            else:
                s_t = s_prev
            s_out[t] = s_t
            # Note: s_prev intentionally NOT updated — keep last finite scale.
            continue
        if s_prev is None or momentum == 0.0:
            s_t = s_star
        else:
            s_t = momentum * s_prev + (1.0 - momentum) * s_star
        s_out[t] = s_t
        s_prev = s_t
    return s_out

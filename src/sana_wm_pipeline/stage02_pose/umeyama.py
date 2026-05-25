"""Umeyama Sim(3) alignment with iterative inlier filtering.

Paper App. B.1: "Umeyama Sim(3) alignment [99] recovers the metric scale factor
from GT trajectories, with 80th-percentile inlier filtering."

Used in two places:
  (a) GT-pose annotation mode (Sekai-Game, DL3DV) to recover metric scale
  (b) Benchmark evaluation (App. D.3) to align Pi3X-recovered camera path to GT
"""
from __future__ import annotations
import numpy as np

DEFAULT_INLIER_PERCENTILE = 80.0   # paper App. B.1


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Single-pass Umeyama Sim(3) alignment (no inlier filtering).

    Solves for s, R, t minimizing sum_i || s R src_i + t - dst_i ||^2.

    Args:
        src: (N, 3) source points
        dst: (N, 3) target points

    Returns:
        (s, R, t) where s is scalar, R is (3,3), t is (3,).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must be (N, 3); got {src.shape}, {dst.shape}")
    if len(src) < 3:
        raise ValueError(f"need at least 3 correspondences; got {len(src)}")

    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    X = src - src_c
    Y = dst - dst_c

    # SVD of cross-covariance
    U, S, Vt = np.linalg.svd(X.T @ Y / len(X))
    D_diag = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D_diag[2, 2] = -1.0
    R = (U @ D_diag @ Vt).T

    var_src = float((X * X).sum() / len(X))
    if var_src < 1e-12:
        raise ValueError("source points are degenerate (zero variance)")
    s = float((S * np.diag(D_diag)).sum() / var_src)
    t = dst_c - s * R @ src_c
    return s, R, t


def umeyama_sim3_inlier_filter(src: np.ndarray, dst: np.ndarray,
                                inlier_percentile: float = DEFAULT_INLIER_PERCENTILE,
                                max_iter: int = 5,
                                ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively refit Umeyama Sim(3) with percentile-based inlier rejection.

    Algorithm:
      1. Fit Sim(3) on all correspondences.
      2. Compute residual r_i = ||s R src_i + t - dst_i||.
      3. Keep points with r_i <= percentile(r, inlier_percentile).
      4. Refit on inliers; repeat until inlier set stable or max_iter.

    Args:
        src: (N, 3) source points
        dst: (N, 3) target points
        inlier_percentile: keep points below this residual percentile (paper: 80)
        max_iter: maximum refit iterations

    Returns:
        (s, R, t, inlier_mask) where inlier_mask is (N,) bool of final inliers.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if not (0.0 < inlier_percentile <= 100.0):
        raise ValueError(f"inlier_percentile must be in (0, 100]; got {inlier_percentile}")

    N = len(src)
    mask = np.ones(N, dtype=bool)
    s, R, t = umeyama_sim3(src, dst)

    for _ in range(max_iter):
        # residuals on ALL points (not just current inliers) — keeps the cutoff stable
        res = np.linalg.norm(dst - (s * (src @ R.T) + t), axis=1)
        thr = float(np.percentile(res, inlier_percentile))
        new_mask = res <= thr
        # need enough inliers (>=3) to refit
        if new_mask.sum() < 3:
            break
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
        s, R, t = umeyama_sim3(src[mask], dst[mask])

    return s, R, t, mask

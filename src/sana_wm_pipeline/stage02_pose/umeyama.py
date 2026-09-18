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
_EPS = 1e-12


def umeyama_sim3(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares Sim(3) mapping ``src -> dst`` (Umeyama 1991).

    Finds ``s, R, t`` minimising ``sum_k || s R src_k + t - dst_k ||^2``.
    Returns ``(s, R(3x3), t(3,))``.

    This implementation matches the official SANA-WM code exactly.

    Args:
        src: (N, D) source points
        dst: (N, D) target points
        with_scale: if True, compute scale; if False, return s=1.0

    Returns:
        (s, R, t) where s is scalar, R is (D, D), t is (D,).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt

    if with_scale:
        var_src = (src_c ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / (var_src + _EPS))
    else:
        s = 1.0

    t = mu_dst - s * (R @ mu_src)
    return s, R, t


def recover_metric_scale(
    pred_positions: np.ndarray,
    gt_positions: np.ndarray,
    inlier_percentile: float = 80.0,
) -> float:
    """Metric scale factor aligning predicted to GT camera positions.

    Two-pass: fit Sim(3) on all points, keep points whose residual is below the
    ``inlier_percentile`` percentile, then re-fit and return the scale. This
    rejects gross trajectory outliers from imperfect structure prediction.

    This implementation matches the official SANA-WM code exactly.

    Args:
        pred_positions: (N, 3) predicted camera positions
        gt_positions: (N, 3) ground-truth camera positions
        inlier_percentile: percentile threshold for inlier selection (default: 80.0)

    Returns:
        scale: scalar metric scale factor
    """
    pred = np.asarray(pred_positions, dtype=np.float64)
    gt = np.asarray(gt_positions, dtype=np.float64)

    s, R, t = umeyama_sim3(pred, gt)
    resid = np.linalg.norm((s * (R @ pred.T)).T + t - gt, axis=1)
    thresh = np.percentile(resid, inlier_percentile)
    inliers = resid <= thresh
    if inliers.sum() >= 3:  # need enough points for a stable re-fit
        s, _, _ = umeyama_sim3(pred[inliers], gt[inliers])
    return float(s)


def umeyama_sim3_inlier_filter(src: np.ndarray, dst: np.ndarray,
                                inlier_percentile: float = DEFAULT_INLIER_PERCENTILE,
                                max_iter: int = 5,
                                ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively refit Umeyama Sim(3) with percentile-based inlier rejection.

    NOTE: This is an extended version with iterative refinement. For GT-pose mode
    alignment that exactly matches the official code, use `recover_metric_scale()`.

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

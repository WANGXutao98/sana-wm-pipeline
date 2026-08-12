"""Similarity (Sim(3)) alignment for GT-pose metric-scale recovery.

For sources with ground-truth trajectories (Sekai-Game, DL3DV), Pi3X predicts
the scene structure up to scale; we recover the metric scale factor by aligning
the predicted camera positions to the GT trajectory with a Umeyama Sim(3)
solution, using 80th-percentile inlier filtering for robustness (Appendix B.1).
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def umeyama_sim3(
    src: np.ndarray, dst: np.ndarray, with_scale: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares Sim(3) mapping ``src -> dst`` (Umeyama 1991).

    Finds ``s, R, t`` minimising ``sum_k || s R src_k + t - dst_k ||^2``.
    Returns ``(s, R(3x3), t(3,))``.
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

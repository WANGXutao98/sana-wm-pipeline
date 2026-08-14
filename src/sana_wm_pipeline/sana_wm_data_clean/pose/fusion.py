"""Pi3X + MoGe-2 depth fusion (SANA-WM Appendix B.1).

Pi3X gives long-sequence-consistent (but scale-ambiguous) depth; MoGe-2 gives a
per-frame metric-scale anchor. We fuse them by solving for a per-frame scale
factor ``s`` minimising

    sum_i w_i (s * d_Pi3X_i - d_MoGe_i)^2 ,   w_i = 1 / d_i

with inverse-depth weights, then smoothing the per-frame scale temporally with
an exponential moving average (momentum 0.99). The fused depth for a frame is
``s_smoothed * d_Pi3X``: it keeps Pi3X's temporally-consistent structure while
adopting MoGe-2's metric scale.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def solve_frame_scale(d_pi3x: np.ndarray, d_moge: np.ndarray) -> float:
    """Weighted-least-squares scale for one frame.

    Minimises ``sum_i w_i (s*a_i - b_i)^2`` with ``a=d_pi3x``, ``b=d_moge`` and
    inverse-depth weights ``w_i = 1/b_i`` (the metric reference depth). The
    closed-form solution is ``s = sum(w a b) / sum(w a^2)``.
    """
    a = np.asarray(d_pi3x, dtype=np.float64).ravel()
    b = np.asarray(d_moge, dtype=np.float64).ravel()
    # only use valid, positive-depth pixels
    mask = np.isfinite(a) & np.isfinite(b) & (a > _EPS) & (b > _EPS)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return 1.0
    w = 1.0 / (b + _EPS)
    num = np.sum(w * a * b)
    den = np.sum(w * a * a) + _EPS
    return float(num / den)


def fuse_depth_sequence(
    d_pi3x: np.ndarray, d_moge: np.ndarray, ema_momentum: float = 0.99
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse a (T, ...) depth sequence.

    Returns ``(fused_depth, scales)`` where ``scales`` is the EMA-smoothed
    per-frame scale (length T) and ``fused_depth[t] = scales[t] * d_pi3x[t]``.
    """
    d_pi3x = np.asarray(d_pi3x, dtype=np.float64)
    d_moge = np.asarray(d_moge, dtype=np.float64)
    T = d_pi3x.shape[0]

    scales = np.empty(T, dtype=np.float64)
    ema = None
    for t in range(T):
        s_raw = solve_frame_scale(d_pi3x[t], d_moge[t])
        ema = s_raw if ema is None else ema_momentum * ema + (1 - ema_momentum) * s_raw
        scales[t] = ema

    fused = scales.reshape((T,) + (1,) * (d_pi3x.ndim - 1)) * d_pi3x
    return fused, scales

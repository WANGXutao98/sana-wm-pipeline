"""Camera-specific filters (SANA-WM Appendix B.3).

Applied uniformly across all datasets. Given frame resolution (W, H) and
intrinsics (fx, fy, cx, cy):

* horizontal/vertical field of view  theta = 2*arctan(dim / (2*f)), must lie in
  [25 deg, 120 deg];
* focal divergence  |fx - fy| / ((fx + fy) / 2), must be <= 0.20;
* metric-scale coefficient of variation  std(s_t) / (mean(s_t) + eps) over the
  per-frame scale factors, must be <= 2.0.

All pure CPU math, deterministic, no external deps beyond numpy.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

_EPS = 1e-8


def fov_degrees(width: float, height: float, fx: float, fy: float) -> tuple[float, float]:
    """Horizontal and vertical field of view in degrees."""
    fov_x = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    fov_y = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    return fov_x, fov_y


def focal_divergence(fx: float, fy: float) -> float:
    """Symmetric normalized focal mismatch |fx-fy| / ((fx+fy)/2)."""
    denom = (fx + fy) / 2.0
    return abs(fx - fy) / (denom + _EPS)


def scale_cov(scale_factors: Sequence[float]) -> float:
    """Coefficient of variation of per-frame metric scale factors.

    Returns ``inf`` for an empty sequence so the gate REJECTS clips with no
    metric scale recovered, rather than silently passing them (CoV 0 <= max).
    A genuinely missing scale means metric reconstruction failed.
    """
    s = np.asarray(list(scale_factors), dtype=np.float64)
    if s.size == 0:
        return float("inf")
    return float(np.std(s) / (np.mean(s) + _EPS))


def camera_filter_pass(
    width: float,
    height: float,
    fx: float,
    fy: float,
    scale_factors: Sequence[float],
    cfg: dict,
) -> tuple[bool, list[str]]:
    """Return (kept, reject_reasons) for the camera filters.

    ``cfg`` is a camera-threshold mapping:
    ``{"fov_deg": [lo, hi], "focal_div_max": float, "scale_cov_max": float}``.
    Each per-frame intrinsic may also be passed as the median over frames.
    """
    reasons: list[str] = []

    fov_lo, fov_hi = cfg["fov_deg"]
    fov_x, fov_y = fov_degrees(width, height, fx, fy)
    if not (fov_lo <= fov_x <= fov_hi):
        reasons.append(f"fov_x={fov_x:.2f}deg outside [{fov_lo},{fov_hi}]")
    if not (fov_lo <= fov_y <= fov_hi):
        reasons.append(f"fov_y={fov_y:.2f}deg outside [{fov_lo},{fov_hi}]")

    fdiv = focal_divergence(fx, fy)
    if fdiv > cfg["focal_div_max"]:
        reasons.append(f"focal_div={fdiv:.3f} > {cfg['focal_div_max']}")

    cov = scale_cov(scale_factors)
    if cov > cfg["scale_cov_max"]:
        reasons.append(f"scale_cov={cov:.3f} > {cfg['scale_cov_max']}")

    return (len(reasons) == 0), reasons

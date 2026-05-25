"""Camera-specific pose quality filters per paper App. B.3 (page 13).

Three uniform constraints across all sources:
  1. FOV θ_x, θ_y ∈ [25°, 120°]
  2. |fx - fy| / ((fx + fy) / 2) ≤ 0.20
  3. std(s_t) / (mean(s_t) + ε) ≤ 2.0
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

FOV_DEG_MIN = 25.0           # paper App. B.3
FOV_DEG_MAX = 120.0          # paper App. B.3
FOCAL_DIV_MAX = 0.20         # paper App. B.3
SCALE_CV_MAX = 2.0           # paper App. B.3
EPSILON = 1e-6               # avoid div-by-zero in CV (paper uses "+ ε")


@dataclass(frozen=True)
class PoseQCResult:
    passed: bool
    reasons: tuple[str, ...]     # empty when passed, else one string per violation
    fov_x_min: float
    fov_x_max: float
    fov_y_min: float
    fov_y_max: float
    focal_divergence_max: float
    scale_cv: float


def horizontal_fov_deg(fx: np.ndarray, image_w: int) -> np.ndarray:
    """θ_x = 2 arctan(W / (2 fx)). Returns per-frame degrees."""
    fx = np.asarray(fx, dtype=np.float64)
    return 2.0 * np.degrees(np.arctan(image_w / (2.0 * fx)))


def vertical_fov_deg(fy: np.ndarray, image_h: int) -> np.ndarray:
    fy = np.asarray(fy, dtype=np.float64)
    return 2.0 * np.degrees(np.arctan(image_h / (2.0 * fy)))


def focal_divergence(fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Symmetric normalized focal mismatch |fx - fy| / ((fx + fy) / 2). Per-frame."""
    fx = np.asarray(fx, dtype=np.float64)
    fy = np.asarray(fy, dtype=np.float64)
    return np.abs(fx - fy) / (0.5 * (fx + fy) + EPSILON)


def scale_coefficient_of_variation(scale_per_frame: np.ndarray) -> float:
    """CV = std(s) / (mean(s) + ε). Paper App. B.3."""
    s = np.asarray(scale_per_frame, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("inf")
    return float(s.std(ddof=0) / (s.mean() + EPSILON))


def evaluate_pose_quality(intrinsics_NVD: np.ndarray,
                          image_wh: tuple[int, int],
                          scale_per_frame: np.ndarray) -> PoseQCResult:
    """Apply the three paper App. B.3 hard constraints.

    Args:
        intrinsics_NVD: (N, V, 4) per-frame intrinsics (fx, fy, cx, cy). V can be 1.
                        Filters use the V=0 (first/only) view.
        image_wh: (W, H) image resolution in pixels.
        scale_per_frame: (N,) per-frame metric scale s_t (NaN-tolerant).

    Returns:
        PoseQCResult with `passed` and `reasons` populated.
    """
    intr = np.asarray(intrinsics_NVD)
    if intr.ndim != 3 or intr.shape[2] != 4:
        raise ValueError(f"intrinsics_NVD must be (N, V, 4); got {intr.shape}")
    W, H = image_wh
    fx = intr[:, 0, 0]
    fy = intr[:, 0, 1]

    fov_x = horizontal_fov_deg(fx, W)
    fov_y = vertical_fov_deg(fy, H)
    div = focal_divergence(fx, fy)
    cv = scale_coefficient_of_variation(scale_per_frame)

    reasons: list[str] = []
    if not ((fov_x >= FOV_DEG_MIN).all() and (fov_x <= FOV_DEG_MAX).all()):
        reasons.append(
            f"fov_x out of [{FOV_DEG_MIN},{FOV_DEG_MAX}]: "
            f"[{float(fov_x.min()):.2f}, {float(fov_x.max()):.2f}]"
        )
    if not ((fov_y >= FOV_DEG_MIN).all() and (fov_y <= FOV_DEG_MAX).all()):
        reasons.append(
            f"fov_y out of [{FOV_DEG_MIN},{FOV_DEG_MAX}]: "
            f"[{float(fov_y.min()):.2f}, {float(fov_y.max()):.2f}]"
        )
    if (div > FOCAL_DIV_MAX).any():
        reasons.append(f"focal_divergence={float(div.max()):.3f} > {FOCAL_DIV_MAX}")
    if cv > SCALE_CV_MAX:
        reasons.append(f"scale_cv={cv:.3f} > {SCALE_CV_MAX}")

    return PoseQCResult(
        passed=(len(reasons) == 0),
        reasons=tuple(reasons),
        fov_x_min=float(fov_x.min()),
        fov_x_max=float(fov_x.max()),
        fov_y_min=float(fov_y.min()),
        fov_y_max=float(fov_y.max()),
        focal_divergence_max=float(div.max()),
        scale_cv=cv,
    )

"""Lightweight geometric quality checker for single pose samples.

This module provides fast validation of camera pose geometric properties:
- Level 1: Mathematical validity (SO(3), first frame alignment, no NaN/Inf)
- Level 2: Physical reasonableness (FOV range, focal divergence, scale CV)

Designed for single-sample verification with minimal dependencies (NumPy only).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

# Level 1: Mathematical validity thresholds
SO3_DET_ATOL = 1e-3          # |det(R) - 1.0| tolerance
SO3_ORTH_ATOL = 1e-3         # |R @ R^T - I| tolerance
FIRST_FRAME_ATOL = 1e-2      # |pose[0] - I| tolerance

# Level 2: Physical reasonableness thresholds
FOV_DEG_MIN = 25.0           # Minimum FOV (degrees)
FOV_DEG_MAX = 120.0          # Maximum FOV (degrees)
FOCAL_DIV_MAX = 0.20         # Maximum |fx-fy|/((fx+fy)/2)
SCALE_CV_MAX = 2.0           # Maximum scale coefficient of variation
EPSILON = 1e-6               # Avoid division by zero


# ── Result Data Structure ────────────────────────────────────────────────────

@dataclass
class GeometryCheckResult:
    """Geometry quality check result."""

    # Overall verdict
    passed: bool
    level1_passed: bool
    level2_passed: bool

    # Level 1: Mathematical validity
    so3_valid: bool
    so3_det_mean: float
    so3_det_std: float
    so3_orth_err: float

    first_frame_aligned: bool
    first_frame_dev: float

    no_nan_inf: bool
    nan_inf_fields: list[str]

    # Level 2: Physical reasonableness
    fov_valid: bool
    fov_x_min: float
    fov_x_max: float
    fov_y_min: float
    fov_y_max: float

    focal_div_valid: bool
    focal_div_max: float

    scale_cv_valid: bool
    scale_cv: float

    # Failure reasons
    failure_reasons: list[str]

    # Metadata
    num_frames: int
    image_wh: tuple[int, int]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert NumPy types to Python native types for JSON serialization
        def convert_value(v):
            if isinstance(v, np.integer):
                return int(v)
            elif isinstance(v, np.floating):
                return float(v)
            elif isinstance(v, np.bool_):
                return bool(v)
            elif isinstance(v, np.ndarray):
                return v.tolist()
            elif isinstance(v, (list, tuple)):
                return [convert_value(x) for x in v]
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            return v

        return {k: convert_value(v) for k, v in d.items()}

    def to_json(self, path: Optional[Path] = None) -> str:
        """Convert to JSON string or save to file."""
        json_str = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path:
            Path(path).write_text(json_str, encoding='utf-8')
        return json_str


# ── Level 1: Mathematical Validity Checks ────────────────────────────────────

def _check_so3(poses_c2w: np.ndarray) -> tuple[bool, float, float, float]:
    """Check SO(3) validity: det(R)=1 and R @ R^T = I.

    Args:
        poses_c2w: (N, 4, 4) camera-to-world transformation matrices

    Returns:
        (valid, det_mean, det_std, orth_err)
    """
    R = poses_c2w[:, :3, :3].astype(np.float64)

    # Check determinant
    dets = np.linalg.det(R)
    det_mean = float(dets.mean())
    det_std = float(dets.std())
    det_valid = abs(det_mean - 1.0) <= SO3_DET_ATOL

    # Check orthogonality: R @ R^T = I
    RRT = R @ R.transpose(0, 2, 1)  # (N, 3, 3)
    I = np.eye(3)
    orth_err = float(np.max(np.abs(RRT - I)))
    orth_valid = orth_err <= SO3_ORTH_ATOL

    valid = det_valid and orth_valid
    return valid, det_mean, det_std, orth_err


def _check_first_frame(poses_c2w: np.ndarray) -> tuple[bool, float]:
    """Check if first frame is aligned to identity (origin).

    Args:
        poses_c2w: (N, 4, 4) camera-to-world transformation matrices

    Returns:
        (aligned, max_deviation)
    """
    I = np.eye(4)
    dev = float(np.max(np.abs(poses_c2w[0].astype(np.float64) - I)))
    aligned = dev <= FIRST_FRAME_ATOL
    return aligned, dev


def _check_nan_inf(
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    scale: np.ndarray
) -> tuple[bool, list[str]]:
    """Check for NaN/Inf values in all arrays.

    Args:
        poses_c2w: (N, 4, 4) camera-to-world transformation matrices
        intrinsics: (N, V, 4) or (N, 4) camera intrinsics
        scale: (N,) per-frame scale factors

    Returns:
        (valid, fields_with_nan_inf)
    """
    fields = []

    if not np.isfinite(poses_c2w).all():
        count = int((~np.isfinite(poses_c2w)).sum())
        fields.append(f"poses_c2w: {count} non-finite values")

    if not np.isfinite(intrinsics).all():
        count = int((~np.isfinite(intrinsics)).sum())
        fields.append(f"intrinsics: {count} non-finite values")

    if not np.isfinite(scale).all():
        count = int((~np.isfinite(scale)).sum())
        fields.append(f"scale: {count} non-finite values")

    valid = len(fields) == 0
    return valid, fields


# ── Level 2: Physical Reasonableness Checks ──────────────────────────────────

def _horizontal_fov_deg(fx: np.ndarray, image_w: int) -> np.ndarray:
    """Compute horizontal field of view: θ_x = 2 * arctan(W / (2 * fx))."""
    fx = np.asarray(fx, dtype=np.float64)
    return 2.0 * np.degrees(np.arctan(image_w / (2.0 * fx)))


def _vertical_fov_deg(fy: np.ndarray, image_h: int) -> np.ndarray:
    """Compute vertical field of view: θ_y = 2 * arctan(H / (2 * fy))."""
    fy = np.asarray(fy, dtype=np.float64)
    return 2.0 * np.degrees(np.arctan(image_h / (2.0 * fy)))


def _check_fov(
    intrinsics: np.ndarray,
    image_wh: tuple[int, int]
) -> tuple[bool, float, float, float, float]:
    """Check if FOV is within valid range [25°, 120°].

    Args:
        intrinsics: (N, V, 4) camera intrinsics [fx, fy, cx, cy]
        image_wh: (W, H) image dimensions

    Returns:
        (valid, fov_x_min, fov_x_max, fov_y_min, fov_y_max)
    """
    W, H = image_wh

    # Extract fx, fy from first view (V=0)
    if intrinsics.ndim == 3:
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 0, 1]
    else:  # (N, 4)
        fx = intrinsics[:, 0]
        fy = intrinsics[:, 1]

    fov_x = _horizontal_fov_deg(fx, W)
    fov_y = _vertical_fov_deg(fy, H)

    fov_x_min = float(fov_x.min())
    fov_x_max = float(fov_x.max())
    fov_y_min = float(fov_y.min())
    fov_y_max = float(fov_y.max())

    x_valid = (fov_x >= FOV_DEG_MIN).all() and (fov_x <= FOV_DEG_MAX).all()
    y_valid = (fov_y >= FOV_DEG_MIN).all() and (fov_y <= FOV_DEG_MAX).all()

    valid = x_valid and y_valid
    return valid, fov_x_min, fov_x_max, fov_y_min, fov_y_max


def _check_focal_divergence(intrinsics: np.ndarray) -> tuple[bool, float]:
    """Check focal length divergence: |fx - fy| / ((fx + fy) / 2).

    Args:
        intrinsics: (N, V, 4) or (N, 4) camera intrinsics

    Returns:
        (valid, max_divergence)
    """
    # Extract fx, fy
    if intrinsics.ndim == 3:
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 0, 1]
    else:  # (N, 4)
        fx = intrinsics[:, 0]
        fy = intrinsics[:, 1]

    fx = fx.astype(np.float64)
    fy = fy.astype(np.float64)

    div = np.abs(fx - fy) / (0.5 * (fx + fy) + EPSILON)
    max_div = float(div.max())

    valid = max_div <= FOCAL_DIV_MAX
    return valid, max_div


def _check_scale_cv(scale: np.ndarray) -> tuple[bool, float]:
    """Check scale coefficient of variation: std(s) / (mean(s) + ε).

    Args:
        scale: (N,) per-frame scale factors

    Returns:
        (valid, cv)
    """
    s = scale.astype(np.float64)
    s = s[np.isfinite(s)]  # Filter NaN/Inf

    if s.size == 0:
        return False, float('inf')

    cv = float(s.std(ddof=0) / (s.mean() + EPSILON))
    valid = cv <= SCALE_CV_MAX
    return valid, cv


# ── Main Check Function ──────────────────────────────────────────────────────

def check_pose_geometry(
    poses_c2w: np.ndarray,
    intrinsics: np.ndarray,
    scale: np.ndarray,
    image_wh: tuple[int, int],
) -> GeometryCheckResult:
    """Execute complete geometric quality checks.

    Args:
        poses_c2w: (N, 4, 4) camera-to-world transformation matrices
        intrinsics: (N, V, 4) or (N, 4) camera intrinsics [fx, fy, cx, cy]
        scale: (N,) per-frame metric scale factors
        image_wh: (W, H) image dimensions in pixels

    Returns:
        GeometryCheckResult with detailed check results
    """
    poses_c2w = np.asarray(poses_c2w, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)

    num_frames = int(poses_c2w.shape[0])
    failure_reasons = []

    # ── Level 1: Mathematical Validity ───────────────────────────────────────

    # Check SO(3)
    so3_valid, so3_det_mean, so3_det_std, so3_orth_err = _check_so3(poses_c2w)
    if not so3_valid:
        if abs(so3_det_mean - 1.0) > SO3_DET_ATOL:
            failure_reasons.append(
                f"SO(3) invalid: det_mean={so3_det_mean:.6f} "
                f"(expected 1.0 ± {SO3_DET_ATOL})"
            )
        if so3_orth_err > SO3_ORTH_ATOL:
            failure_reasons.append(
                f"SO(3) invalid: orth_err={so3_orth_err:.3e} "
                f"(expected ≤ {SO3_ORTH_ATOL})"
            )

    # Check first frame alignment
    first_frame_aligned, first_frame_dev = _check_first_frame(poses_c2w)
    if not first_frame_aligned:
        failure_reasons.append(
            f"First frame not aligned: dev={first_frame_dev:.6f} "
            f"(expected ≤ {FIRST_FRAME_ATOL})"
        )

    # Check NaN/Inf
    no_nan_inf, nan_inf_fields = _check_nan_inf(poses_c2w, intrinsics, scale)
    if not no_nan_inf:
        failure_reasons.extend(nan_inf_fields)

    level1_passed = so3_valid and first_frame_aligned and no_nan_inf

    # ── Level 2: Physical Reasonableness ─────────────────────────────────────

    # Check FOV
    fov_valid, fov_x_min, fov_x_max, fov_y_min, fov_y_max = _check_fov(
        intrinsics, image_wh
    )
    if not fov_valid:
        if fov_x_min < FOV_DEG_MIN or fov_x_max > FOV_DEG_MAX:
            failure_reasons.append(
                f"FOV horizontal out of range: [{fov_x_min:.1f}°, {fov_x_max:.1f}°] "
                f"(expected [{FOV_DEG_MIN}°, {FOV_DEG_MAX}°])"
            )
        if fov_y_min < FOV_DEG_MIN or fov_y_max > FOV_DEG_MAX:
            failure_reasons.append(
                f"FOV vertical out of range: [{fov_y_min:.1f}°, {fov_y_max:.1f}°] "
                f"(expected [{FOV_DEG_MIN}°, {FOV_DEG_MAX}°])"
            )

    # Check focal divergence
    focal_div_valid, focal_div_max = _check_focal_divergence(intrinsics)
    if not focal_div_valid:
        failure_reasons.append(
            f"Focal divergence too high: {focal_div_max:.3f} "
            f"(expected ≤ {FOCAL_DIV_MAX})"
        )

    # Check scale CV
    scale_cv_valid, scale_cv = _check_scale_cv(scale)
    if not scale_cv_valid:
        failure_reasons.append(
            f"Scale CV too high: {scale_cv:.3f} (expected ≤ {SCALE_CV_MAX})"
        )

    level2_passed = fov_valid and focal_div_valid and scale_cv_valid

    # ── Overall Verdict ──────────────────────────────────────────────────────

    passed = level1_passed and level2_passed

    return GeometryCheckResult(
        passed=passed,
        level1_passed=level1_passed,
        level2_passed=level2_passed,
        so3_valid=so3_valid,
        so3_det_mean=so3_det_mean,
        so3_det_std=so3_det_std,
        so3_orth_err=so3_orth_err,
        first_frame_aligned=first_frame_aligned,
        first_frame_dev=first_frame_dev,
        no_nan_inf=no_nan_inf,
        nan_inf_fields=nan_inf_fields,
        fov_valid=fov_valid,
        fov_x_min=fov_x_min,
        fov_x_max=fov_x_max,
        fov_y_min=fov_y_min,
        fov_y_max=fov_y_max,
        focal_div_valid=focal_div_valid,
        focal_div_max=focal_div_max,
        scale_cv_valid=scale_cv_valid,
        scale_cv=scale_cv,
        failure_reasons=failure_reasons,
        num_frames=num_frames,
        image_wh=image_wh,
    )


# ── JSON Loader ──────────────────────────────────────────────────────────────

def _infer_image_wh(intrinsics: np.ndarray) -> tuple[int, int]:
    """Infer image dimensions from principal point (cx, cy).

    Assumes principal point is at image center: W = 2*cx, H = 2*cy
    """
    if intrinsics.ndim == 3:
        cx = intrinsics[0, 0, 2]
        cy = intrinsics[0, 0, 3]
    else:
        cx = intrinsics[0, 2]
        cy = intrinsics[0, 3]

    return (int(cx * 2), int(cy * 2))


def load_pose_from_json(json_path: Path) -> dict:
    """Load pose data from JSON file with field name variants support.

    Args:
        json_path: Path to pose_artifact_default.json

    Returns:
        {
            'poses_c2w': np.ndarray (N, 4, 4),
            'intrinsics': np.ndarray (N, V, 4),
            'scale': np.ndarray (N,),
            'image_wh': tuple[int, int],
            'missing_fields': list[str]
        }
    """
    json_path = Path(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    missing_fields = []

    # Load poses_c2w (support variants)
    poses_key = None
    for key in ['poses_c2w', 'poses', 'camera_poses', 'extrinsics']:
        if key in data:
            poses_key = key
            break

    if poses_key is None:
        missing_fields.append('poses_c2w')
        poses_c2w = None
    else:
        poses_c2w = np.array(data[poses_key], dtype=np.float64)

    # Load intrinsics (support variants)
    intr_key = None
    for key in ['intrinsics', 'intrinsic', 'K', 'camera_intrinsics']:
        if key in data:
            intr_key = key
            break

    if intr_key is None:
        missing_fields.append('intrinsics')
        intrinsics = None
    else:
        intrinsics = np.array(data[intr_key], dtype=np.float64)

    # Load scale (support variants)
    scale_key = None
    for key in ['scale_per_frame', 'scale', 'scales', 'metric_scale']:
        if key in data:
            scale_key = key
            break

    if scale_key is None:
        missing_fields.append('scale')
        scale = None
    else:
        scale = np.array(data[scale_key], dtype=np.float64)

    # Infer or load image_wh
    image_wh = None
    for key in ['image_wh', 'resolution', 'image_size']:
        if key in data:
            wh = data[key]
            if isinstance(wh, (list, tuple)) and len(wh) == 2:
                image_wh = tuple(wh)
                break

    # Fallback: infer from intrinsics
    if image_wh is None and intrinsics is not None:
        image_wh = _infer_image_wh(intrinsics)

    if image_wh is None:
        missing_fields.append('image_wh')

    return {
        'poses_c2w': poses_c2w,
        'intrinsics': intrinsics,
        'scale': scale,
        'image_wh': image_wh,
        'missing_fields': missing_fields,
    }


def check_pose_json(json_path: Path) -> GeometryCheckResult:
    """Convenient interface to check pose JSON file directly.

    Args:
        json_path: Path to pose_artifact_default.json

    Returns:
        GeometryCheckResult

    Raises:
        ValueError: If required fields are missing
    """
    json_path = Path(json_path)
    data = load_pose_from_json(json_path)

    if data['missing_fields']:
        raise ValueError(
            f"Missing required fields in {json_path}: {data['missing_fields']}"
        )

    return check_pose_geometry(
        poses_c2w=data['poses_c2w'],
        intrinsics=data['intrinsics'],
        scale=data['scale'],
        image_wh=data['image_wh'],
    )


# ── CLI Entry Point (optional) ───────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python lite_geometric_check.py <pose_json_path> [output_json]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    result = check_pose_json(input_path)

    if output_path:
        result.to_json(output_path)
        print(f"✓ Report saved to: {output_path}")
    else:
        print(result.to_json())

    sys.exit(0 if result.passed else 1)

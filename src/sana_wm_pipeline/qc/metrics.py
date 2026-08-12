"""Stage 1 QC: pure computation functions. No IO, no GPU."""
from __future__ import annotations
import re
import numpy as np
from sana_wm_pipeline.stage02_pose.pose_quality import evaluate_pose_quality
from sana_wm_pipeline.stage04_filter.visual_metrics import mean_saturation

_SO3_DET_ATOL = 1e-3
_SO3_ORTH_ATOL = 1e-3
_FIRST_FRAME_ATOL = 1e-2

# Strong camera-action words per paper §4.4 (phrases, not single words, to reduce false positives)
_CAMERA_ACTION_PATTERNS: list[str] = [
    r"\bpan(?:s|ned|ning)?\s+(?:left|right)\b",
    r"\btilt(?:s|ed|ing)?\s+(?:up|down)\b",
    r"\bzoom(?:s|ed|ing)?\s+(?:in|out)\b",
    r"\bdolly\b",
    r"\bcamera\s+(?:moves?|tracks?|follows?|sweeps?|pushes?|pulls?)\b",
    r"\bcamera\s+(?:pan|tilt|roll|orbit)s?\b",
    r"\b(?:tracking|follow)\s+shot\b",
]
_CAMERA_ACTION_RE = [re.compile(p, re.IGNORECASE) for p in _CAMERA_ACTION_PATTERNS]


def check_so3(poses_c2w: np.ndarray) -> tuple[float, float, float]:
    R = poses_c2w[:, :3, :3].astype(np.float64)
    dets = np.linalg.det(R)
    orth_err = float(np.max(np.abs(R @ R.transpose(0, 2, 1) - np.eye(3))))
    return float(dets.mean()), float(dets.std()), orth_err


def check_first_frame(poses_c2w: np.ndarray, atol: float = _FIRST_FRAME_ATOL) -> tuple[bool, float]:
    dev = float(np.max(np.abs(poses_c2w[0].astype(np.float64) - np.eye(4))))
    return dev <= atol, dev


def check_trajectory(poses_c2w: np.ndarray, jump_threshold_m: float) -> tuple[float, float, float, int]:
    t = poses_c2w[:, :3, 3].astype(np.float64)
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    if steps.size == 0:
        return 0.0, 0.0, 0.0, 0
    return float(steps.sum()), float(steps.mean()), float(steps.max()), int((steps > jump_threshold_m).sum())


def check_no_nan_inf(arrays: dict[str, np.ndarray]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for name, arr in arrays.items():
        if not np.isfinite(np.asarray(arr)).all():
            reasons.append(f"{name}: {int((~np.isfinite(arr)).sum())} non-finite values")
    return len(reasons) == 0, reasons


def check_caption(caption: str, min_len: int = 50) -> tuple[bool, int]:
    s = caption.strip()
    # If min_len is 0, allow any caption (including empty/placeholder)
    if min_len == 0:
        return True, len(s)
    # Otherwise, reject placeholders and check length
    if s.lower() in {"", "n/a", "none", "no caption", "null", "tbd"}:
        return False, len(s)
    return len(s) >= min_len, len(s)


def check_color_saturation(frames_rgb: np.ndarray) -> float:
    """Mean HSV-S across all frames; [0,180] range matching paper Table 6."""
    return mean_saturation(frames_rgb)


def check_caption_camera_words(caption: str) -> list[str]:
    """Return list of matched strong camera-action phrases (empty = clean)."""
    hits: list[str] = []
    for pat in _CAMERA_ACTION_RE:
        m = pat.search(caption)
        if m:
            hits.append(m.group(0))
    return hits


def compute_stage1_metrics(
    poses: np.ndarray,
    intrinsics: np.ndarray,
    scale: np.ndarray,
    caption: str,
    meta_T: int,
    image_wh: tuple[int, int],
    jump_threshold_m: float,
    min_caption_len: int,
    frames_rgb: np.ndarray | None = None,  # optional; if provided → compute saturation
) -> dict:
    """All Stage 1 checks. Returns flat dict of Python scalars/lists for JSON."""
    T = int(poses.shape[0])
    reasons: list[str] = []

    t_aligned = (poses.shape[0] == intrinsics.shape[0] == scale.shape[0] == meta_T)
    if not t_aligned:
        reasons.append(
            f"t_mismatch: poses={poses.shape[0]} intr={intrinsics.shape[0]} "
            f"scale={scale.shape[0]} meta={meta_T}"
        )

    nan_ok, nan_reasons = check_no_nan_inf({"poses": poses, "intrinsics": intrinsics, "scale": scale})
    reasons.extend(nan_reasons)

    det_mean, det_std, orth_err = check_so3(poses)
    so3_valid = abs(det_mean - 1.0) <= _SO3_DET_ATOL and orth_err <= _SO3_ORTH_ATOL
    if not so3_valid:
        reasons.append(f"so3_invalid: det_mean={det_mean:.6f} orth_err={orth_err:.2e}")

    first_ok, first_dev = check_first_frame(poses)
    if not first_ok:
        reasons.append(f"first_frame_dev={first_dev:.4f}")

    traj_total, step_mean, step_max, n_jumps = check_trajectory(poses, jump_threshold_m)

    pqr = evaluate_pose_quality(intrinsics, image_wh, scale)
    if not pqr.passed:
        reasons.extend(list(pqr.reasons))

    cap_ok, cap_len = check_caption(caption, min_caption_len)
    if not cap_ok:
        reasons.append(f"caption_len={cap_len} < {min_caption_len} or placeholder")

    camera_words = check_caption_camera_words(caption)

    saturation = None
    if frames_rgb is not None:
        try:
            saturation = round(check_color_saturation(frames_rgb), 2)
        except Exception as e:
            reasons.append(f"saturation_error: {e}")

    return {
        "T": T,
        "t_aligned": t_aligned,
        "no_nan_inf": nan_ok,
        "so3_valid": so3_valid,
        "det_R_mean": round(det_mean, 8),
        "orth_err_max": float(f"{orth_err:.3e}"),
        "first_frame_ok": first_ok,
        "first_frame_dev": round(first_dev, 6),
        "traj_total_m": round(traj_total, 3),
        "step_mean_m": round(step_mean, 4),
        "step_max_m": round(step_max, 4),
        "n_jumps": n_jumps,
        "jump_threshold_m": jump_threshold_m,
        "pose_quality_ok": pqr.passed,
        "fov_ok": pqr.passed or not any("fov" in r.lower() for r in pqr.reasons),
        "focal_div_ok": pqr.focal_divergence_max <= 0.20,
        "focal_div_max": round(pqr.focal_divergence_max, 4),
        "scale_cv": round(pqr.scale_cv, 4),
        "fx_mean": round(float(intrinsics[:, 0, 0].mean()), 2),
        "caption_ok": cap_ok,
        "caption_len": cap_len,
        "camera_words": camera_words,
        "saturation": saturation,
        "scale_all_ones": bool(np.all(scale == 1.0)),
        "reasons": reasons,
    }

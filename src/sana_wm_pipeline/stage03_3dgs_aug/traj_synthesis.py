"""DL3DV 3DGS trajectory synthesis (paper App. B.2).

Per scene we emit 40 candidate camera trajectories:
  * 10  spline-interp trajectories around the training views
  * 30  from 8 motion families: orbit, spiral, dolly, fly-through,
        random_walk, crane/boom, pendulum, compound

Each trajectory is:
  - scaled by the scene's median camera-to-content distance
  - anchored at a sampled training camera
  - oriented toward the Gaussian centroid (forward-facing scene)
  - clamped to the training coverage volume
  - Gaussian-smoothed with σ ≈ n_frames / 200
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


FAMILIES_30: Tuple[str, ...] = (
    "orbit", "spiral", "dolly", "flythrough",
    "random_walk", "crane", "pendulum", "compound",
)
# Per paper App. B.2: 30 trajectories split across 8 families.  We distribute
# evenly with a small tilt toward the families that have higher reported value
# in App. B.2 ("orbit/spiral/dolly/flythrough/random_walk/compound" = 4 each,
# "crane/pendulum" = 3 each).
FAMILY_COUNTS: Dict[str, int] = {
    "orbit": 4, "spiral": 4, "dolly": 4, "flythrough": 4,
    "random_walk": 4, "crane": 3, "pendulum": 3, "compound": 4,
}
assert sum(FAMILY_COUNTS.values()) == 30, "30-traj allotment must sum to 30"
TOTAL_TRAJECTORIES: int = 10 + 30  # paper App. B.2: 40 per scene


@dataclass
class Trajectory:
    family: str
    poses_c2w: np.ndarray   # (N, 4, 4) float32
    intrinsics: np.ndarray  # (N, 4)    float32  (fx, fy, cx, cy)


def synthesize_40_trajectories(
    scene_stats: dict,
    n_frames: int = 961,
    fps: int = 16,
    seed: int = 0,
) -> List[Trajectory]:
    rng = np.random.default_rng(seed)
    centroid = np.asarray(scene_stats["centroid"], dtype=np.float32)
    radius = float(scene_stats["median_radius"])
    train_cams = np.asarray(scene_stats["training_cam_positions"], dtype=np.float32)
    if len(train_cams) == 0:
        raise ValueError("scene_stats.training_cam_positions is empty")

    K = _default_K(n_frames)
    trajs: List[Trajectory] = []

    # 10 spline-interp trajectories
    for k in range(10):
        idx = rng.choice(len(train_cams), size=min(6, len(train_cams)),
                         replace=False)
        positions = _spline_traj(train_cams[idx], n_frames)
        positions = _gaussian_smooth(positions, sigma=n_frames / 200.0)
        positions = _clamp_to_coverage(positions, scene_stats)
        poses = _pose_from_positions(positions, look_at=centroid)
        trajs.append(Trajectory("spline_interp", poses, K))

    # 30 family trajectories (8 families, paper distribution)
    for fam, n in FAMILY_COUNTS.items():
        for k in range(n):
            positions = _family_traj(fam, scene_stats, n_frames, seed=int(rng.integers(1 << 31)))
            positions = _gaussian_smooth(positions, sigma=n_frames / 200.0)
            positions = _clamp_to_coverage(positions, scene_stats)
            poses = _pose_from_positions(positions, look_at=centroid)
            trajs.append(Trajectory(fam, poses, K))

    if len(trajs) != TOTAL_TRAJECTORIES:
        raise AssertionError(
            f"expected {TOTAL_TRAJECTORIES} trajectories, built {len(trajs)}"
        )
    return trajs


# ---- Geometry helpers ------------------------------------------------------
def _default_K(n_frames: int, fx: float = 700.0, fy: float = 700.0,
               cx: float = 640.0, cy: float = 360.0) -> np.ndarray:
    return np.tile([fx, fy, cx, cy], (n_frames, 1)).astype(np.float32)


def _spline_traj(waypoints: np.ndarray, n_frames: int) -> np.ndarray:
    """Catmull-Rom-ish spline through `waypoints`; falls back to linear interp."""
    waypoints = np.asarray(waypoints, dtype=np.float32)
    if len(waypoints) < 2:
        return np.tile(waypoints[0], (n_frames, 1))
    seg_t = np.linspace(0.0, 1.0, len(waypoints))
    t = np.linspace(0.0, 1.0, n_frames)
    return np.stack(
        [np.interp(t, seg_t, waypoints[:, d]) for d in range(3)], axis=-1,
    ).astype(np.float32)


def _gaussian_smooth(positions: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian smoothing along the time axis (per coordinate)."""
    if sigma <= 0.5 or positions.shape[0] < 3:
        return positions
    half = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    n = positions.shape[0]
    pad = np.pad(positions, ((half, half), (0, 0)), mode="edge")
    out = np.empty_like(positions)
    for d in range(positions.shape[1]):
        out[:, d] = np.convolve(pad[:, d], kernel, mode="valid")[:n]
    return out


def _clamp_to_coverage(positions: np.ndarray, stats: dict) -> np.ndarray:
    """Soft clamp to the bounding box of the training cameras + 10% margin."""
    train_cams = np.asarray(stats["training_cam_positions"], dtype=np.float32)
    lo = train_cams.min(axis=0)
    hi = train_cams.max(axis=0)
    span = hi - lo
    lo_pad = lo - 0.1 * span
    hi_pad = hi + 0.1 * span
    return np.clip(positions, lo_pad, hi_pad)


def _pose_from_positions(positions: np.ndarray, look_at: np.ndarray) -> np.ndarray:
    """Build (N, 4, 4) c2w poses with each frame looking at `look_at`."""
    N = positions.shape[0]
    fwd = look_at[None, :] - positions
    fwd = _safe_normalize(fwd)
    up = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float32), (N, 1))
    right = _safe_normalize(np.cross(up, fwd))
    up_corr = np.cross(fwd, right)
    R = np.stack([right, up_corr, fwd], axis=-1)   # (N, 3, 3)
    poses = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))
    poses[:, :3, :3] = R.astype(np.float32)
    poses[:, :3, 3] = positions.astype(np.float32)
    return poses


def _safe_normalize(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


# ---- Motion families -------------------------------------------------------
def _family_traj(family: str, stats: dict, n_frames: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centroid = np.asarray(stats["centroid"], dtype=np.float32)
    radius = float(stats["median_radius"])
    train_cams = np.asarray(stats["training_cam_positions"], dtype=np.float32)
    anchor = train_cams[int(rng.integers(len(train_cams)))]
    t = np.linspace(0.0, 1.0, n_frames, dtype=np.float32)

    if family == "orbit":
        theta = 2 * np.pi * t
        return centroid + radius * np.stack(
            [np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=-1,
        )
    if family == "spiral":
        theta = 4 * np.pi * t
        r = radius * (1.0 + 0.5 * t)
        return centroid + np.stack(
            [r * np.cos(theta), 0.5 * radius * (t - 0.5), r * np.sin(theta)],
            axis=-1,
        )
    if family == "dolly":
        direction = _safe_normalize((centroid - anchor)[None, :])[0]
        return anchor[None, :] + (t[:, None] - 0.5) * radius * direction[None, :]
    if family == "flythrough":
        direction = _safe_normalize(rng.standard_normal(3).astype(np.float32)[None, :])[0]
        return anchor[None, :] + (t[:, None] - 0.5) * 2.0 * radius * direction[None, :]
    if family == "random_walk":
        step = rng.standard_normal((n_frames, 3)).astype(np.float32) * (radius / 50.0)
        return anchor[None, :] + np.cumsum(step, axis=0)
    if family == "crane":
        ax = rng.standard_normal(3).astype(np.float32)
        ax = _safe_normalize(ax[None, :])[0]
        return anchor[None, :] + (t[:, None] - 0.5) * radius * ax[None, :]
    if family == "pendulum":
        phase = 2 * np.pi * t
        side = _safe_normalize(rng.standard_normal(3).astype(np.float32)[None, :])[0]
        return anchor[None, :] + 0.5 * radius * np.sin(phase)[:, None] * side[None, :]
    if family == "compound":
        theta = 2 * np.pi * t
        side = _safe_normalize(rng.standard_normal(3).astype(np.float32)[None, :])[0]
        circle = radius * np.stack(
            [np.cos(theta), np.zeros_like(theta), np.sin(theta)], axis=-1,
        )
        sway = 0.3 * radius * np.sin(3 * theta)[:, None] * side[None, :]
        return centroid + circle + sway
    raise ValueError(f"unknown trajectory family {family!r}")

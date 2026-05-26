"""Thin wrapper around the FCGS (Fast Compressed Gaussian Splatting) optimiser.

Paper App. B.2 only fixes the structure of the call: "fit one FCGS per
DL3DV scene".  Hyperparameters follow FCGS defaults.  The actual model is
injected so unit tests can stub it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class GSScene:
    """Opaque handle on a fitted 3DGS scene.

    `params_path` points to a checkpoint readable by the renderer; the optional
    `train_views` / `train_intrinsics` cached for later trajectory synthesis.
    """
    params_path: Path
    train_views: np.ndarray         # (M, 4, 4) float32, c2w
    train_intrinsics: np.ndarray    # (M, 4)    float32
    scene_stats: dict


def fit_fcgs(
    frames_rgb: np.ndarray,
    train_poses_c2w: np.ndarray,
    train_intrinsics_fxfycxcy: np.ndarray,
    out_dir: Path,
    fit_fn: Optional[Callable] = None,
) -> GSScene:
    """Run FCGS on a scene; `fit_fn(images, poses, intrinsics, out_dir)` is
    injected for tests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fit_fn is None:
        raise NotImplementedError(
            "fit_fn must be injected — wire in FCGS official entry point at deploy time"
        )
    params_path = fit_fn(frames_rgb, train_poses_c2w, train_intrinsics_fxfycxcy, out_dir)
    stats = _compute_scene_stats(train_poses_c2w)
    return GSScene(
        params_path=Path(params_path),
        train_views=train_poses_c2w.astype(np.float32),
        train_intrinsics=train_intrinsics_fxfycxcy.astype(np.float32),
        scene_stats=stats,
    )


def _compute_scene_stats(train_poses_c2w: np.ndarray) -> dict:
    """Centroid, median radius, height range, PCA directions, anchors."""
    centers = train_poses_c2w[:, :3, 3]
    centroid = centers.mean(axis=0)
    radii = np.linalg.norm(centers - centroid, axis=1)
    median_radius = float(np.median(radii)) if len(radii) else 1.0
    heights = centers[:, 1]
    height_range = (float(heights.min()), float(heights.max())) if len(heights) else (0.0, 0.0)
    # PCA directions
    cov = np.cov((centers - centroid).T) if len(centers) > 1 else np.eye(3)
    _, _, vt = np.linalg.svd(cov)
    return {
        "centroid": centroid.astype(np.float32),
        "median_radius": median_radius,
        "height_range": height_range,
        "pca_dirs": vt.astype(np.float32),
        "training_cam_positions": centers.astype(np.float32),
        "training_cam_orientations": train_poses_c2w[:, :3, :3].astype(np.float32),
    }

"""GT-pose pose-annotation mode (paper App. B.1).

Targets: Sekai-Game, DL3DV.
Pipeline: trust the GT camera trajectory; run Pi3X for structure;
Umeyama-Sim(3) with 80%-inlier filter aligns Pi3X scene scale to GT.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from ._common import PoseArtifact
from .umeyama import DEFAULT_INLIER_PERCENTILE, umeyama_sim3_inlier_filter


def run_gtpose(
    clip_path: Path,
    gt_poses_path: Path,
    work_dir: Path,
    pi3x_cmd: Sequence[str] = ("python", "-m", "pi3x.infer"),
    inlier_percentile: float = DEFAULT_INLIER_PERCENTILE,
) -> PoseArtifact:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cams_json = work_dir / "cams_pi3x.json"
    pts_npy = work_dir / "pts_pi3x.npy"

    cmd = [
        *pi3x_cmd,
        "--video", str(clip_path),
        "--emit-points", str(pts_npy),
        "--emit-cams", str(cams_json),
    ]
    subprocess.check_call(cmd)

    poses_gt = np.load(gt_poses_path).astype(np.float32)
    cams_pi3x = json.loads(cams_json.read_text())
    return _build_artifact(poses_gt, cams_pi3x, inlier_percentile)


def _build_artifact(
    poses_gt: np.ndarray,
    cams_pi3x: dict,
    inlier_percentile: float,
) -> PoseArtifact:
    if poses_gt.ndim != 3 or poses_gt.shape[1:] != (4, 4):
        raise ValueError(f"poses_gt must be (T,4,4), got {poses_gt.shape}")
    frames = cams_pi3x["frames"]
    if len(frames) != len(poses_gt):
        raise ValueError(
            f"Pi3X cam count {len(frames)} != GT pose count {len(poses_gt)}"
        )

    # Pi3X camera centers vs GT centers -> Sim(3) alignment
    centers_pi3x = np.array([c["center"] for c in frames], dtype=np.float64)
    centers_gt = poses_gt[:, :3, 3].astype(np.float64)
    s, _R, _t, _inliers = umeyama_sim3_inlier_filter(
        centers_pi3x, centers_gt, inlier_percentile=inlier_percentile,
    )

    # Intrinsics arrive as a 3x3 K per Pi3X frame.
    K_arr = np.array([c["K"] for c in frames], dtype=np.float32)
    fx = K_arr[:, 0, 0]
    fy = K_arr[:, 1, 1]
    cx = K_arr[:, 0, 2]
    cy = K_arr[:, 1, 2]
    intr_NVD = np.stack([fx, fy, cx, cy], axis=-1)[:, None, :].astype(np.float32)

    scale = np.full(len(poses_gt), float(s), dtype=np.float32)
    return PoseArtifact(
        poses_c2w=poses_gt,
        intrinsics=intr_NVD,
        scale_per_frame=scale,
        depth_downsampled=None,
    )

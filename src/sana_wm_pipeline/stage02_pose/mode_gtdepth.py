"""GT-depth pose-annotation mode (paper App. B.1).

Targets: OmniWorld (synthetic, has perfectly-known depth).
Pipeline: feed GT depth straight into VIPE's SLAM/BA; run MoGe-2 to obtain a
*metric* anchor and fuse it against GT to recover the per-frame scale `s_t`.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from ._common import PoseArtifact
from .depth_fusion import fuse_metric_scale


SAMPLE_GRID = 32


def run_gtdepth(
    clip_path: Path,
    gt_depth_path: Path,
    work_dir: Path,
    vipe_cmd: Sequence[str] = ("python", "-m", "vipe.cli"),
) -> PoseArtifact:
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pose_json = work_dir / "pose.json"
    moge_npy = work_dir / "moge2.npy"

    cmd = [
        *vipe_cmd,
        "--video", str(clip_path),
        "--depth-backend", "gt_depth",
        "--gt-depth-path", str(gt_depth_path),
        "--emit-moge2", str(moge_npy),
        "--per-frame-intrinsics",
        "--out", str(pose_json),
    ]
    subprocess.check_call(cmd)
    return _load_artifact(pose_json, gt_depth_path, moge_npy)


def _load_artifact(pose_json: Path, gt_depth_path: Path, moge_npy: Path) -> PoseArtifact:
    d = json.loads(Path(pose_json).read_text())
    poses = np.asarray(d["poses_c2w"], dtype=np.float32)
    intr = np.asarray(d["intrinsics_per_frame_NVD"], dtype=np.float32)
    d_gt = np.load(gt_depth_path)
    d_moge = np.load(moge_npy)
    if d_gt.shape != d_moge.shape:
        raise ValueError(
            f"GT depth and MoGe-2 depth shape mismatch: {d_gt.shape} vs {d_moge.shape}"
        )
    T, H, W = d_gt.shape
    # Uniform grid sample of paired depths.
    ys = np.linspace(0, H - 1, SAMPLE_GRID).astype(int)
    xs = np.linspace(0, W - 1, SAMPLE_GRID).astype(int)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d_gt_pts = d_gt[:, yy, xx].reshape(T, -1).astype(np.float32)
    d_moge_pts = d_moge[:, yy, xx].reshape(T, -1).astype(np.float32)
    scale = fuse_metric_scale(d_gt_pts, d_moge_pts, momentum=0.99).astype(np.float32)
    depth_ds = (d_gt * scale[:, None, None])[:, ::4, ::4].astype(np.float32)
    return PoseArtifact(
        poses_c2w=poses,
        intrinsics=intr,
        scale_per_frame=scale,
        depth_downsampled=depth_ds,
    )

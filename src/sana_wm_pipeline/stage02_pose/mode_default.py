"""Default pose-annotation mode (paper §4 + App. B.1).

Targets: SpatialVID-HQ, Sekai-Walking-HQ, MiraData.
Pipeline: modified VIPE SLAM front-end → Pi3X + MoGe-2 fused depth →
per-frame intrinsics BA → c2w poses + (N, 1, 4) intrinsics + per-frame scale.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from ._common import PoseArtifact


DEFAULT_VIPE_CMD: Sequence[str] = ("python", "-m", "vipe.cli")


def run_default(
    clip_path: Path,
    work_dir: Path,
    vipe_cmd: Sequence[str] = DEFAULT_VIPE_CMD,
) -> PoseArtifact:
    """Invoke the patched VIPE binary on `clip_path` and parse outputs.

    The patched VIPE writes:
      work_dir/pose.json        — {"poses_c2w": [T,4,4], "intrinsics_per_frame_NVD": [T,1,4]}
      work_dir/depth.npy        — (T, H, W) float32  metric depth
      work_dir/scale.npy        — (T,)       float32  per-frame scale
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    pose_json = work_dir / "pose.json"
    depth_npy = work_dir / "depth.npy"
    scale_npy = work_dir / "scale.npy"

    cmd = [
        *vipe_cmd,
        "--video", str(clip_path),
        "--depth-backend", "pi3x_moge2_fused",   # injected via vipe_patch
        "--per-frame-intrinsics",                 # injected via vipe_patch
        "--bundle-adjust",
        "--out", str(pose_json),
        "--out-depth", str(depth_npy),
        "--out-scale", str(scale_npy),
    ]
    subprocess.check_call(cmd)
    return _load_artifact(pose_json, depth_npy, scale_npy)


def _load_artifact(pose_json: Path, depth_npy: Path, scale_npy: Path) -> PoseArtifact:
    d = json.loads(Path(pose_json).read_text())
    poses = np.asarray(d["poses_c2w"], dtype=np.float32)
    intr = np.asarray(d["intrinsics_per_frame_NVD"], dtype=np.float32)
    scale = np.load(scale_npy).astype(np.float32)
    if Path(depth_npy).exists():
        depth = np.load(depth_npy)
        depth_ds = depth[:, ::4, ::4].astype(np.float32)
    else:
        depth_ds = None
    return PoseArtifact(
        poses_c2w=poses,
        intrinsics=intr,
        scale_per_frame=scale,
        depth_downsampled=depth_ds,
    )

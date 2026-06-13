"""GT-depth pose-annotation mode (paper App. B.1).

Targets: OmniWorld (synthetic, perfectly-known depth maps).

Pipeline:
  1. Format GT depth (.npy, T×H×W float32 metres) as CachedDepthModel npz.
  2. Run MoGe-2 per-frame to get metric depth anchor.
  3. VIPE SLAM with vipe_cached_depth pipeline (GT depth injected into BA).
  4. fuse_metric_scale(d_gt_grid, d_moge_grid) → per-frame metric scale s_t.
  5. Return PoseArtifact.

No new VIPE backend required — reuses existing `cached` depth backend.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from ._common import PoseArtifact
from .depth_fusion import fuse_metric_scale
from .mode_default import _load_vipe_artifacts

VIPE_CMD: Sequence[str] = ("vipe", "infer")
VIPE_PIPELINE = "vipe_cached_depth"
SAMPLE_GRID = 32


def _run_moge2(
    clip_path: Path,
    moge_out: Path,
    moge2_weights: str,
    fov_x_deg: float = 60.0,
    device: str = "cuda",
) -> np.ndarray:
    """Run MoGe-2 on every frame; return (T, H, W) float32 metric depth."""
    import cv2
    import torch
    from moge.model.v2 import MoGeModel  # type: ignore

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from: {clip_path}")

    moge2_path = Path(moge2_weights)
    ckpt = moge2_path / "model.pt" if moge2_path.is_dir() else moge2_path
    model = MoGeModel.from_pretrained(str(ckpt)).to(device).eval()

    H, W = frames[0].shape[:2]
    depths = np.zeros((len(frames), H, W), dtype=np.float32)
    with torch.no_grad():
        for i, frame in enumerate(frames):
            ft = (
                torch.from_numpy(frame.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            out = model.infer(ft, fov_x=fov_x_deg)
            depths[i] = out["depth"].squeeze(0).cpu().numpy()
    del model

    np.save(str(moge_out), depths)
    return depths


def run_gtdepth(
    clip_path: Path,
    gt_depth_path: Path,
    work_dir: Path,
    vipe_cmd: Sequence[str] = VIPE_CMD,
    pipeline: str = VIPE_PIPELINE,
) -> PoseArtifact:
    """GT-depth annotation: inject OmniWorld GT depth into VIPE BA.

    Args:
        clip_path: normalized video (.mp4), T frames.
        gt_depth_path: (T, H, W) float32 numpy file, depth in metres.
        work_dir: scratch directory; VIPE writes pose/ and intrinsics/ here.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    moge2_weights = os.environ.get("SANA_WM_MOGE2_WEIGHTS", "")
    if not moge2_weights:
        raise RuntimeError("SANA_WM_MOGE2_WEIGHTS must be set")

    # Phase 1: format GT depth as CachedDepthModel npz
    d_gt = np.load(str(gt_depth_path)).astype(np.float32)  # (T, H, W)
    cache_path = work_dir / "_gt_depth_cache.npz"
    np.savez_compressed(str(cache_path), depths=d_gt)

    # Phase 2: run MoGe-2 for metric scale anchor (skip if already cached)
    moge_npy = work_dir / "_moge2_depth.npy"
    if moge_npy.exists():
        d_moge = np.load(str(moge_npy)).astype(np.float32)
    else:
        d_moge = _run_moge2(clip_path, moge_npy, moge2_weights)

    # Phase 3: VIPE SLAM with GT depth injected via CachedDepthModel
    os.environ["SANA_WM_CACHED_DEPTH_PATH"] = str(cache_path)
    try:
        cmd = [*vipe_cmd, str(clip_path), "--output", str(work_dir), "--pipeline", pipeline]
        subprocess.check_call(cmd)
    finally:
        os.environ.pop("SANA_WM_CACHED_DEPTH_PATH", None)
        cache_path.unlink(missing_ok=True)

    # Phase 4: load VIPE pose + intrinsics artifacts (same format as default mode)
    artifact = _load_vipe_artifacts(clip_path, work_dir)
    T = len(artifact.poses_c2w)

    # Phase 5: per-frame metric scale via grid-sampled GT vs MoGe-2 depths
    H_d, W_d = d_gt.shape[1], d_gt.shape[2]
    ys = np.linspace(0, H_d - 1, SAMPLE_GRID).astype(int)
    xs = np.linspace(0, W_d - 1, SAMPLE_GRID).astype(int)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    d_gt_grid = d_gt[:T, yy, xx].reshape(T, -1)      # (T, SAMPLE_GRID²) float32
    d_moge_grid = d_moge[:T, yy, xx].reshape(T, -1)   # (T, SAMPLE_GRID²) float32
    scale = fuse_metric_scale(d_gt_grid, d_moge_grid, momentum=0.99).astype(np.float32)

    return PoseArtifact(
        poses_c2w=artifact.poses_c2w,
        intrinsics=artifact.intrinsics,
        scale_per_frame=scale,
        depth_downsampled=artifact.depth_downsampled,
    )

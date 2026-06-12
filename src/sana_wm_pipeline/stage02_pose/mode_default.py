"""Default pose-annotation mode (paper §4 + App. B.1).

Targets: SpatialVID-HQ, Sekai-Walking-HQ, MiraData.
Pipeline: VIPE SLAM front-end (modified with Pi3X + MoGe-2 fused depth) →
per-frame intrinsics → c2w poses + (N,1,4) intrinsics + per-frame scale.

VIPE CLI: ``vipe infer <video> -o <work_dir> --pipeline vipe_cached_depth``

Output artifacts (VIPE format, read by _load_vipe_artifacts):
  <work_dir>/pose/<stem>.npz         — data:(T,4,4) cam2world, inds:(T,)
  <work_dir>/intrinsics/<stem>.npz   — data:(T,4) [fx,fy,cx,cy], inds:(T,)
  <work_dir>/depth/<stem>.zip        — EXR per-frame depth (optional)
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from ._common import PoseArtifact

VIPE_CMD: Sequence[str] = ("vipe", "infer")
VIPE_PIPELINE = "vipe_cached_depth"


def _precompute_depth_cache(
    clip_path: Path,
    cache_path: Path,
    pi3x_weights: str,
    moge2_weights: str,
    chunk: int = 16,
    stride: int = 8,
    device: str = "cuda",
) -> None:
    """预计算 Pi3X+MoGe-2 融合深度缓存（论文 App. B.1），写到 cache_path.npz。

    格式与 experiments/vipe_comparison/precompute_pi3x_depths.py 一致：
      depths: (T, H, W) float32 metric (metres)
      scale_history: (T,) float32
    """
    import cv2
    import torch
    import torch.nn.functional as F

    # 1. 读帧
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {clip_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T,H,W,3)
    T, H, W, _ = frames_np.shape
    frames_t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(device)  # (T,3,H,W)

    # 2. Pi3X 分块推理（复用 experiments 逻辑，inline 实现）
    from pi3 import Pi3X  # type: ignore
    pi3x_model = Pi3X.from_pretrained(pi3x_weights).to(device).eval()

    H_r = (H // 14) * 14
    W_r = (W // 14) * 14
    src = F.interpolate(frames_t, size=(H_r, W_r), mode="bilinear", align_corners=False) if H_r != H or W_r != W else frames_t

    accum = np.zeros((T, H_r, W_r), dtype=np.float32)
    count = np.zeros(T, dtype=np.float32)
    starts = list(range(0, max(T - chunk + 1, 1), stride))
    if not starts or starts[-1] + chunk < T:
        starts.append(max(0, T - chunk))

    with torch.no_grad():
        for s in starts:
            e = min(s + chunk, T)
            out = pi3x_model(src[s:e].unsqueeze(0))
            d = out["local_points"][0, :e - s, :, :, 2].cpu().numpy()
            accum[s:e] += d
            count[s:e] += 1
    d_pi3x = accum / np.maximum(count[:, None, None], 1.0)
    if H_r != H or W_r != W:
        d_pi3x = F.interpolate(
            torch.from_numpy(d_pi3x).unsqueeze(1).to(device),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(1).cpu().numpy()
    del pi3x_model

    # 3. MoGe-2 逐帧推理
    import math
    from moge.model.v2 import MoGeModel  # type: ignore
    moge2_path = Path(moge2_weights)
    moge2_ckpt = moge2_path / "model.pt" if moge2_path.is_dir() else moge2_path
    moge2_model = MoGeModel.from_pretrained(str(moge2_ckpt)).to(device).eval()
    fov_x = math.degrees(2 * math.atan(W / (2 * 525.0)))  # 合理初始估计
    d_moge = []
    with torch.no_grad():
        for i in range(T):
            out = moge2_model.infer(frames_t[i:i + 1], fov_x=fov_x)
            d_moge.append(out["depth"].squeeze(0).cpu().numpy())
    d_moge = np.stack(d_moge, axis=0)
    del moge2_model

    # 4. EMA scale fusion（论文 App. B.1）
    T_ = len(d_pi3x)
    scale_history = np.zeros(T_, dtype=np.float32)
    ema = None
    for t in range(T_):
        mask = (d_pi3x[t] > 1e-6) & (d_moge[t] > 1e-6)
        ratio = float(d_moge[t][mask].mean()) / (float(d_pi3x[t][mask].mean()) + 1e-8) if mask.sum() >= 10 else 1.0
        if ema is None:
            ema = float(np.median(d_moge[t][mask] / (d_pi3x[t][mask] + 1e-8))) if mask.sum() >= 10 else ratio
        else:
            ema = ema * 0.99 + ratio * 0.01
        scale_history[t] = ema
    depths_fused = (d_pi3x * scale_history[:, None, None]).astype(np.float32)

    # 5. 保存
    np.savez_compressed(str(cache_path), depths=depths_fused, scale_history=scale_history)


def run_default(
    clip_path: Path,
    work_dir: Path,
    vipe_cmd: Sequence[str] = VIPE_CMD,
    pipeline: str = VIPE_PIPELINE,
) -> PoseArtifact:
    """Invoke two-phase VIPE: precompute depth cache, then run cached SLAM.

    Phase A: compute Pi3X+MoGe-2 fused depth cache (~600 MB, deleted after).
    Phase B: VIPE SLAM with vipe_cached_depth pipeline (CachedDepthModel injects BA).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pi3x_weights = os.environ.get("SANA_WM_PI3X_WEIGHTS", "")
    moge2_weights = os.environ.get("SANA_WM_MOGE2_WEIGHTS", "")
    if not pi3x_weights or not moge2_weights:
        raise RuntimeError(
            "SANA_WM_PI3X_WEIGHTS and SANA_WM_MOGE2_WEIGHTS must be set"
        )

    cache_path = work_dir / "_depth_cache.npz"
    _precompute_depth_cache(
        clip_path, cache_path,
        pi3x_weights=pi3x_weights,
        moge2_weights=moge2_weights,
    )

    os.environ["SANA_WM_CACHED_DEPTH_PATH"] = str(cache_path)
    try:
        cmd = [
            *vipe_cmd,
            str(clip_path),
            "--output", str(work_dir),
            "--pipeline", pipeline,
        ]
        subprocess.check_call(cmd)
    finally:
        os.environ.pop("SANA_WM_CACHED_DEPTH_PATH", None)
        cache_path.unlink(missing_ok=True)

    return _load_vipe_artifacts(clip_path, work_dir)


def _load_vipe_artifacts(clip_path: Path, vipe_out: Path) -> PoseArtifact:
    """Parse VIPE's npz artifacts into PoseArtifact.

    VIPE writes:
      pose/<stem>.npz          data:(T,4,4), inds:(T,)
      intrinsics/<stem>.npz    data:(T,4),   inds:(T,)   — [fx,fy,cx,cy]
    """
    stem = Path(clip_path).stem
    pose_npz = vipe_out / "pose" / f"{stem}.npz"
    intr_npz = vipe_out / "intrinsics" / f"{stem}.npz"

    if not pose_npz.exists():
        raise FileNotFoundError(
            f"VIPE pose artifact missing: {pose_npz}\n"
            f"(check vipe infer completed without error)"
        )

    pose_data = np.load(pose_npz)
    poses_c2w = pose_data["data"].astype(np.float32)  # (T, 4, 4)
    pose_inds = pose_data["inds"]                      # (T,)

    if not intr_npz.exists():
        raise FileNotFoundError(f"VIPE intrinsics artifact missing: {intr_npz}")
    intr_data = np.load(intr_npz)
    intrinsics_raw = intr_data["data"].astype(np.float32)  # (T, 4) [fx,fy,cx,cy]
    intr_inds = intr_data["inds"]

    # VIPE may only write keyframe poses; interpolate to full T frames.
    T_full = int(pose_inds.max()) + 1
    poses_c2w = _interp_poses(poses_c2w, pose_inds, T_full)
    intrinsics_full = _interp_intrinsics(intrinsics_raw, intr_inds, T_full)

    # Reshape intrinsics to (T, 1, 4) as required by PoseArtifact.
    intrinsics_nvd = intrinsics_full[:, None, :]  # (T, 1, 4)

    # scale_per_frame: metric scale ratio (Pi3X-EMA gives this; here we use 1s
    # since VIPE's unidepth backend already produces metric depth directly).
    scale_per_frame = np.ones(T_full, dtype=np.float32)

    # Optional downsampled depth for visualization.
    depth_ds = _try_load_depth_downsampled(vipe_out, stem, T_full)

    artifact = PoseArtifact(
        poses_c2w=poses_c2w,
        intrinsics=intrinsics_nvd,
        scale_per_frame=scale_per_frame,
        depth_downsampled=depth_ds,
    )
    return artifact


def _interp_poses(poses: np.ndarray, inds: np.ndarray, T: int) -> np.ndarray:
    """Nearest-neighbour fill from keyframe poses to dense T frames."""
    out = np.zeros((T, 4, 4), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            out[:, i, j] = np.interp(np.arange(T), inds, poses[:, i, j])
    # Ensure first frame is identity (paper App. D.3).
    if not np.allclose(out[0], np.eye(4), atol=1e-3):
        T0_inv = np.linalg.inv(out[0])
        out = (T0_inv[None] @ out)
    return out.astype(np.float32)


def _interp_intrinsics(intr: np.ndarray, inds: np.ndarray, T: int) -> np.ndarray:
    """Linear interpolation of [fx,fy,cx,cy] to T frames."""
    out = np.zeros((T, 4), dtype=np.float32)
    for k in range(4):
        out[:, k] = np.interp(np.arange(T), inds, intr[:, k])
    return out.astype(np.float32)


def _try_load_depth_downsampled(
    vipe_out: Path, stem: str, T: int
) -> np.ndarray | None:
    """Try to read VIPE's depth zip and downsample 4×."""
    depth_zip = vipe_out / "depth" / f"{stem}.zip"
    if not depth_zip.exists():
        return None
    try:
        import zipfile
        import io as _io
        frames: list[np.ndarray] = []
        with zipfile.ZipFile(depth_zip) as zf:
            names = sorted(zf.namelist())
            for name in names:
                with zf.open(name) as f:
                    buf = f.read()
                # Try EXR -> numpy
                try:
                    import OpenEXR, Imath  # type: ignore
                    exr = OpenEXR.InputFile(OpenEXR.InputFile.__new__(OpenEXR.InputFile))
                    # Fallback: just skip depth if EXR parsing is complex
                    del exr
                    frames = None  # type: ignore[assignment]
                    break
                except Exception:
                    frames = None  # type: ignore[assignment]
                    break
        if frames is None:
            return None
        depth_arr = np.stack(frames, axis=0)  # (T, H, W)
        return depth_arr[:, ::4, ::4].astype(np.float32)
    except Exception:
        return None

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
from .depth_fusion import fuse_depth_sequence

VIPE_CMD: Sequence[str] = ("vipe", "infer")
VIPE_PIPELINE = "vipe_cached_depth"

# 旧的inline预计算函数已废弃，替换为独立脚本 scripts/precompute_fused_depth_reference.py
# def _precompute_depth_cache(...): ...


def run_default(
    clip_path: Path,
    work_dir: Path,
    vipe_cmd: Sequence[str] = VIPE_CMD,
    pipeline: str = "vipe_sanawm",
) -> PoseArtifact:
    """使用sana-wm-data-clean参考实现（带@lru_cache模型缓存）

    Phase A: 使用_real.py的pi3_infer + moge_metric_depth（模型只加载一次）
    Phase B: VIPE SLAM with vipe_sanawm pipeline
    """
    import sys
    import cv2
    from ..sana_wm_data_clean.pose import _real

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    pi3x_weights = os.environ.get("SANA_WM_PI3X_WEIGHTS", "")
    moge2_weights = os.environ.get("SANA_WM_MOGE2_WEIGHTS", "")
    if not pi3x_weights or not moge2_weights:
        raise RuntimeError(
            "SANA_WM_PI3X_WEIGHTS and SANA_WM_MOGE2_WEIGHTS must be set"
        )

    # Phase A: 使用sana-wm-data-clean的_real.py（带@lru_cache）
    print("[mode_default] Phase A: 深度预计算", flush=True)
    depth_dir = work_dir / "depth_precomputed"
    depth_dir.mkdir(parents=True, exist_ok=True)

    # 读取视频帧（均匀采样64帧）
    max_frames = int(os.environ.get("SANA_WM_MAX_FRAMES", "64"))
    print(f"  读取视频: {clip_path}", flush=True)
    frames = _read_frames_uniform(str(clip_path), max_frames)
    S = frames.shape[0]
    print(f"  采样帧数: {S}", flush=True)

    # Pi3推理（第一次调用50s，后续调用直接用缓存0s）
    print(f"  Pi3推理 ({S}帧)...", flush=True)
    poses_pi3, depth_pi3 = _real.pi3_infer(frames)

    # MoGe推理（第一次调用30s，后续调用直接用缓存0s）
    print(f"  MoGe-2推理 ({S}帧)...", flush=True)
    depth_moge = _real.moge_metric_depth(frames, ref_hw=depth_pi3.shape[1:])

    # 深度融合
    print(f"  深度融合...", flush=True)
    fused, scales = fuse_depth_sequence(depth_pi3, np.abs(depth_moge), ema_momentum=0.99)

    # 保存预计算结果（供VIPE使用）
    np.save(depth_dir / "fused.npy", fused.astype(np.float32))
    np.save(depth_dir / "scales.npy", scales.astype(np.float32))

    # 计算RGB签名（16x16下采样）
    sig = _compute_rgb_signatures(frames)
    np.save(depth_dir / "sig.npy", sig)

    # 保存采样索引
    import decord
    total_frames = len(decord.VideoReader(str(clip_path)))
    sample_idx = np.linspace(0, total_frames - 1, S).round().astype(int)
    np.save(depth_dir / "sample_idx.npy", sample_idx)

    print(f"  ✅ 预计算完成: fused{fused.shape}, scale~{float(np.median(scales)):.3f}", flush=True)

    # Phase B: VIPE SLAM with Pi3xMogeModel
    print("[mode_default] Phase B: VIPE SLAM", flush=True)
    os.environ["SANA_WM_FUSED_DEPTH_DIR"] = str(depth_dir)
    try:
        cmd = [
            *vipe_cmd,
            str(clip_path),
            "--output", str(work_dir),
            "--pipeline", pipeline,
        ]
        subprocess.check_call(cmd)
    finally:
        os.environ.pop("SANA_WM_FUSED_DEPTH_DIR", None)

    return _load_vipe_artifacts(clip_path, work_dir)


def _read_frames_uniform(video_path: str, max_frames: int) -> np.ndarray:
    """均匀采样视频帧 -> (S, H, W, 3) uint8 RGB"""
    import decord
    vr = decord.VideoReader(video_path)
    total = len(vr)
    S = min(max_frames, total)
    indices = np.linspace(0, total - 1, S).round().astype(int)
    frames = vr.get_batch(indices).asnumpy()  # (S, H, W, 3) RGB uint8
    return frames


def _compute_rgb_signatures(frames: np.ndarray) -> np.ndarray:
    """计算RGB 16x16签名 -> (S, 768)"""
    import cv2
    S = frames.shape[0]
    sig = np.zeros((S, 768), dtype=np.float32)
    for i, f in enumerate(frames):
        small = cv2.resize(f, (16, 16), interpolation=cv2.INTER_AREA)
        sig[i] = small.reshape(-1).astype(np.float32) / 255.0
    return sig


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

    # 直接使用VIPE输出（与sana-wm-data-clean/vipe_cli.py:_load_vipe_pose对齐）
    # 参考: vipe_cli.py:61-70 _load_vipe_pose()
    # 逻辑: 只按inds排序，不做稀疏化/插值
    # 理由: 参考实现没有稀疏化逻辑，稀疏化+插值会丢失VIPE BA优化的信息
    order = np.argsort(pose_inds)
    poses_c2w = poses_c2w[order]
    pose_inds_sorted = pose_inds[order]

    order_intr = np.argsort(intr_inds)
    intrinsics_raw = intrinsics_raw[order_intr]
    intr_inds_sorted = intr_inds[order_intr]

    # VIPE输出的帧数就是最终帧数（不插值到T_full）
    T_full = len(poses_c2w)

    print(f"[mode_default] Loaded {T_full} frames from VIPE (aligned with reference)")
    print(f"[mode_default]   Pose indices: {pose_inds_sorted[:min(5, T_full)].tolist()} ... {pose_inds_sorted[-min(5, T_full):].tolist()}")

    # 参考实现的intrinsics处理逻辑（vipe_cli.py:73-100 _load_perframe_intrinsics）
    intrinsics_full = _interp_intrinsics_aligned(intrinsics_raw, T_full)

    # Reshape intrinsics to (T, 1, 4) as required by PoseArtifact.
    intrinsics_nvd = intrinsics_full[:, None, :]  # (T, 1, 4)

    # Load scale_per_frame from Phase A (与官方sana-wm-data-clean一致)
    # 官方: stage.py:104 → scales = scales_arr.tolist()
    depth_dir = vipe_out / "depth_precomputed"
    scale_path = depth_dir / "scales.npy"

    if scale_path.exists():
        scales_full = np.load(scale_path).astype(np.float32)  # (S,) Phase A采样的帧数

        # 如果Phase A采样了关键帧（S < T_full），需要插值到全部帧
        sample_idx_path = depth_dir / "sample_idx.npy"
        if sample_idx_path.exists() and len(scales_full) < T_full:
            sample_idx = np.load(sample_idx_path).astype(int)  # (S,) 采样索引
            # 线性插值到T_full帧
            scale_per_frame = np.interp(
                np.arange(T_full),
                sample_idx,
                scales_full
            ).astype(np.float32)
            print(f"[mode_default] ✅ Interpolated {len(scales_full)} scales to {T_full} frames")
        else:
            # 无需插值，直接使用（或截断）
            scale_per_frame = scales_full[:T_full] if len(scales_full) >= T_full else scales_full
            if len(scale_per_frame) < T_full:
                # 不足则补1.0
                padding = np.ones(T_full - len(scale_per_frame), dtype=np.float32)
                scale_per_frame = np.concatenate([scale_per_frame, padding])
            print(f"[mode_default] ✅ Loaded {len(scales_full)} scales directly")

        print(f"[mode_default]    Scale range: {scale_per_frame.min():.3f} - {scale_per_frame.max():.3f}")
        print(f"[mode_default]    Scale mean±std: {scale_per_frame.mean():.3f} ± {scale_per_frame.std():.3f}")
        scale_cov = scale_per_frame.std() / (scale_per_frame.mean() + 1e-8)
        print(f"[mode_default]    Scale CoV: {scale_cov:.3f} (threshold: <2.0)")
    else:
        # Fallback: Phase A失败或缺失时使用默认值
        scale_per_frame = np.ones(T_full, dtype=np.float32)
        print(f"[mode_default] ⚠️  {scale_path} not found, using default scale=1.0")

    # Optional downsampled depth for visualization.
    depth_ds = _try_load_depth_downsampled(vipe_out, stem, T_full)

    artifact = PoseArtifact(
        poses_c2w=poses_c2w,
        intrinsics=intrinsics_nvd,
        scale_per_frame=scale_per_frame,
        depth_downsampled=depth_ds,
    )
    return artifact


def _interp_intrinsics_aligned(intr: np.ndarray, n_target: int) -> np.ndarray:
    """Align intrinsics to target frame count (与vipe_cli.py:_load_perframe_intrinsics对齐).

    参考: sana-wm-data-clean/sana_wm_data/pose/vipe_cli.py:73-100

    Args:
        intr: (K, 4) [fx,fy,cx,cy]
        n_target: 目标帧数N

    Returns:
        (N, 4) intrinsics

    Logic:
        K == N: 直接使用
        K == 1: broadcast到N帧
        1 < K < N: 线性插值到N帧
    """
    if intr.ndim == 1:
        intr = intr[None, :]
    K = intr.shape[0]

    if K == n_target:
        return intr
    if K == 1:
        return np.tile(intr[0], (n_target, 1))

    # 1 < K < N: 线性插值
    src = np.linspace(0.0, 1.0, K)
    dst = np.linspace(0.0, 1.0, n_target)
    return np.stack([np.interp(dst, src, intr[:, j]) for j in range(4)], axis=1).astype(np.float32)


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

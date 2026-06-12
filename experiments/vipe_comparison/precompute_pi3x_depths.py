#!/usr/bin/env python3
"""
预计算 Pi3X + MoGe-2 融合深度，保存为 .npz 缓存供 CachedDepthModel 使用。

论文公式 (App. B.1):
  per-frame scale s_i = Σ_j w_j·d^MoGe_j / Σ_j w_j·d^Pi3X_j
                       w_j = 1/d^Pi3X_j  (inverse-depth weighting)
                     ≡ mean(d^MoGe) / mean(d^Pi3X)  on valid pixels
  EMA: s_ema_t = s_ema_{t-1} * 0.99 + s_t * 0.01
  depth_fused_i = s_ema_i · d^Pi3X_i

输出: {out_dir}/cache_pi3x_moge2_{seq_name}.npz
  depths:        (T, H, W) float32, metres
  scale_history: (T,)      float32

用法:
  python experiments/vipe_comparison/precompute_pi3x_depths.py \
    --video experiments/vipe_comparison/data/rgbd_dataset_freiburg1_desk/video.mp4 \
    --out   experiments/vipe_comparison/results/cache_pi3x_moge2_fr1_desk.npz
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import pathlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def read_video_frames(video_path: str) -> np.ndarray:
    """返回 (T, H, W, 3) uint8 BGR→RGB 帧数组。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    log.info(f"Read {len(frames)} frames from {video_path}")
    return np.stack(frames, axis=0)  # (T, H, W, 3) uint8


@torch.no_grad()
def run_pi3x(
    model,
    frames_t: torch.Tensor,   # (T, 3, H, W) float32 [0,1] on GPU
    chunk: int = 16,
    stride: int = 8,
) -> np.ndarray:
    """返回 (T, H, W) 相对深度。"""
    T, _, H, W = frames_t.shape
    H_r = (H // 14) * 14
    W_r = (W // 14) * 14
    if H_r != H or W_r != W:
        src = F.interpolate(frames_t, size=(H_r, W_r), mode="bilinear", align_corners=False)
    else:
        src = frames_t

    accum = np.zeros((T, H_r, W_r), dtype=np.float32)
    count = np.zeros(T, dtype=np.float32)

    starts = list(range(0, max(T - chunk + 1, 1), stride))
    if not starts or starts[-1] + chunk < T:
        starts.append(max(0, T - chunk))

    for i, s in enumerate(starts):
        e = min(s + chunk, T)
        log.info(f"  Pi3X chunk {i+1}/{len(starts)}: frames [{s}, {e})")
        out = model(src[s:e].unsqueeze(0))  # (1, N, H_r, W_r, 3)
        d = out["local_points"][0, :e - s, :, :, 2].cpu().numpy()
        accum[s:e] += d
        count[s:e] += 1

    d_r = accum / np.maximum(count[:, None, None], 1.0)

    if H_r != H or W_r != W:
        d_r = F.interpolate(
            torch.from_numpy(d_r).unsqueeze(1).cuda(),
            size=(H, W), mode="bilinear", align_corners=False,
        ).squeeze(1).cpu().numpy()
    return d_r  # (T, H, W)


@torch.no_grad()
def run_moge2(
    model,
    frames_t: torch.Tensor,   # (T, 3, H, W) float32 [0,1] on GPU
    fov_x: float | None = None,
) -> np.ndarray:
    """返回 (T, H, W) metric depth (metres)。"""
    results = []
    T = len(frames_t)
    for i in range(T):
        if i % 100 == 0:
            log.info(f"  MoGe-2 frame {i}/{T}")
        out = model.infer(frames_t[i:i + 1], fov_x=fov_x)
        results.append(out["depth"].squeeze(0).cpu().numpy())
    return np.stack(results, axis=0)  # (T, H, W)


def ema_fuse(d_pi3x: np.ndarray, d_moge: np.ndarray, momentum: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    """
    论文 App. B.1 EMA scale fusion。
    返回 fused metric depth (T,H,W) 和 scale_history (T,)。
    """
    T = len(d_pi3x)
    scale_history = np.zeros(T, dtype=np.float32)
    ema: float | None = None

    for t in range(T):
        mask = (d_pi3x[t] > 1e-6) & (d_moge[t] > 1e-6)
        if mask.sum() < 10:
            ratio = 1.0
        else:
            # s_t = mean(d_moge) / mean(d_pi3x)  ≡ WLS with w=1/d_pi3x
            ratio = float(d_moge[t][mask].mean()) / (float(d_pi3x[t][mask].mean()) + 1e-8)

        if ema is None:
            # 第一帧用 median 比值初始化，避免异常值影响 EMA 起点
            if mask.sum() >= 10:
                ema = float(np.median((d_moge[t][mask] / (d_pi3x[t][mask] + 1e-8))))
            else:
                ema = ratio
        else:
            ema = ema * momentum + ratio * (1.0 - momentum)

        scale_history[t] = ema

    fused = (d_pi3x * scale_history[:, None, None]).astype(np.float32)
    return fused, scale_history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="input video.mp4 path")
    parser.add_argument("--out", required=True, help="output .npz path")
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pi3x_weights = os.environ.get("SANA_WM_PI3X_WEIGHTS")
    moge2_weights = os.environ.get("SANA_WM_MOGE2_WEIGHTS")
    if not pi3x_weights or not moge2_weights:
        raise RuntimeError("Set SANA_WM_PI3X_WEIGHTS and SANA_WM_MOGE2_WEIGHTS")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. 读帧 ──────────────────────────────────────────────────────────────
    frames_np = read_video_frames(args.video)  # (T, H, W, 3) uint8
    T, H, W, _ = frames_np.shape
    frames_f32 = frames_np.astype(np.float32) / 255.0  # (T, H, W, 3) [0,1]
    frames_t = torch.from_numpy(frames_f32).permute(0, 3, 1, 2).to(args.device)  # (T,3,H,W)

    # fov_x for MoGe-2 (TUM fr1/fr2 fx≈525 for 640px wide)
    fx_tum = 525.0
    fov_x = math.degrees(2 * math.atan(W / (2 * fx_tum)))
    log.info(f"Using fov_x={fov_x:.2f}° (W={W}, fx={fx_tum})")

    # ── 2. 加载模型 ───────────────────────────────────────────────────────────
    log.info("Loading Pi3X...")
    from pi3 import Pi3X  # type: ignore
    pi3x_model = Pi3X.from_pretrained(pi3x_weights).to(args.device).eval()

    log.info("Loading MoGe-2...")
    from moge.model.v2 import MoGeModel  # type: ignore
    moge2_path = pathlib.Path(moge2_weights)
    moge2_ckpt = moge2_path / "model.pt" if moge2_path.is_dir() else moge2_path
    moge2_model = MoGeModel.from_pretrained(str(moge2_ckpt)).to(args.device).eval()

    # ── 3. Pi3X 分块推理 ─────────────────────────────────────────────────────
    log.info(f"Running Pi3X on {T} frames (chunk={args.chunk}, stride={args.stride})...")
    d_pi3x = run_pi3x(pi3x_model, frames_t, chunk=args.chunk, stride=args.stride)
    log.info(f"Pi3X done. depth range: {d_pi3x.min():.3f}~{d_pi3x.max():.3f}")

    # ── 4. MoGe-2 逐帧推理 ───────────────────────────────────────────────────
    log.info(f"Running MoGe-2 on {T} frames...")
    d_moge = run_moge2(moge2_model, frames_t, fov_x=fov_x)
    log.info(f"MoGe-2 done. depth range: {d_moge.min():.3f}~{d_moge.max():.3f}")

    # ── 5. EMA scale fusion ──────────────────────────────────────────────────
    log.info(f"EMA scale fusion (momentum={args.ema_momentum})...")
    depths_fused, scale_history = ema_fuse(d_pi3x, d_moge, momentum=args.ema_momentum)
    log.info(f"Scale range: {scale_history.min():.4f}~{scale_history.max():.4f}")
    log.info(f"Fused depth range: {depths_fused.min():.3f}~{depths_fused.max():.3f}")

    # ── 6. 保存 ──────────────────────────────────────────────────────────────
    np.savez_compressed(out_path, depths=depths_fused, scale_history=scale_history)
    log.info(f"Saved cache ({T} frames, {out_path.stat().st_size / 1e6:.1f} MB) → {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""预计算 Pi3X + MoGe-2 融合深度（独立脚本，对齐 sana-wm-data-clean）

使用方式:
    python scripts/precompute_fused_depth_reference.py <video.mp4> <out_dir>

输出:
    <out_dir>/fused.npy       (S, h, w) 融合深度
    <out_dir>/sig.npy         (S, 768) RGB 16x16签名
    <out_dir>/scales.npy      (S,) 逐帧尺度因子
    <out_dir>/sample_idx.npy  (S,) 采样索引

环境变量:
    SANA_WM_PI3X_WEIGHTS  - Pi3X模型权重路径
    SANA_WM_MOGE2_WEIGHTS - MoGe-2模型权重路径
    SANA_WM_MAX_FRAMES    - 最大采样帧数（默认64）
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sana_wm_pipeline.stage02_pose.depth_fusion import fuse_depth_sequence


def read_frames_uniform(video_path: str, max_frames: int) -> np.ndarray:
    """均匀采样视频帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames)).round().astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")

    return np.stack(frames, axis=0).astype(np.float32) / 255.0  # (S, H, W, 3)


def pi3x_infer(frames: np.ndarray, weights_path: str, device: str = "cuda") -> np.ndarray:
    """Pi3X推理（分块处理）"""
    from pi3 import Pi3X

    model = Pi3X.from_pretrained(weights_path).to(device).eval()

    S, H, W, _ = frames.shape
    H_r = (H // 14) * 14
    W_r = (W // 14) * 14

    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)
    if H_r != H or W_r != W:
        frames_t = F.interpolate(frames_t, size=(H_r, W_r), mode="bilinear", align_corners=False)

    chunk = min(16, S)
    stride = max(8, chunk // 2)

    accum = np.zeros((S, H_r, W_r), dtype=np.float32)
    count = np.zeros(S, dtype=np.float32)

    starts = list(range(0, max(S - chunk + 1, 1), stride))
    if not starts or starts[-1] + chunk < S:
        starts.append(max(0, S - chunk))

    with torch.no_grad():
        for s in starts:
            e = min(s + chunk, S)
            out = model(frames_t[s:e].unsqueeze(0))
            d = out["local_points"][0, :e - s, :, :, 2].cpu().numpy()
            accum[s:e] += d
            count[s:e] += 1

    d_pi3x = accum / np.maximum(count[:, None, None], 1.0)

    if H_r != H or W_r != W:
        d_pi3x = F.interpolate(
            torch.from_numpy(d_pi3x).unsqueeze(1).to(device),
            size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(1).cpu().numpy()

    del model
    torch.cuda.empty_cache()

    return d_pi3x  # (S, H, W)


def moge2_infer(frames: np.ndarray, weights_path: str, device: str = "cuda") -> np.ndarray:
    """MoGe-2推理（逐帧）"""
    import math
    from moge.model.v2 import MoGeModel

    ckpt_path = Path(weights_path) / "model.pt" if Path(weights_path).is_dir() else Path(weights_path)
    model = MoGeModel.from_pretrained(str(ckpt_path)).to(device).eval()

    S, H, W, _ = frames.shape
    fov_x = math.degrees(2 * math.atan(W / (2 * 525.0)))

    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device)

    depths = []
    with torch.no_grad():
        for i in range(S):
            out = model.infer(frames_t[i:i + 1], fov_x=fov_x)
            depths.append(out["depth"].squeeze(0).cpu().numpy())

    del model
    torch.cuda.empty_cache()

    return np.stack(depths, axis=0)  # (S, H, W)


def compute_rgb_signatures(frames: np.ndarray) -> np.ndarray:
    """计算RGB 16x16签名用于帧匹配"""
    sigs = []
    for frame in frames:
        frame_uint8 = (frame * 255).astype(np.uint8)
        resized = cv2.resize(frame_uint8, (16, 16))
        sigs.append(resized.astype(np.float32).ravel())
    return np.stack(sigs, axis=0)  # (S, 768)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    video_path = sys.argv[1]
    out_dir = Path(sys.argv[2])
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("SANA_WM_MAX_FRAMES", "64"))

    pi3x_weights = os.environ.get("SANA_WM_PI3X_WEIGHTS")
    moge2_weights = os.environ.get("SANA_WM_MOGE2_WEIGHTS")

    if not pi3x_weights or not moge2_weights:
        raise RuntimeError("SANA_WM_PI3X_WEIGHTS and SANA_WM_MOGE2_WEIGHTS must be set")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 读取视频帧...")
    frames = read_frames_uniform(video_path, max_frames)
    S = frames.shape[0]

    # 计算采样索引
    import decord
    total_frames = len(decord.VideoReader(video_path))
    sample_idx = np.linspace(0, total_frames - 1, S).round().astype(int)

    print(f"[2/5] Pi3X推理 ({S}帧)...")
    d_pi3x = pi3x_infer(frames, pi3x_weights)

    print(f"[3/5] MoGe-2推理 ({S}帧)...")
    d_moge = moge2_infer(frames, moge2_weights)

    print(f"[4/5] 深度融合 (加权最小二乘 + EMA)...")
    fused, scales = fuse_depth_sequence(d_pi3x, np.abs(d_moge), ema_momentum=0.99)

    print(f"[5/5] 计算RGB签名...")
    sig = compute_rgb_signatures(frames)

    # 保存
    np.save(out_dir / "fused.npy", fused.astype(np.float32))
    np.save(out_dir / "sig.npy", sig)
    np.save(out_dir / "sample_idx.npy", sample_idx)
    np.save(out_dir / "scales.npy", scales.astype(np.float32))

    print(f"✅ PRECOMPUTE_DONE fused{fused.shape} ({S} sampled of {total_frames}), "
          f"scale~{float(np.median(scales)):.3f} -> {out_dir}")


if __name__ == "__main__":
    main()

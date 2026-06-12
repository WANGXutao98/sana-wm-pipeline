#!/usr/bin/env python3
"""Pi3X inference CLI — produces cams_pi3x.json + pts_pi3x.npy for mode_gtpose.

Usage (mirrors the interface expected by mode_gtpose.py):
  python scripts/pi3x_infer_cli.py \
    --video <path.mp4> \
    --emit-cams <out/cams_pi3x.json> \
    --emit-points <out/pts_pi3x.npy>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def read_video_frames(video_path: str) -> np.ndarray:
    """Return (T, H, W, 3) uint8 RGB frames."""
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
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return np.stack(frames, axis=0)


def recover_intrinsics_from_rays(rays: np.ndarray) -> np.ndarray:
    """Recover K (3x3) from Pi3X ray directions (H, W, 3).

    rays[v, u] ≈ normalize((u - cx, v - cy, f))
    Least-squares on center row/column.
    """
    H, W = rays.shape[:2]
    # fx, cx from center row
    row = rays[H // 2]  # (W, 3)
    us = np.arange(W, dtype=np.float64)
    ratios_x = row[:, 0] / np.clip(row[:, 2], 1e-9, None)
    A = np.stack([ratios_x, np.ones(W)], axis=1)
    fx, cx = np.linalg.lstsq(A, us, rcond=None)[0]
    # fy, cy from center column
    col = rays[:, W // 2]  # (H, 3)
    vs = np.arange(H, dtype=np.float64)
    ratios_y = col[:, 1] / np.clip(col[:, 2], 1e-9, None)
    A = np.stack([ratios_y, np.ones(H)], axis=1)
    fy, cy = np.linalg.lstsq(A, vs, rcond=None)[0]
    K = np.eye(3)
    K[0, 0] = abs(fx)
    K[1, 1] = abs(fy)
    K[0, 2] = cx
    K[1, 2] = cy
    return K


@torch.no_grad()
def run_pi3x_chunked(
    model,
    frames_t: torch.Tensor,  # (T, 3, H, W) float32 [0,1]
    chunk: int = 16,
    stride: int = 8,
    device: str = "cuda",
) -> dict:
    """Run Pi3X in overlapping chunks; return merged camera_poses and rays.

    Returns:
        dict with keys:
          "camera_poses": (T, 4, 4) float32 — world-frame camera extrinsics
          "rays":         (T, H, W, 3) float32 — unit ray directions
          "points":       (T, H, W, 3) float32 — local 3D point cloud (Z=depth)
    """
    T, _, H, W = frames_t.shape
    H_r = (H // 14) * 14
    W_r = (W // 14) * 14
    if H_r != H or W_r != W:
        src = F.interpolate(frames_t, size=(H_r, W_r), mode="bilinear", align_corners=False)
    else:
        src = frames_t

    # Accumulators for overlapping chunks
    poses_accum = np.zeros((T, 4, 4), dtype=np.float64)
    poses_count = np.zeros(T, dtype=np.float64)
    rays_accum = np.zeros((T, H_r, W_r, 3), dtype=np.float64)
    pts_accum = np.zeros((T, H_r, W_r, 3), dtype=np.float64)

    starts = list(range(0, max(T - chunk + 1, 1), stride))
    if not starts or starts[-1] + chunk < T:
        starts.append(max(0, T - chunk))

    for s in starts:
        e = min(s + chunk, T)
        out = model(src[s:e].unsqueeze(0))  # output dict
        n = e - s
        poses_accum[s:e] += out["camera_poses"][0, :n].cpu().numpy()
        poses_count[s:e] += 1
        rays_accum[s:e] += out["rays"][0, :n].cpu().numpy()
        pts_accum[s:e] += out["local_points"][0, :n].cpu().numpy()

    poses = (poses_accum / np.maximum(poses_count[:, None, None], 1.0)).astype(np.float32)
    rays = (rays_accum / np.maximum(poses_count[:, None, None, None], 1.0)).astype(np.float32)
    pts = (pts_accum / np.maximum(poses_count[:, None, None, None], 1.0)).astype(np.float32)

    # Resize back to original resolution if needed
    if H_r != H or W_r != W:
        rays = F.interpolate(
            torch.from_numpy(rays).permute(0, 3, 1, 2).to(device),
            size=(H, W), mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).cpu().numpy()
        pts = F.interpolate(
            torch.from_numpy(pts).permute(0, 3, 1, 2).to(device),
            size=(H, W), mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).cpu().numpy()

    return {"camera_poses": poses, "rays": rays, "points": pts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi3X inference CLI for mode_gtpose")
    parser.add_argument("--video", required=True, help="Input video.mp4 path")
    parser.add_argument("--emit-cams", required=True, help="Output cams_pi3x.json path")
    parser.add_argument("--emit-points", required=True, help="Output pts_pi3x.npy path")
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pi3x_weights = os.environ.get("SANA_WM_PI3X_WEIGHTS")
    if not pi3x_weights:
        raise RuntimeError("SANA_WM_PI3X_WEIGHTS env var must be set")

    # Read frames
    frames_np = read_video_frames(args.video)  # (T, H, W, 3) uint8
    T, H, W, _ = frames_np.shape
    frames_f32 = frames_np.astype(np.float32) / 255.0
    frames_t = torch.from_numpy(frames_f32).permute(0, 3, 1, 2).to(args.device)

    # Load Pi3X
    from pi3 import Pi3X  # type: ignore
    model = Pi3X.from_pretrained(pi3x_weights).to(args.device).eval()

    # Run inference
    result = run_pi3x_chunked(model, frames_t, chunk=args.chunk, stride=args.stride, device=args.device)
    camera_poses = result["camera_poses"]  # (T, 4, 4)
    rays = result["rays"]                  # (T, H, W, 3)
    pts = result["points"]                 # (T, H, W, 3)

    # Build cams_pi3x.json
    frames_list = []
    for t in range(T):
        center = camera_poses[t, :3, 3].tolist()
        K = recover_intrinsics_from_rays(rays[t])
        frames_list.append({
            "center": center,
            "K": K.tolist(),
        })
    cams_json = {"frames": frames_list}

    # Write outputs
    out_cams = pathlib.Path(args.emit_cams)
    out_pts = pathlib.Path(args.emit_points)
    out_cams.parent.mkdir(parents=True, exist_ok=True)
    out_pts.parent.mkdir(parents=True, exist_ok=True)

    out_cams.write_text(json.dumps(cams_json, indent=None))
    np.save(str(out_pts), pts.reshape(T, -1, 3))  # (T, H*W, 3)
    print(f"Wrote {out_cams} ({T} frames)")
    print(f"Wrote {out_pts} ({T} frames, {H*W} points each)")


if __name__ == "__main__":
    main()

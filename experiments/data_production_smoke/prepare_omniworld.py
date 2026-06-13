#!/usr/bin/env python3
"""Convert a single OmniWorld-Game scene to sana_wm_pipeline input format.

Usage:
  python prepare_omniworld.py \
    --scene-dir /mnt/afs/davidwang/data/omniworld/OmniWorld-Game/b04f88d1f85a \
    --out-dir   /mnt/afs/davidwang/workspace/data/omniworld_smoke/b04f88d1f85a \
    [--depth-scale 1000.0]   # uint16 / depth_scale = metres (default: 1000, mm→m)
    [--fps 30.0]             # override if fps.txt missing
    [--max-frames N]         # truncate to N frames (smoke tests)

Outputs:
  video.mp4         — RGB frames as H264 video (ffmpeg)
  gt_depth.npy      — (T, H, W) float32, metres
  gt_poses.npy      — (T, 4, 4) float32, camera-to-world
  gt_intrinsics.npy — (4,) float32 [fx, fy, cx, cy]
  orig_fps.txt      — frame rate as string

OmniWorld-Game scene structure:
  <scene_id>/color/frame_XXXXXX.png  — uint8 RGB
  <scene_id>/depth/frame_XXXXXX.png  — uint16 depth (divide by depth_scale → metres)
  <scene_id>/camera/split_0.json     — intrinsics + extrinsics per frame
  <scene_id>/fps.txt                 — frame rate
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def load_camera_json(camera_dir: Path) -> dict:
    """Load camera/split_0.json (or first available split_*.json)."""
    candidates = sorted(camera_dir.glob("split_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No split_*.json found in {camera_dir}")
    return json.loads(candidates[0].read_text())


def parse_poses_and_intrinsics(cam_data: dict, T: int):
    """Extract (T, 4, 4) c2w poses and (4,) intrinsics from camera JSON.

    OmniWorld format (top-level or per-frame intrinsics):
      {"fx": 960, "fy": 960, "cx": 640, "cy": 360,
       "frames": [{"transform_matrix": [[...]], ...}, ...]}
    """
    # Intrinsics — try top-level first, then first frame
    def _get(d, key):
        return d.get(key) or d.get("frames", [{}])[0].get(key) or \
               d.get("intrinsics", {}).get(key)

    fx, fy, cx, cy = _get(cam_data, "fx"), _get(cam_data, "fy"), \
                     _get(cam_data, "cx"), _get(cam_data, "cy")
    if any(v is None for v in [fx, fy, cx, cy]):
        raise ValueError(
            f"Cannot find fx/fy/cx/cy in camera JSON. Top-level keys: {list(cam_data.keys())}"
        )
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)

    # Poses: (T, 4, 4) c2w
    frames = cam_data.get("frames", [])[:T]
    if len(frames) < T:
        raise ValueError(f"Camera JSON has {len(frames)} frames but expected {T}")
    poses = np.stack(
        [np.array(f["transform_matrix"], dtype=np.float32) for f in frames], axis=0
    )  # (T, 4, 4)
    return poses, intrinsics


def main():
    p = argparse.ArgumentParser(description="Prepare OmniWorld scene for sana_wm_pipeline.")
    p.add_argument("--scene-dir", required=True, type=Path,
                   help="OmniWorld scene dir (contains color/, depth/, camera/)")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for pipeline artifacts")
    p.add_argument("--depth-scale", type=float, default=1000.0,
                   help="Divide uint16 depth by this to get metres (default 1000 = mm→m)")
    p.add_argument("--fps", type=float, default=None,
                   help="Override FPS (reads fps.txt if omitted)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Truncate to first N frames")
    args = p.parse_args()

    scene_dir = args.scene_dir.resolve()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # FPS
    fps_txt = scene_dir / "fps.txt"
    if args.fps is not None:
        fps = args.fps
    elif fps_txt.exists():
        fps = float(fps_txt.read_text().strip())
    else:
        fps = 30.0
        print(f"[WARN] fps.txt not found; assuming {fps} fps")
    (out_dir / "orig_fps.txt").write_text(str(fps))
    print(f"FPS: {fps}")

    # RGB frames
    color_dir = scene_dir / "color"
    rgb_files = sorted(color_dir.glob("frame_*.png")) or sorted(color_dir.glob("*.png"))
    if not rgb_files:
        print(f"[ERROR] No PNG frames in {color_dir}", file=sys.stderr)
        sys.exit(1)
    if args.max_frames:
        rgb_files = rgb_files[: args.max_frames]
    T = len(rgb_files)
    print(f"Found {T} RGB frames")

    # Video
    video_path = out_dir / "video.mp4"
    if not video_path.exists():
        list_file = out_dir / "_frame_list.txt"
        list_file.write_text("\n".join(f"file '{f}'" for f in rgb_files) + "\n")
        subprocess.check_call([
            "ffmpeg", "-y", "-r", str(fps),
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            str(video_path),
        ])
        list_file.unlink()
        first = cv2.imread(str(rgb_files[0]))
        H, W = first.shape[:2]
        print(f"Video: {video_path}  ({W}×{H}, {T} frames, {fps}fps)")
    else:
        print(f"Video already exists: {video_path}")

    # Depth
    depth_dir = scene_dir / "depth"
    depth_files = sorted(depth_dir.glob("frame_*.png")) or sorted(depth_dir.glob("*.png"))
    if len(depth_files) < T:
        print(f"[WARN] Only {len(depth_files)} depth frames for {T} RGB; truncating")
        T = min(T, len(depth_files))
        depth_files = depth_files[:T]
        rgb_files = rgb_files[:T]

    gt_depth_path = out_dir / "gt_depth.npy"
    if not gt_depth_path.exists():
        sample_d = cv2.imread(str(depth_files[0]), cv2.IMREAD_ANYDEPTH)
        DH, DW = sample_d.shape
        depths = np.zeros((T, DH, DW), dtype=np.float32)
        for i, df in enumerate(depth_files[:T]):
            d16 = cv2.imread(str(df), cv2.IMREAD_ANYDEPTH)
            if d16 is None:
                raise RuntimeError(f"Cannot read depth: {df}")
            depths[i] = d16.astype(np.float32) / args.depth_scale
        np.save(str(gt_depth_path), depths)
        print(f"GT depth: {gt_depth_path}  shape={depths.shape}  "
              f"range=[{depths[depths > 0].min():.2f}, {depths.max():.2f}]m")
    else:
        depths = np.load(str(gt_depth_path))
        print(f"GT depth (cached): {gt_depth_path}  shape={depths.shape}")

    # Camera
    camera_dir = scene_dir / "camera"
    if not camera_dir.exists():
        print(f"[WARN] No camera/ dir — skipping pose/intrinsics")
    else:
        cam_data = load_camera_json(camera_dir)
        poses, intrinsics = parse_poses_and_intrinsics(cam_data, T)
        np.save(str(out_dir / "gt_poses.npy"), poses)
        np.save(str(out_dir / "gt_intrinsics.npy"), intrinsics)
        print(f"GT poses: shape={poses.shape}")
        print(f"GT intrinsics: {intrinsics}  (fx, fy, cx, cy)")

    print(f"\nDone → {out_dir}")
    print(f"  video.mp4       — {T} frames @ {fps}fps")
    print(f"  gt_depth.npy    — (T={T}, H, W) float32 metres")
    if (out_dir / "gt_poses.npy").exists():
        print(f"  gt_poses.npy    — (T={T}, 4, 4) c2w")
        print(f"  gt_intrinsics.npy — [fx, fy, cx, cy]")


if __name__ == "__main__":
    main()

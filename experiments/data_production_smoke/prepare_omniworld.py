#!/usr/bin/env python3
"""Convert a single OmniWorld-Game scene (ModelScope tar.gz format) to sana_wm_pipeline inputs.

真实数据结构（ModelScope 下载格式）:
  annotations/OmniWorld-Game/<scene_id>/
    <scene_id>_others.tar.gz   — fps.txt, split_info.json, camera/split_N.json,
                                  droidclib/split_N.json
    <scene_id>_depth_NNNN.tar.gz — depth/<NNNNNN>.png  (uint16, /1000 → metres)
  videos/OmniWorld-Game/<scene_id>/
    <scene_id>_rgb_NNNN.tar.gz   — color/<NNNNNN>.png  (uint8 RGB)

Usage:
  python prepare_omniworld.py \\
    --annot-dir /path/to/annotations/OmniWorld-Game/020c2bed1dbb \\
    --video-dir /path/to/videos/OmniWorld-Game/020c2bed1dbb \\
    --out-dir   /path/to/output \\
    [--split-idx 0]      # 只处理第 N 个 split（smoke test 用）
    [--max-frames 120]   # 只取前 N 帧（smoke test 用）

Outputs (written to --out-dir):
  video.mp4         — RGB frames as H264 video
  gt_depth.npy      — (T, H, W) float32, metres
  gt_poses.npy      — (T, 4, 4) float32, camera-to-world
  gt_intrinsics.npy — (4,) float32 [fx, fy, cx, cy]
  orig_fps.txt      — fps as string
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def extract_others(annot_dir: Path, extract_dir: Path) -> Path:
    """Extract *_others.tar.gz to extract_dir/_others/ if not already done."""
    out = extract_dir / "_others"
    done_flag = out / ".done"
    if done_flag.exists():
        print(f"[skip] others already extracted → {out}")
        return out

    candidates = sorted(annot_dir.glob("*_others.tar.gz"))
    if not candidates:
        raise FileNotFoundError(f"No *_others.tar.gz found in {annot_dir}")
    others_tar = candidates[0]
    print(f"Extracting others: {others_tar.name} → {out}")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(others_tar) as tf:
        tf.extractall(out)
    done_flag.touch()
    return out


def parse_fps(others_dir: Path) -> float:
    """Parse fps.txt: 'FPS: 30.0 \\nProcessing time: ...'"""
    fps_path = others_dir / "fps.txt"
    if not fps_path.exists():
        print("[WARN] fps.txt not found; assuming 30.0")
        return 30.0
    content = fps_path.read_text()
    m = re.search(r"FPS:\s*([\d.]+)", content)
    if not m:
        print(f"[WARN] Cannot parse fps.txt ({content!r}); assuming 30.0")
        return 30.0
    return float(m.group(1))


def parse_split_info(others_dir: Path) -> dict:
    """Return split_info.json as dict."""
    p = others_dir / "split_info.json"
    if not p.exists():
        raise FileNotFoundError(f"split_info.json not found in {others_dir}")
    return json.loads(p.read_text())


def load_droidclib(others_dir: Path, split_idx: int) -> dict:
    """Load droidclib/split_N.json."""
    p = others_dir / "droidclib" / f"split_{split_idx}.json"
    if not p.exists():
        raise FileNotFoundError(f"droidclib/split_{split_idx}.json not found in {others_dir}")
    return json.loads(p.read_text())


def build_frame_map(
    others_dir: Path,
    split_info: dict,
    split_idx: Optional[int],
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Build sorted list of (frame_idx, pose, intrinsics).

    Returns:
        frame_indices: sorted list of global frame indices
        poses: (N, 4, 4) float32 c2w
        intrinsics: (4,) float32 [fx, fy, cx, cy]  — from first split
    """
    splits_to_use = (
        [split_idx] if split_idx is not None
        else list(range(split_info["split_num"]))
    )

    frame_to_pose: Dict[int, np.ndarray] = {}
    intrinsics_out: Optional[np.ndarray] = None

    for si in splits_to_use:
        droid = load_droidclib(others_dir, si)
        frame_list = droid["split"]   # list of global frame indices for this split
        extrinsics = droid["extrinsics"]  # list of 4x4 lists, same length
        intr = droid["crop_intrinsic"]
        k = np.array([intr["fx"], intr["fy"], intr["cx"], intr["cy"]], dtype=np.float32)
        if intrinsics_out is None:
            intrinsics_out = k

        if len(frame_list) != len(extrinsics):
            print(f"[WARN] split_{si}: frame_list len={len(frame_list)} != extrinsics len={len(extrinsics)}")
            n = min(len(frame_list), len(extrinsics))
            frame_list = frame_list[:n]
            extrinsics = extrinsics[:n]

        for fi, ex in zip(frame_list, extrinsics):
            frame_to_pose[fi] = np.array(ex, dtype=np.float32)

    if not frame_to_pose:
        raise RuntimeError("No frames found in selected splits")

    sorted_frames = sorted(frame_to_pose.keys())
    poses = np.stack([frame_to_pose[fi] for fi in sorted_frames], axis=0)  # (N,4,4)
    return sorted_frames, poses, intrinsics_out


def extract_tar_filtered(
    tar_path: Path,
    out_dir: Path,
    wanted_names: Optional[set],
    done_flag: Path,
) -> None:
    """Extract tar_path to out_dir, optionally filtering to wanted_names.

    Skips if done_flag exists.
    """
    if done_flag.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {tar_path.name} ...", end="", flush=True)
    with tarfile.open(tar_path) as tf:
        if wanted_names is None:
            tf.extractall(out_dir)
            count = len(tf.getnames())
        else:
            count = 0
            for member in tf.getmembers():
                if member.name in wanted_names:
                    tf.extract(member, out_dir)
                    count += 1
    print(f" {count} files")
    done_flag.touch()


def extract_depth_frames(
    annot_dir: Path,
    extract_dir: Path,
    needed_indices: set,
) -> Path:
    """Extract needed depth frames from all depth tar.gz archives."""
    depth_out = extract_dir / "_depth"
    depth_out.mkdir(parents=True, exist_ok=True)

    depth_tars = sorted(annot_dir.glob("*_depth_*.tar.gz"))
    if not depth_tars:
        raise FileNotFoundError(f"No *_depth_*.tar.gz found in {annot_dir}")

    print(f"Extracting depth frames ({len(needed_indices)} needed) from {len(depth_tars)} archives:")
    for dtar in depth_tars:
        done_flag = depth_out / f".done_{dtar.stem}"
        if done_flag.exists():
            print(f"  [skip] {dtar.name} already extracted")
            continue
        # Build set of wanted member names
        wanted = {f"depth/{idx:06d}.png" for idx in needed_indices}
        extract_tar_filtered(dtar, depth_out, wanted, done_flag)

    return depth_out


def extract_rgb_frames(
    video_dir: Path,
    extract_dir: Path,
    needed_indices: set,
) -> Path:
    """Extract needed RGB frames from all rgb tar.gz archives."""
    rgb_out = extract_dir / "_rgb"
    rgb_out.mkdir(parents=True, exist_ok=True)

    rgb_tars = sorted(video_dir.glob("*_rgb_*.tar.gz"))
    if not rgb_tars:
        raise FileNotFoundError(f"No *_rgb_*.tar.gz found in {video_dir}")

    print(f"Extracting RGB frames ({len(needed_indices)} needed) from {len(rgb_tars)} archives:")
    for rtar in rgb_tars:
        done_flag = rgb_out / f".done_{rtar.stem}"
        if done_flag.exists():
            print(f"  [skip] {rtar.name} already extracted")
            continue
        wanted = {f"color/{idx:06d}.png" for idx in needed_indices}
        extract_tar_filtered(rtar, rgb_out, wanted, done_flag)

    return rgb_out


def build_video_ffmpeg(frame_files: List[Path], fps: float, out_path: Path) -> None:
    """Write an H264 mp4 from a list of PNG paths using ffmpeg concat demuxer."""
    list_file = out_path.parent / "_frame_list.txt"
    list_file.write_text(
        "\n".join(f"file '{f.resolve()}'" for f in frame_files) + "\n"
    )
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-vcodec", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    list_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Prepare an OmniWorld-Game scene (tar.gz format) for sana_wm_pipeline."
    )
    p.add_argument("--annot-dir", required=True, type=Path,
                   help="annotations/OmniWorld-Game/<scene_id>/ containing *_others.tar.gz and *_depth_*.tar.gz")
    p.add_argument("--video-dir", required=True, type=Path,
                   help="videos/OmniWorld-Game/<scene_id>/ containing *_rgb_*.tar.gz")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for pipeline artifacts")
    p.add_argument("--split-idx", type=int, default=None,
                   help="Only process this split index (0-based). Default: all splits.")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Truncate to first N frames (smoke test).")
    args = p.parse_args()

    annot_dir = args.annot_dir.resolve()
    video_dir = args.video_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = out_dir / "_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"annot dir : {annot_dir}")
    print(f"video dir : {video_dir}")
    print(f"out dir   : {out_dir}")

    # ------------------------------------------------------------------
    # 1. Extract & parse metadata
    # ------------------------------------------------------------------
    print("\n[1/5] Extracting metadata (_others.tar.gz)...")
    others_dir = extract_others(annot_dir, extract_dir)

    fps = parse_fps(others_dir)
    print(f"FPS: {fps}")

    split_info = parse_split_info(others_dir)
    split_num = split_info["split_num"]
    for si, sf in enumerate(split_info["split"]):
        print(f"  Split {si}: {len(sf)} frames (indices {sf[0]}-{sf[-1]})")

    # ------------------------------------------------------------------
    # 2. Build frame index → pose mapping
    # ------------------------------------------------------------------
    print(f"\n[2/5] Building frame-pose mapping (split_idx={args.split_idx})...")
    frame_indices, poses_all, intrinsics = build_frame_map(
        others_dir, split_info, args.split_idx
    )
    print(f"  Total frames from selected splits: {len(frame_indices)}")

    # Apply --max-frames
    if args.max_frames is not None and args.max_frames < len(frame_indices):
        frame_indices = frame_indices[: args.max_frames]
        poses_all = poses_all[: args.max_frames]
        print(f"  Truncated to {len(frame_indices)} frames (--max-frames)")

    T = len(frame_indices)
    needed_indices = set(frame_indices)

    # ------------------------------------------------------------------
    # 3. Extract depth frames
    # ------------------------------------------------------------------
    print(f"\n[3/5] Extracting depth frames...")
    depth_root = extract_depth_frames(annot_dir, extract_dir, needed_indices)

    # ------------------------------------------------------------------
    # 4. Extract RGB frames
    # ------------------------------------------------------------------
    print(f"\n[4/5] Extracting RGB frames...")
    rgb_root = extract_rgb_frames(video_dir, extract_dir, needed_indices)

    # ------------------------------------------------------------------
    # 5. Build outputs
    # ------------------------------------------------------------------
    print(f"\n[5/5] Building output files ({T} frames)...")

    # --- depth array ---
    gt_depth_path = out_dir / "gt_depth.npy"
    if gt_depth_path.exists():
        print(f"  [skip] gt_depth.npy already exists")
        gt_depth = np.load(str(gt_depth_path))
    else:
        sample_path = depth_root / "depth" / f"{frame_indices[0]:06d}.png"
        sample = cv2.imread(str(sample_path), cv2.IMREAD_ANYDEPTH)
        if sample is None:
            raise RuntimeError(f"Cannot read depth sample: {sample_path}")
        DH, DW = sample.shape
        gt_depth = np.zeros((T, DH, DW), dtype=np.float32)
        for i, fi in enumerate(frame_indices):
            dpath = depth_root / "depth" / f"{fi:06d}.png"
            d16 = cv2.imread(str(dpath), cv2.IMREAD_ANYDEPTH)
            if d16 is None:
                raise RuntimeError(f"Cannot read depth frame: {dpath}")
            gt_depth[i] = d16.astype(np.float32) / 1000.0
        np.save(str(gt_depth_path), gt_depth)

    valid = gt_depth[gt_depth > 0]
    print(f"  GT depth: shape={gt_depth.shape}  range=[{valid.min():.3f}, {valid.max():.3f}]m")

    # --- poses & intrinsics ---
    gt_poses_path = out_dir / "gt_poses.npy"
    gt_intr_path = out_dir / "gt_intrinsics.npy"
    np.save(str(gt_poses_path), poses_all)
    np.save(str(gt_intr_path), intrinsics)
    print(f"  GT poses: shape={poses_all.shape}")
    print(f"  GT intrinsics: {intrinsics}")

    # --- orig_fps.txt ---
    (out_dir / "orig_fps.txt").write_text(str(fps))

    # --- video.mp4 ---
    video_path = out_dir / "video.mp4"
    if video_path.exists():
        print(f"  [skip] video.mp4 already exists")
    else:
        rgb_files = []
        for fi in frame_indices:
            rp = rgb_root / "color" / f"{fi:06d}.png"
            if not rp.exists():
                raise RuntimeError(f"Missing RGB frame: {rp}")
            rgb_files.append(rp)
        print(f"  Writing video.mp4 ({len(rgb_files)} frames @ {fps}fps) ...")
        build_video_ffmpeg(rgb_files, fps, video_path)

    video_size_mb = video_path.stat().st_size / 1024 / 1024
    print(f"  video.mp4: {T} frames ({video_size_mb:.1f} MB)")

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()

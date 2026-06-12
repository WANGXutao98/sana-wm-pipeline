#!/usr/bin/env python3
"""Prepare DL3DV scene for SANA-WM pipeline smoke test.

Input:  <scene_dir>/images/*.png
        <scene_dir>/transforms.json
Output: <scene_dir>/video.mp4         (ffmpeg, original frame rate)
        <scene_dir>/gt_poses.npy      (T, 4, 4) float32, OpenCV c2w
        <scene_dir>/gt_intrinsics.npy (4,) = [fx, fy, cx, cy]
        <scene_dir>/orig_fps.txt      (float, for frame alignment at eval time)

DL3DV convention: OpenCV c2w (X right, Y down, Z forward) — matches pipeline,
no coordinate conversion needed.

Frame alignment note: normalize.py resamples to 16fps.
Frame i (16fps) corresponds to GT frame round(i * orig_fps / 16).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def transforms_to_poses(transforms_path: Path) -> np.ndarray:
    """Parse transforms.json -> (T, 4, 4) float32 OpenCV c2w."""
    data = json.loads(transforms_path.read_text())
    frames = sorted(data["frames"], key=lambda f: f["file_path"])
    poses = np.array([f["transform_matrix"] for f in frames], dtype=np.float32)
    return poses


def transforms_to_intrinsics(transforms_path: Path) -> np.ndarray:
    """Parse transforms.json -> (4,) = [fx, fy, cx, cy] float32."""
    data = json.loads(transforms_path.read_text())
    return np.array(
        [data["fl_x"], data["fl_y"], data["cx"], data["cy"]],
        dtype=np.float32,
    )


def images_to_video(scene_dir: Path, output_mp4: Path) -> float:
    """Convert images/*.png to video.mp4 via ffmpeg; return fps."""
    img_dir = scene_dir / "images"
    if not img_dir.exists():
        raise FileNotFoundError(f"images/ not found in {scene_dir}")

    # Detect frame rate from transforms.json if available
    transforms_path = scene_dir / "transforms.json"
    fps = 30.0  # DL3DV default
    if transforms_path.exists():
        try:
            data = json.loads(transforms_path.read_text())
            fps = float(data.get("fps", data.get("camera_fps", 30.0)))
        except (KeyError, ValueError):
            pass

    # Sort images and create ffmpeg input list
    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        raise FileNotFoundError(f"No images found in {img_dir}")

    # Use ffmpeg glob pattern
    # Try to find the static-ffmpeg or system ffmpeg
    ffmpeg_bin = _find_ffmpeg(scene_dir)

    cmd = [
        ffmpeg_bin, "-y",
        "-framerate", str(fps),
        "-pattern_type", "glob",
        "-i", str(img_dir / "*.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_mp4),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try jpg if png failed
        cmd[cmd.index("*.png")] = "*.jpg"
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    return fps


def _find_ffmpeg(scene_dir: Path) -> str:
    """Find ffmpeg: project .bin/ffmpeg > static_ffmpeg > system ffmpeg."""
    # Project-local ffmpeg (installed via static-ffmpeg pip)
    project_root = Path(__file__).parent.parent.parent
    local_ffmpeg = project_root / ".bin" / "ffmpeg"
    if local_ffmpeg.exists():
        return str(local_ffmpeg)
    # Try static-ffmpeg package
    try:
        import static_ffmpeg  # type: ignore
        static_ffmpeg.add_paths()
    except ImportError:
        pass
    return "ffmpeg"  # fall back to system


def prepare_scene(scene_dir: Path) -> None:
    """Prepare one DL3DV scene for pipeline ingestion."""
    scene_dir = Path(scene_dir)
    transforms_path = scene_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(f"transforms.json not found in {scene_dir}")

    output_mp4 = scene_dir / "video.mp4"
    output_poses = scene_dir / "gt_poses.npy"
    output_intr = scene_dir / "gt_intrinsics.npy"
    output_fps = scene_dir / "orig_fps.txt"

    print(f"[prepare_dl3dv] scene: {scene_dir.name}")

    # 1. Convert images to video
    if not output_mp4.exists():
        print("  -> Converting images to video.mp4 ...")
        fps = images_to_video(scene_dir, output_mp4)
        output_fps.write_text(str(fps))
        print(f"  -> video.mp4 written (fps={fps})")
    else:
        print("  -> video.mp4 already exists, skipping")
        # Still compute fps for gt_poses alignment
        try:
            data = json.loads(transforms_path.read_text())
            fps = float(data.get("fps", data.get("camera_fps", 30.0)))
        except Exception:
            fps = 30.0
        if not output_fps.exists():
            output_fps.write_text(str(fps))

    # 2. Extract GT poses (T, 4, 4)
    poses = transforms_to_poses(transforms_path)
    np.save(str(output_poses), poses)
    print(f"  -> gt_poses.npy: shape={poses.shape}")

    # 3. Extract intrinsics (4,)
    intr = transforms_to_intrinsics(transforms_path)
    np.save(str(output_intr), intr)
    print(f"  -> gt_intrinsics.npy: {intr}")

    print(f"  [OK] {scene_dir.name}: {len(poses)} frames, fps={fps}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DL3DV scene(s) for SANA-WM pipeline smoke test"
    )
    parser.add_argument(
        "scene_dirs", nargs="+",
        help="One or more scene directories (each must contain images/ + transforms.json)"
    )
    args = parser.parse_args()

    for scene_str in args.scene_dirs:
        scene_dir = Path(scene_str)
        prepare_scene(scene_dir)

    print("\nAll scenes prepared.")


if __name__ == "__main__":
    main()

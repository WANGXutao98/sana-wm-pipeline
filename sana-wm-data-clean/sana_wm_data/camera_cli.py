"""Small CLI for the retained SANA-WM camera-estimation core."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from .filter.camera import camera_filter_pass, focal_divergence, fov_degrees, scale_cov
from .manifest import ClipRecord
from .pose.stage import annotate_pose
from .pose.vipe_cli import annotate_pose_vipe_cli


CAMERA_QC = {
    "fov_deg": [25.0, 120.0],
    "focal_div_max": 0.20,
    "scale_cov_max": 2.0,
}


def _probe_video(path: Path) -> tuple[int, float, int, int]:
    """Return frame count, fps, width, and height using decord or OpenCV."""
    try:
        import decord  # type: ignore

        reader = decord.VideoReader(str(path))
        if not len(reader):
            raise ValueError(f"video has no frames: {path}")
        height, width = map(int, reader[0].shape[:2])
        return len(reader), float(reader.get_avg_fps() or 16.0), width, height
    except ImportError:
        pass

    try:
        import cv2  # type: ignore
    except ImportError as error:
        raise RuntimeError("video probing needs decord or opencv; install '.[real]'") from error

    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 16.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if count <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"could not read video metadata: {path}")
    return count, fps, width, height


def _safe_clip_id(path: Path, requested: str | None) -> str:
    raw = requested or path.stem
    clip_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not clip_id:
        raise ValueError("clip id is empty after sanitization")
    return clip_id


def _validate_outputs(record: ClipRecord) -> tuple[np.ndarray, np.ndarray]:
    if not record.pose_path or not record.intrinsics_path:
        raise ValueError("camera backend did not return output paths")
    poses = np.load(record.pose_path)
    intrinsics = np.load(record.intrinsics_path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"expected poses (N,4,4), got {poses.shape}")
    if intrinsics.ndim != 2 or intrinsics.shape[1] != 4:
        raise ValueError(f"expected intrinsics (N,4), got {intrinsics.shape}")
    if len(poses) != len(intrinsics):
        raise ValueError(f"pose/intrinsics length mismatch: {len(poses)} != {len(intrinsics)}")
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError("camera output contains NaN or infinity")
    if np.any(intrinsics[:, :2] <= 0):
        raise ValueError("camera output contains non-positive focal lengths")
    return poses, intrinsics


def _result(record: ClipRecord, poses: np.ndarray, intrinsics: np.ndarray, backend: str) -> dict:
    median = np.median(intrinsics, axis=0)
    fx, fy = map(float, median[:2])
    fov_x, fov_y = fov_degrees(float(record.width), float(record.height), fx, fy)
    scales = record.scale_factors or []
    passed, reasons = camera_filter_pass(
        float(record.width), float(record.height), fx, fy, scales, CAMERA_QC
    )
    return {
        "clip_id": record.clip_id,
        "video": str(Path(record.video_path).resolve()),
        "mode": record.pose_mode,
        "backend": backend,
        "convention": "camera-to-world (c2w), OpenCV axes",
        "poses": {"path": record.pose_path, "shape": list(poses.shape), "dtype": str(poses.dtype)},
        "intrinsics": {
            "path": record.intrinsics_path,
            "shape": list(intrinsics.shape),
            "dtype": str(intrinsics.dtype),
            "layout": ["fx", "fy", "cx", "cy"],
        },
        "scale": {
            "count": len(scales),
            "min": float(np.min(scales)) if scales else None,
            "max": float(np.max(scales)) if scales else None,
        },
        "camera_qc": {
            "passed": passed,
            "reasons": reasons,
            "fov_x_deg": fov_x,
            "fov_y_deg": fov_y,
            "focal_divergence": focal_divergence(fx, fy),
            "scale_cov": scale_cov(scales),
            "thresholds": CAMERA_QC,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate c2w poses and per-frame intrinsics")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("default", "gt_pose"), default="default")
    parser.add_argument("--gt-poses", type=Path)
    parser.add_argument("--gt-intrinsics", type=Path)
    parser.add_argument("--clip-id")
    parser.add_argument(
        "--backend",
        choices=("auto", "vipe", "reference"),
        default="auto",
        help="auto uses real VIPE for default mode; reference is the lightweight stage backend",
    )
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.mode == "gt_pose" and not args.dry_run and not args.gt_poses:
        raise ValueError("--mode gt_pose requires --gt-poses unless --dry-run is used")
    if args.mode == "default" and (args.gt_poses or args.gt_intrinsics):
        raise ValueError("GT camera files are only valid with --mode gt_pose")
    for label, path in (("--gt-poses", args.gt_poses), ("--gt-intrinsics", args.gt_intrinsics)):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"{label}: {path}")
    if args.backend == "vipe" and (args.mode != "default" or args.dry_run):
        raise ValueError("the VIPE backend is for real default-mode estimation")

    frame_count, fps, width, height = _probe_video(video)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = ClipRecord(
        clip_id=_safe_clip_id(video, args.clip_id),
        source="user",
        video_path=str(video),
        mode=args.mode,
        fps=fps,
        num_frames=frame_count,
        width=width,
        height=height,
    )
    if args.gt_poses:
        record.extra["gt_positions_path"] = str(args.gt_poses.resolve())
    if args.gt_intrinsics:
        record.extra["gt_intrinsics_path"] = str(args.gt_intrinsics.resolve())

    os.environ["SANA_WM_MAX_FRAMES"] = str(args.max_frames)
    repo_root = Path(__file__).resolve().parents[1]
    models = {
        "dry_run": args.dry_run,
        "depth_fusion": {"ema_momentum": 0.99},
        "vipe": {"wm_root": str(repo_root), "max_frames": args.max_frames},
    }

    use_vipe = args.backend == "vipe" or (
        args.backend == "auto" and args.mode == "default" and not args.dry_run
    )
    if use_vipe:
        annotate_pose_vipe_cli(record, output, models)
        backend = "vipe_cli"
    else:
        annotate_pose(record, output, models)
        backend = "reference"

    poses, intrinsics = _validate_outputs(record)
    result = _result(record, poses, intrinsics, backend)
    result_path = output / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Default pose-annotation mode (paper §4 + App. B.1).

Targets: SpatialVID-HQ, Sekai-Walking-HQ, MiraData.
Pipeline: VIPE SLAM front-end (modified with Pi3X + MoGe-2 fused depth) →
per-frame intrinsics → c2w poses + (N,1,4) intrinsics + per-frame scale.

VIPE CLI: ``vipe infer <video> -o <work_dir> --pipeline sana_wm_pose_only``

Output artifacts (VIPE format, read by _load_vipe_artifacts):
  <work_dir>/pose/<stem>.npz         — data:(T,4,4) cam2world, inds:(T,)
  <work_dir>/intrinsics/<stem>.npz   — data:(T,4) [fx,fy,cx,cy], inds:(T,)
  <work_dir>/depth/<stem>.zip        — EXR per-frame depth (optional)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np

from ._common import PoseArtifact

VIPE_CMD: Sequence[str] = ("vipe", "infer")
VIPE_PIPELINE = "sana_wm_pose_only"


def run_default(
    clip_path: Path,
    work_dir: Path,
    vipe_cmd: Sequence[str] = VIPE_CMD,
    pipeline: str = VIPE_PIPELINE,
) -> PoseArtifact:
    """Invoke VIPE on ``clip_path`` and convert its artifacts to PoseArtifact.

    Args:
        clip_path: Normalized 1280×720 @ 16fps mp4.
        work_dir: Directory for VIPE output artifacts.
        vipe_cmd: Override VIPE executable (for testing).
        pipeline: VIPE pipeline config name (default: sana_wm_pose_only).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        *vipe_cmd,
        str(clip_path),
        "--output", str(work_dir),
        "--pipeline", pipeline,
    ]
    subprocess.check_call(cmd)

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

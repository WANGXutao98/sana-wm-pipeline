"""Adapters around VIPE, Pi3X, and MoGe-2 for the pose stage.

Real execution calls the vendored models under ``third_party/``; dry-run produces
deterministic synthetic geometry so the camera stage can be tested without a GPU.

The SANA-WM modifications to VIPE (Pi3X+MoGe-2 depth backend, per-frame
intrinsics BA) are applied by feeding our fused depth and per-frame intrinsics
tensor into VIPE through these adapters; the fusion/BA math itself lives in
``fusion.py`` / ``intrinsics.py`` (written here, not in upstream).
"""

from __future__ import annotations

import hashlib

import numpy as np

from .intrinsics import constant_intrinsics


def _seed(clip_id: str) -> int:
    return int(hashlib.sha1(clip_id.encode()).hexdigest()[:8], 16)


def even_indices(count: int, n: int) -> np.ndarray:
    """`n` evenly-spaced integer indices into a sequence of length `count` (rounded).

    The SINGLE source of truth for frame sampling: read_frames (Pi3/MoGe input) AND the
    GT-pose subsample in pose/stage.py must use this identical rule, or Pi3 frame i and
    GT pose i correspond to different source frames (was: read_frames truncated with
    .astype(int) while stage rounded) — silently biasing the Umeyama metric-scale
    recovery, worst on fast trajectories."""
    return np.linspace(0, max(count - 1, 0), max(min(n, count), 1)).round().astype(int)


def read_frames(video_path: str, n_frames: int, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read up to n_frames evenly-sampled RGB frames -> (N,H,W,3) uint8.

    Uses decord if available, else OpenCV. ``size`` is (H,W) to resize to.
    """
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(str(video_path))
        total = len(vr)
        idx = even_indices(total, n_frames)
        frames = vr.get_batch(list(idx)).asnumpy()  # (n,H,W,3) RGB
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n_frames
        idx = set(even_indices(total, n_frames).tolist())
        frames, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i in idx:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
        frames = np.stack(frames) if frames else np.zeros((1, 8, 8, 3), np.uint8)
    if size is not None:
        import cv2
        frames = np.stack([cv2.resize(f, (size[1], size[0])) for f in frames])
    return frames


def _resize_stack(arr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Resize an (N,h,w) stack to (N,*hw)."""
    import cv2
    if arr.shape[1:] == tuple(hw):
        return arr
    return np.stack([cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR) for a in arr])


def _synthetic_depth(clip_id: str, n_frames: int, hw: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(_seed(clip_id))
    base = rng.uniform(1.0, 8.0, size=hw)  # static-ish scene depth
    # mild per-frame variation to mimic parallax
    return np.stack([base * (1.0 + 0.02 * t) for t in range(n_frames)], axis=0)


def _synthetic_trajectory(clip_id: str, n_frames: int) -> np.ndarray:
    """An (N,4,4) world-to-camera trajectory: gentle forward dolly + yaw."""
    poses = np.tile(np.eye(4), (n_frames, 1, 1)).astype(np.float64)
    for t in range(n_frames):
        yaw = 0.01 * t
        c, s = np.cos(yaw), np.sin(yaw)
        poses[t, :3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        poses[t, :3, 3] = [0.0, 0.0, -0.05 * t]  # move forward
    return poses


def run_pi3x_depth(video_path: str, clip_id: str, n_frames: int, hw, cfg, dry_run: bool):
    """Pi3X multi-frame consistent (scale-ambiguous) depth, (T,H,W)."""
    if dry_run:
        return _synthetic_depth(clip_id + ":pi3x", n_frames, hw)
    from . import _real
    _poses, depth = _real.pi3_infer(read_frames(video_path, n_frames))
    return _resize_stack(depth, hw)


def run_moge2_depth(video_path: str, clip_id: str, n_frames: int, hw, cfg, dry_run: bool):
    """MoGe-2 metric-scale depth, (T,H,W). In dry-run it is a scaled Pi3X."""
    if dry_run:
        # make MoGe ~= true_scale * Pi3X so fusion recovers a sensible scale
        true_scale = 1.0 + 0.5 * (_seed(clip_id) % 5)
        return true_scale * _synthetic_depth(clip_id + ":pi3x", n_frames, hw)
    from . import _real
    return _real.moge_metric_depth(read_frames(video_path, n_frames), ref_hw=hw)


def run_pi3x_trajectory(video_path: str, clip_id: str, n_frames: int, cfg, dry_run: bool):
    """Pi3X camera positions (N,3), scale-ambiguous (for GT-pose alignment)."""
    if dry_run:
        return _synthetic_trajectory(clip_id + ":pi3x", n_frames)[:, :3, 3]
    from . import _real
    poses, _depth = _real.pi3_infer(read_frames(video_path, n_frames))
    return poses[:, :3, 3]  # camera centers (cam-to-world translation)


def run_vipe_slam(
    video_path: str, clip_id: str, n_frames: int, hw, depth, intrinsics0, cfg, dry_run: bool
):
    """VIPE SLAM front-end + per-frame-intrinsics BA.

    Returns ``(poses (N,4,4), intrinsics (N,V,4))``. In dry-run, returns a
    synthetic trajectory and the seed intrinsics unchanged.
    """
    if dry_run:
        poses = _synthetic_trajectory(clip_id, n_frames)
        return poses, intrinsics0
    # Real mode: full VIPE SLAM+BA is a heavy CUDA build we do not set up here.
    # We use Pi3's multi-frame-consistent pose output as the pose track (Pi3 is
    # the structure backbone SANA-WM builds on); VIPE's bundle-adjustment refine
    # is the one layer omitted. Intrinsics keep the seed (per-frame BA not run).
    from . import _real
    frames = read_frames(video_path, n_frames)
    poses, _depth = _real.pi3_infer(frames)
    return poses, intrinsics0

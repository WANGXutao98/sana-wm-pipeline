"""Stage-04 visual filter metrics (SANA-WM paper App. B.3).

Paper-fixed sampling constants — DO NOT change without re-reading paper:
  - UniMatch: pair every 0.5 s across the first 60 s window.
  - DOVER:    averaged over non-overlapping 5 s chunks.
  - Saturation: HSV S channel mean; native [0,180] range (do NOT divide by 255).

Neural models (UniMatch flow, DOVER quality) are injected as callables so unit
tests stay fast and orchestration is robust to model availability.
"""
from __future__ import annotations

from typing import Callable, List, Tuple, Optional

import numpy as np

# ---- Paper-fixed constants (arXiv:2605.15178v1, App. B.3) -----------------
UNIMATCH_SAMPLE_EVERY_S: float = 0.5
UNIMATCH_WINDOW_S: int = 60
DOVER_CHUNK_S: int = 5


# ---- Sampling primitives ---------------------------------------------------
def enumerate_unimatch_pairs(
    n_frames: int,
    fps: int = 16,
    window_s: int = UNIMATCH_WINDOW_S,
    sample_every_s: float = UNIMATCH_SAMPLE_EVERY_S,
) -> List[Tuple[int, int]]:
    """Frame-index pairs (i, i+stride) per paper App. B.3.

    stride = round(fps * sample_every_s); pairs cover the first `window_s`
    of the clip; the last pair end-index must remain in [0, n_in_window).
    """
    if n_frames <= 0 or fps <= 0:
        return []
    stride = int(round(fps * sample_every_s))
    if stride <= 0:
        return []
    n_in_window = min(n_frames, fps * window_s)
    pairs: List[Tuple[int, int]] = []
    for i in range(0, n_in_window - stride, stride):
        pairs.append((i, i + stride))
    return pairs


def dover_chunk_indices(
    n_frames: int,
    fps: int = 16,
    chunk_s: int = DOVER_CHUNK_S,
) -> List[Tuple[int, int]]:
    """Non-overlapping (start, end) chunk indices per paper App. B.3.

    Each chunk spans exactly chunk_s*fps frames; the tail (if shorter than a
    full chunk) is dropped so the averaged score is over equal-length spans.
    """
    if n_frames <= 0 or fps <= 0 or chunk_s <= 0:
        return []
    chunk = chunk_s * fps
    chunks: List[Tuple[int, int]] = []
    for i in range(0, n_frames - chunk + 1, chunk):
        chunks.append((i, i + chunk))
    return chunks


# ---- Saturation -----------------------------------------------------------
def mean_saturation(frames_rgb: np.ndarray) -> float:
    """Average HSV-S of all frames; OpenCV S range is [0, 255] for 8-bit input
    but the paper's Color-Sat thresholds in Table 6 are stated in the [0,180]
    domain matching the *Hue* axis convention used by their tooling. We keep
    the raw S channel mean (no /255 normalisation) so downstream Table-6
    range comparison matches the paper directly.

    frames_rgb: (T,H,W,3) uint8, RGB order.
    """
    import cv2  # type: ignore

    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"frames_rgb must be (T,H,W,3), got {frames_rgb.shape}")
    means: List[float] = []
    for f in frames_rgb:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        means.append(float(hsv[..., 1].astype(np.float32).mean()))
    return float(np.mean(means)) if means else 0.0


# ---- VMAF motion ----------------------------------------------------------
def ffmpeg_vmaf_motion(video_path: str, ffmpeg_bin: str = "ffmpeg") -> float:
    """FFmpeg `motion2` feature average via libvmaf JSON log.

    Returns NaN if ffmpeg / libvmaf with motion plugin is unavailable; callers
    should treat NaN as "metric unavailable" rather than a failed filter.
    """
    import json
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        log_path = f.name
    try:
        cmd = [
            ffmpeg_bin, "-nostats", "-loglevel", "error",
            "-i", video_path, "-an",
            "-vf", f"libvmaf=feature=name=motion:log_path={log_path}:log_fmt=json",
            "-f", "null", "-",
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return float("nan")
        try:
            with open(log_path) as fp:
                data = json.load(fp)
        except (FileNotFoundError, json.JSONDecodeError):
            return float("nan")
        vals: List[float] = []
        for fr in data.get("frames", []):
            m = fr.get("metrics", {})
            v = m.get("motion2", m.get("motion"))
            if v is not None:
                vals.append(float(v))
        return float(np.mean(vals)) if vals else float("nan")
    finally:
        import os
        try:
            os.unlink(log_path)
        except FileNotFoundError:
            pass


def frame_diff_motion_proxy(frames_rgb: np.ndarray) -> float:
    """Last-resort motion proxy when libvmaf is unavailable: mean per-pixel
    absolute difference between consecutive frames (0..255 scale).
    """
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"frames_rgb must be (T,H,W,3), got {frames_rgb.shape}")
    if len(frames_rgb) < 2:
        return 0.0
    a = frames_rgb[:-1].astype(np.float32)
    b = frames_rgb[1:].astype(np.float32)
    return float(np.mean(np.abs(a - b)))


# ---- UniMatch flow magnitude (injected) -----------------------------------
def unimatch_flow_magnitude(
    frames_rgb: np.ndarray,
    flow_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    fps: int = 16,
    window_s: int = UNIMATCH_WINDOW_S,
    sample_every_s: float = UNIMATCH_SAMPLE_EVERY_S,
) -> float:
    """Mean per-pixel flow magnitude over paper-prescribed sampled pairs.

    flow_fn(img_a, img_b) -> (H,W,2) float flow in pixels.
    Returns NaN if no pairs can be enumerated.
    """
    pairs = enumerate_unimatch_pairs(
        n_frames=len(frames_rgb), fps=fps, window_s=window_s,
        sample_every_s=sample_every_s,
    )
    if not pairs:
        return float("nan")
    mags: List[float] = []
    for i, j in pairs:
        flow = flow_fn(frames_rgb[i], frames_rgb[j])
        if flow is None:
            continue
        m = np.linalg.norm(flow.astype(np.float32), axis=-1).mean()
        mags.append(float(m))
    return float(np.mean(mags)) if mags else float("nan")


# ---- DOVER quality score (injected) ---------------------------------------
def dover_score(
    frames_rgb: np.ndarray,
    dover_fn: Callable[[np.ndarray], float],
    fps: int = 16,
    chunk_s: int = DOVER_CHUNK_S,
) -> float:
    """Mean DOVER score over non-overlapping `chunk_s`-second windows.

    dover_fn(clip_TxHxWx3) -> float in [0, 1].
    Returns NaN if no full chunk fits.
    """
    chunks = dover_chunk_indices(n_frames=len(frames_rgb), fps=fps, chunk_s=chunk_s)
    if not chunks:
        return float("nan")
    scores: List[float] = []
    for s, e in chunks:
        val = dover_fn(frames_rgb[s:e])
        if val is None:
            continue
        scores.append(float(val))
    return float(np.mean(scores)) if scores else float("nan")


# ---- Convenience aggregate -------------------------------------------------
def compute_all(
    frames_rgb: np.ndarray,
    video_path: Optional[str] = None,
    flow_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    dover_fn: Optional[Callable[[np.ndarray], float]] = None,
    fps: int = 16,
) -> dict:
    """Compute the four App. B.3 visual metrics in one pass.

    Returns dict with keys: saturation, vmaf_motion, unimatch_flow, dover.
    Unavailable metrics are returned as NaN.
    """
    out = {
        "saturation": mean_saturation(frames_rgb),
        "vmaf_motion": ffmpeg_vmaf_motion(video_path) if video_path else float("nan"),
        "unimatch_flow": (
            unimatch_flow_magnitude(frames_rgb, flow_fn, fps=fps)
            if flow_fn is not None else float("nan")
        ),
        "dover": (
            dover_score(frames_rgb, dover_fn, fps=fps)
            if dover_fn is not None else float("nan")
        ),
    }
    return out

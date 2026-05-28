"""Stage-01: probe + normalize raw videos to 1280x720 @ 16fps.

Paper §5.1 / Appendix D.1: all source videos are resampled to 720p / 16fps
before downstream annotation. Center-crop preserves field-of-view; this matches
upstream SANA-Video practice (paper does not specify crop vs. pad explicitly).
"""
from __future__ import annotations

import json
import subprocess

# If static-ffmpeg is installed, register its binaries into PATH so that
# subprocess calls to "ffmpeg"/"ffprobe" succeed without a system install.
try:
    import static_ffmpeg  # type: ignore
    static_ffmpeg.add_paths()
except Exception:
    pass  # Fallback to conda/system ffmpeg if static-ffmpeg unavailable or AFS flock fails
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float
    n_frames: int


def _parse_rate(rate: str) -> float:
    """Parse an ffprobe rational rate string like ``"30000/1001"`` -> 29.97."""
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f != 0 else 0.0
    return float(rate)


def probe(path: Path) -> VideoInfo:
    """Read width/height/fps/duration/n_frames from a video via ffprobe."""
    path = Path(path)
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    raw = subprocess.check_output(cmd)
    data = json.loads(raw)
    stream = data["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = _parse_rate(stream.get("r_frame_rate") or stream.get("avg_frame_rate") or "0/1")

    duration_s = 0.0
    if stream.get("duration") not in (None, "N/A"):
        duration_s = float(stream["duration"])
    elif data.get("format", {}).get("duration") not in (None, "N/A"):
        duration_s = float(data["format"]["duration"])

    nb_frames_raw = stream.get("nb_frames")
    if nb_frames_raw not in (None, "N/A", "0"):
        n_frames = int(nb_frames_raw)
    else:
        n_frames = int(round(fps * duration_s))

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        n_frames=n_frames,
    )


def normalize_video(
    src: Path,
    dst: Path,
    target_w: int = 1280,
    target_h: int = 720,
    fps: int = 16,
) -> VideoInfo:
    """Resample ``src`` to ``target_w x target_h`` at ``fps`` (center-crop).

    Encoder: H.264 CRF=18, ``-preset medium``, no audio.
    Returns the ffprobe info of the produced file.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale=w={target_w}:h={target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},fps={fps}"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "medium",
        "-an",
        str(dst),
    ]
    subprocess.check_call(cmd)
    return probe(dst)

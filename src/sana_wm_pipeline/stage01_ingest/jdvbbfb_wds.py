"""Ingest adapter for the junchaoh-cs/jdvbbfb-v3-full WebDataset corpus.

Per-sample layout inside each shard tar:
  {key}.mp4          — RGB video (H264)
  {key}.camera.npz   — per_frame_camera_npz_v1 (GT c2w/K_px + vipe_* refs)
Caption text lives in <group>/index.jsonl  →  record["manifest"]["prompt"]["text"]
(json_members_in_shards=false, so prompts are NOT inside the tar).

This module holds only pure / unit-testable helpers. Network + HF download
glue lives in experiments/data_production_smoke/prepare_jdvbbfb.py.
"""
from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class CameraGT:
    """Parsed GT camera state from a {key}.camera.npz member."""
    c2w: np.ndarray          # (T,4,4) float32 opencv c2w
    k_px: np.ndarray         # (T,4)   float32 [fx,fy,cx,cy] original-res pixels
    fps: float
    width: int
    height: int
    vipe_c2w: np.ndarray | None = None   # (T,4,4) float32 reference VIPE poses


def load_camera_gt(npz_bytes: bytes) -> CameraGT:
    """Parse a per_frame_camera_npz_v1 byte blob into CameraGT."""
    z = np.load(io.BytesIO(npz_bytes))
    files = set(z.files)
    if "c2w" not in files:
        raise ValueError(f"camera npz missing 'c2w' (have: {sorted(files)})")
    return CameraGT(
        c2w=z["c2w"].astype(np.float32),
        k_px=z["K_px"].astype(np.float32) if "K_px" in files
             else np.zeros((len(z["c2w"]), 4), np.float32),
        fps=float(z["fps"]) if "fps" in files else 30.0,
        width=int(z["width"]) if "width" in files else 0,
        height=int(z["height"]) if "height" in files else 0,
        vipe_c2w=z["vipe_c2w"].astype(np.float32) if "vipe_c2w" in files else None,
    )


@dataclass(frozen=True)
class SampleRef:
    """One row of <group>/index.jsonl, with caption/fps hoisted for convenience."""
    sample_id: str
    key: str
    shard: str
    video_member: str
    camera_member: str
    caption: str
    fps: float


def read_index(index_path: Path) -> list[SampleRef]:
    """Parse a group's index.jsonl into SampleRef rows."""
    refs: list[SampleRef] = []
    for line in Path(index_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        man = rec.get("manifest", {})
        prompt = man.get("prompt", {}) or {}
        video = man.get("video", {}) or {}
        refs.append(SampleRef(
            sample_id=rec.get("sample_id", rec["key"]),
            key=rec["key"],
            shard=rec["shard"],
            video_member=rec["video_member"],
            camera_member=rec["camera_member"],
            caption=prompt.get("text", "") or "",
            fps=float(video.get("fps", 0.0) or 0.0),
        ))
    return refs


def _split_key(member_name: str) -> tuple[str, str]:
    """'foo.bar.camera.npz' -> ('foo.bar', '.camera.npz'); 'foo.mp4' -> ('foo', '.mp4')."""
    if member_name.endswith(".camera.npz"):
        return member_name[: -len(".camera.npz")], ".camera.npz"
    if member_name.endswith(".mp4"):
        return member_name[: -len(".mp4")], ".mp4"
    return member_name, ""


def iter_tar_samples(fileobj, limit: int | None = None
                     ) -> Iterator[tuple[str, bytes, bytes]]:
    """Stream a shard tar, yielding (key, mp4_bytes, camera_npz_bytes)."""
    pending: dict[str, dict[str, bytes]] = {}
    n = 0
    with tarfile.open(fileobj=fileobj, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            key, ext = _split_key(m.name)
            if ext not in (".mp4", ".camera.npz"):
                continue
            data = tf.extractfile(m).read()
            slot = pending.setdefault(key, {})
            slot[ext] = data
            if ".mp4" in slot and ".camera.npz" in slot:
                yield key, slot[".mp4"], slot[".camera.npz"]
                pending.pop(key, None)
                n += 1
                if limit is not None and n >= limit:
                    return


_STUB_CAPTION = "A static real-world scene with no camera-action description."


def write_scene_dir(out_base: Path, scene_id: str, mp4_bytes: bytes,
                    camera_npz_bytes: bytes, caption: str) -> Path:
    """Materialize one sample into a prepare_omniworld-compatible scene dir."""
    gt = load_camera_gt(camera_npz_bytes)
    scene = Path(out_base) / scene_id
    scene.mkdir(parents=True, exist_ok=True)

    (scene / "video.mp4").write_bytes(mp4_bytes)
    np.save(scene / "gt_poses.npy", gt.c2w)
    np.save(scene / "gt_intrinsics.npy", gt.k_px)
    (scene / "orig_fps.txt").write_text(str(gt.fps))
    (scene / "caption.txt").write_text(caption.strip() or _STUB_CAPTION)
    if gt.vipe_c2w is not None:
        np.save(scene / "vipe_ref_poses.npy", gt.vipe_c2w)
    return scene

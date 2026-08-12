# src/sana_wm_pipeline/qc/stage2_deep.py
"""Stage 2: deep targeted checks (PyAV, scene cut, black frames, frozen trajectory)."""
from __future__ import annotations
import io, json, random, tarfile, tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any
import numpy as np

try:
    import av
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False

from sana_wm_pipeline.stage04_filter.scene_cut import count_scene_cuts

_FROZEN_THRESHOLD = 1e-4
_BLACK_BRIGHTNESS = 10  # pixel mean < this → black frame


def count_video_frames_av(video_bytes: bytes) -> int:
    if not _AV_AVAILABLE:
        raise RuntimeError("pip install av")
    if not video_bytes:
        return 0
    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            if not container.streams.video:
                return 0
            stream = container.streams.video[0]
            if stream.frames > 0:
                return int(stream.frames)
            return sum(1 for _ in container.decode(video=0))
    except Exception:
        return -1  # -1 indicates error


def check_black_frame_ratio(video_bytes: bytes, brightness_threshold: int = _BLACK_BRIGHTNESS) -> float:
    """Fraction of frames with mean brightness < threshold."""
    if not _AV_AVAILABLE or not video_bytes:
        return 0.0
    total, black = 0, 0
    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            for frame in container.decode(video=0):
                arr = frame.to_ndarray(format="gray")
                if arr.mean() < brightness_threshold:
                    black += 1
                total += 1
    except Exception:
        return 0.0
    return black / total if total else 0.0


def check_trajectory_frozen(poses_c2w: np.ndarray, frozen_threshold: float = _FROZEN_THRESHOLD) -> tuple[bool, float]:
    t = poses_c2w[:, :3, 3].astype(np.float64)
    if len(t) < 2:
        return False, 0.0
    steps = np.linalg.norm(np.diff(t, axis=0), axis=1)
    ratio = float((steps < frozen_threshold).mean())
    return ratio > 0.5, ratio


def count_scene_cuts_from_bytes(video_bytes: bytes, threshold: float = 27.0) -> int:
    """Write mp4 to temp file, run PySceneDetect, return cut count."""
    if not video_bytes:
        return -1
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name
        return count_scene_cuts(tmp_path, threshold=threshold)
    except Exception:
        return -1  # -1 = unavailable
    finally:
        if tmp_path:
            import os
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def deep_check_sample(sample_id: str, tar_path: Path, group_name: str) -> dict[str, Any]:
    from sana_wm_pipeline.qc.group_config import get_group_config
    tar_path = Path(tar_path)
    cfg = get_group_config(group_name)
    stage2: dict[str, Any] = {
        "video_T": -1, "video_T_matches_npy": None,
        "black_frame_ratio": None, "scene_cuts": None,
        "traj_frozen": None, "frozen_ratio": None, "reasons": [],
    }
    try:
        with tarfile.open(tar_path, "r") as tf:
            # Video bytes
            try:
                video_bytes = tf.extractfile(tf.getmember(f"{sample_id}.mp4")).read()
            except KeyError:
                stage2["reasons"].append("mp4_not_found")
                return {"sample_id": sample_id, "stage2": stage2}

            # Frame count
            try:
                video_T = count_video_frames_av(video_bytes)
                stage2["video_T"] = video_T
            except Exception as e:
                stage2["reasons"].append(f"av_error: {e}")
                video_T = -1

            # Black frame ratio
            try:
                stage2["black_frame_ratio"] = round(check_black_frame_ratio(video_bytes), 4)
                if stage2["black_frame_ratio"] > 0.30:
                    stage2["reasons"].append(f"black_frame_ratio={stage2['black_frame_ratio']:.2f} > 0.30")
            except Exception:
                pass

            # Scene cuts (only for groups with max_scene_cuts limit)
            if cfg.max_scene_cuts is not None:
                n_cuts = count_scene_cuts_from_bytes(video_bytes)
                if n_cuts < 0:
                    stage2["scene_cuts"] = None
                    stage2["reasons"].append("scene_cuts_error: detection failed")
                else:
                    stage2["scene_cuts"] = n_cuts
                    if n_cuts > cfg.max_scene_cuts:
                        stage2["reasons"].append(f"scene_cuts={n_cuts} > {cfg.max_scene_cuts}")

            # Trajectory frozen
            try:
                poses = np.load(io.BytesIO(tf.extractfile(
                    tf.getmember(f"{sample_id}.poses_c2w.npy")).read()))
                npy_T = int(poses.shape[0])
                stage2["video_T_matches_npy"] = (video_T == npy_T)
                if video_T > 0 and video_T != npy_T:
                    stage2["reasons"].append(f"video_npy_T_mismatch: video={video_T} npy={npy_T}")
                frozen, ratio = check_trajectory_frozen(poses)
                stage2["traj_frozen"] = frozen
                stage2["frozen_ratio"] = round(ratio, 4)
                if frozen:
                    stage2["reasons"].append(f"traj_frozen: {ratio:.1%} frames stationary")
            except Exception as e:
                stage2["reasons"].append(f"poses_error: {e}")

    except Exception as e:
        stage2["reasons"].append(f"tar_error: {e}")

    return {"sample_id": sample_id, "stage2": stage2}


def _worker_fn(args: tuple) -> dict:
    sid, tar_path, group_name = args
    return deep_check_sample(sid, Path(tar_path), group_name)


def run_stage2(
    stage1_jsonl: Path, output_jsonl: Path,
    sample_frac: float = 0.05, n_workers: int = 16,
) -> int:
    stage1_jsonl, output_jsonl = Path(stage1_jsonl), Path(output_jsonl)
    if not stage1_jsonl.exists():
        raise FileNotFoundError(f"stage1_jsonl not found: {stage1_jsonl}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[str, str, str]] = []
    all_records: dict[str, dict] = {}
    rng = random.Random(42)

    with open(stage1_jsonl, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            all_records[sid] = rec
            verdict = rec.get("verdict", "pass")
            if verdict == "fail":
                continue
            if verdict == "flag" or rng.random() < sample_frac:
                selected.append((sid, rec["tar_path"], rec.get("group", "")))

    if not selected:
        output_jsonl.write_text("", encoding="utf-8")
        return 0

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        n_proc = min(max(1, n_workers), len(selected))
        with Pool(processes=n_proc) as pool:
            for s2_rec in pool.imap_unordered(_worker_fn, [(s, t, g) for s, t, g in selected]):
                merged = dict(all_records[s2_rec["sample_id"]])
                merged["stage2"] = s2_rec["stage2"]
                fout.write(json.dumps(merged, ensure_ascii=False) + "\n")

    return len(selected)

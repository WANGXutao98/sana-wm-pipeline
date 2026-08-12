"""Stage 1: fast full-coverage scan of tar shards (CPU-only, multiprocessing)."""
from __future__ import annotations
import io, json, tarfile
from multiprocessing import Pool
from pathlib import Path
from typing import Any
import numpy as np
from sana_wm_pipeline.qc.metrics import compute_stage1_metrics
from sana_wm_pipeline.qc.group_config import get_group_config, compute_verdict

_REQUIRED_EXTS = {".mp4", ".poses_c2w.npy", ".intrinsics.npy", ".scale.npy", ".caption.txt", ".meta.json"}
IMAGE_WH = (1280, 720)


def _extract_samples_from_tar(tar_path: Path) -> dict[str, dict[str, bytes]]:
    """Extract samples from tar, recovering as many as possible even if tar is corrupted.

    Strategy: Read members one-by-one with next() instead of getmembers(),
    so we can recover partial data before hitting corruption.
    """
    samples: dict[str, dict[str, bytes]] = {}
    tf = None
    members_read = 0

    try:
        tf = tarfile.open(tar_path, "r")

        # Read members one by one using next() - this allows partial recovery
        while True:
            try:
                member = tf.next()
                if member is None:  # End of archive
                    break

                if not member.isfile():
                    continue

                name = member.name
                stem, sep, suffix = name.partition(".")
                if not sep:
                    continue

                ext = "." + suffix
                try:
                    f = tf.extractfile(member)
                    if f is not None:
                        samples.setdefault(stem, {})[ext] = f.read()
                        members_read += 1
                except Exception as extract_err:
                    # Individual file extraction error - skip this file but continue
                    import sys
                    print(f"WARNING: Failed to extract {name} from {tar_path}: {extract_err}",
                          file=sys.stderr, flush=True)
                    continue

            except (tarfile.TarError, EOFError, OSError) as member_err:
                # Hit corruption while reading member list - stop here but keep what we got
                import sys
                print(f"WARNING: Corruption in {tar_path} after reading {members_read} members: {member_err}",
                      file=sys.stderr, flush=True)
                break

    except (tarfile.TarError, OSError, EOFError) as open_err:
        # Can't even open the tar - truly corrupted
        import sys
        print(f"WARNING: Cannot open tar {tar_path}: {open_err}", file=sys.stderr, flush=True)
        return {}
    finally:
        if tf is not None:
            try:
                tf.close()
            except Exception:
                pass

    if members_read > 0 and len(samples) > 0:
        import sys
        print(f"INFO: Recovered {len(samples)} samples from {tar_path} (read {members_read} members)",
              file=sys.stderr, flush=True)

    return samples


def _decode_video_frames(mp4_bytes: bytes) -> np.ndarray | None:
    """Decode mp4 → (T,H,W,3) uint8 RGB. Returns None on failure."""
    if not mp4_bytes:
        return None
    try:
        import av
        frames = []
        with av.open(io.BytesIO(mp4_bytes)) as container:
            for packet in container.demux(video=0):
                for frame in packet.decode():
                    frames.append(frame.to_ndarray(format="rgb24"))
        return np.array(frames, dtype=np.uint8) if frames else None
    except Exception:
        return None


def _scan_one_sample(
    sample_id: str,
    file_bytes: dict[str, bytes],
    group_name: str,
    tar_path: str,
    read_video_frames: bool,
) -> dict[str, Any]:
    cfg = get_group_config(group_name)
    missing = [ext for ext in sorted(_REQUIRED_EXTS) if ext not in file_bytes]
    if missing:
        return {
            "sample_id": sample_id, "group": group_name, "tar_path": tar_path,
            "verdict": "fail", "flag_reasons": [f"missing files: {missing}"], "metrics": {},
        }
    try:
        poses = np.load(io.BytesIO(file_bytes[".poses_c2w.npy"]))
        intrinsics = np.load(io.BytesIO(file_bytes[".intrinsics.npy"]))
        scale = np.load(io.BytesIO(file_bytes[".scale.npy"]))
        caption = file_bytes[".caption.txt"].decode("utf-8", errors="replace")
        meta = json.loads(file_bytes[".meta.json"].decode("utf-8"))
        meta_T = int(meta.get("T", -1))
    except Exception as exc:
        return {
            "sample_id": sample_id, "group": group_name, "tar_path": tar_path,
            "verdict": "fail", "flag_reasons": [f"load_error: {exc}"], "metrics": {},
        }

    frames_rgb = None
    if read_video_frames and cfg.saturation_min is not None:
        frames_rgb = _decode_video_frames(file_bytes[".mp4"])

    metrics = compute_stage1_metrics(
        poses=poses, intrinsics=intrinsics, scale=scale,
        caption=caption, meta_T=meta_T, image_wh=IMAGE_WH,
        jump_threshold_m=cfg.jump_threshold_m,
        min_caption_len=cfg.min_caption_len,
        frames_rgb=frames_rgb,
    )
    verdict, flag_reasons = compute_verdict(metrics, cfg)
    return {
        "sample_id": sample_id, "group": group_name, "tar_path": str(tar_path),
        "verdict": verdict, "flag_reasons": flag_reasons, "metrics": metrics,
    }


def scan_tar(tar_path: Path, group_name: str, read_video_frames: bool = False) -> list[dict]:
    tar_path = Path(tar_path)
    samples = _extract_samples_from_tar(tar_path)
    return [
        _scan_one_sample(sid, fb, group_name, str(tar_path), read_video_frames)
        for sid, fb in samples.items()
    ]


def _worker_fn(args: tuple) -> list[dict]:
    tar_path, group_name, read_video_frames = args
    return scan_tar(Path(tar_path), group_name, read_video_frames)


def run_stage1(
    tar_paths: list[Path],
    group_name: str,
    output_jsonl: Path,
    n_workers: int = 32,
    read_video_frames: bool = False,
) -> int:
    if not tar_paths:
        raise ValueError("tar_paths cannot be empty")
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    worker_args = [(str(p), group_name, read_video_frames) for p in tar_paths]
    total = 0
    empty_tars = 0
    with open(output_jsonl, "w", encoding="utf-8") as fout:
        n_proc = min(max(1, n_workers), len(tar_paths))
        with Pool(processes=n_proc, maxtasksperchild=1) as pool:
            for batch in pool.imap_unordered(_worker_fn, worker_args):
                if len(batch) == 0:
                    empty_tars += 1
                for rec in batch:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
    if empty_tars > 0:
        print(f"[stage1] WARNING: Skipped {empty_tars} corrupted/empty tar files", flush=True)
    return total

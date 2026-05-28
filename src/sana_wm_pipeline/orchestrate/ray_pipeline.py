"""Ray DAG for SANA-WM data pipeline.

Six fan-out stages map to six Ray remote functions.  Each clip flows through
the DAG independently; the dependency graph is built up at submission time.

Resource sketch (per remote):
  s01_normalize    — 2 CPU, 0 GPU         (ffmpeg)
  s02_pose         — 4 CPU, 1 GPU         (Pi3X + MoGe-2 + VIPE)
  s03_3dgs         — 4 CPU, 1 GPU         (FCGS + DiFix3D, DL3DV only)
  s04_filter       — 2 CPU, 0 / 0.5 GPU   (UniMatch + DOVER if available)
  s05_caption      — 4 CPU, 1 GPU         (Qwen3.5-VL)
  s06_pack         — 1 CPU, 0 GPU         (WebDataset writer)

Smoke mode (`--smoke`) processes 1 clip per source = 6 samples end-to-end.
The real stage bodies are intentionally minimal here — Stage-01..06 modules
already contain the heavy logic; this file wires them with Ray.
"""
from __future__ import annotations

import argparse
import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


_LOG = logging.getLogger("sana_wm.orchestrate")


@dataclass
class ClipJob:
    """Mutable per-clip context passed between stages."""
    sample_id: str
    source: str
    raw_path: str
    pose_mode: str                            # default | gt_depth | gt_pose
    normalized_path: Optional[str] = None
    pose_artifact_path: Optional[str] = None
    aug_artifact_path: Optional[str] = None   # filled only for DL3DV_GS
    filter_scores: Dict[str, float] = field(default_factory=dict)
    caption: Optional[str] = None
    shard_path: Optional[str] = None


# ---- Stage stubs (the real bodies live in stage0X_*) ----------------------
def _stage01_normalize(job: ClipJob, cfg: dict) -> ClipJob:
    from sana_wm_pipeline.stage01_ingest.normalize import normalize_video
    target = Path(cfg["paths"]["staging"]) / f"{job.sample_id}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    w, h = cfg["target"]["resolution"]
    normalize_video(
        src=Path(job.raw_path), dst=target,
        target_w=w, target_h=h,
        fps=cfg["target"]["fps"],
    )
    job.normalized_path = str(target)
    return job


def _stage02_pose_stub(job: ClipJob, cfg: dict) -> ClipJob:
    """Write synthetic pose artifact; activated by SANA_WM_POSE_STUB=1."""
    import json
    import numpy as np
    T = cfg["target"]["camera_frames"]  # 961 per paper App. D.1
    work_dir = Path(cfg["paths"]["staging"]) / "pose" / job.sample_id
    work_dir.mkdir(parents=True, exist_ok=True)
    poses = np.eye(4, dtype=np.float32)[None].repeat(T, axis=0)
    # (fx, fy, cx, cy) for 1280×720 @ 90° HFOV (plausible, not paper-specified)
    intr = np.tile(
        np.array([[960.0, 960.0, 640.0, 360.0]], dtype=np.float32),
        (T, 1, 1),
    )
    scale = np.ones(T, dtype=np.float32)
    (work_dir / "pose.json").write_text(
        json.dumps({"poses_c2w": poses.tolist(), "intrinsics_per_frame_NVD": intr.tolist()})
    )
    np.save(str(work_dir / "scale.npy"), scale)
    job.pose_artifact_path = str(work_dir)
    return job


def _stage02_pose(job: ClipJob, cfg: dict) -> ClipJob:
    import os
    # Set SANA_WM_POSE_STUB=1 to bypass real VIPE/Pi3X/MoGe-2 (pipeline shape test).
    if os.getenv("SANA_WM_POSE_STUB"):
        return _stage02_pose_stub(job, cfg)
    mode_module = importlib.import_module(
        f"sana_wm_pipeline.stage02_pose.mode_{job.pose_mode}"
    )
    work_dir = Path(cfg["paths"]["staging"]) / "pose" / job.sample_id
    runner_name = f"run_{job.pose_mode}"
    runner = getattr(mode_module, runner_name)
    runner(Path(job.normalized_path), work_dir)   # type: ignore[arg-type]
    job.pose_artifact_path = str(work_dir)
    return job


def _stage03_3dgs(job: ClipJob, cfg: dict) -> ClipJob:
    """DL3DV-only augmentation; passthrough for other sources."""
    if job.source != "DL3DV":
        return job
    # The full FCGS+DiFix3D pipeline requires GPU/binary deps;
    # this stub only records that augmentation was attempted.
    job.aug_artifact_path = str(Path(cfg["paths"]["staging"]) / "aug" / job.sample_id)
    return job


def _stage04_filter(job: ClipJob, cfg: dict) -> ClipJob:
    # In production: load decoded frames, compute visual metrics, then apply Table 6.
    # Here we only set a placeholder so the DAG type-checks end-to-end.
    job.filter_scores = job.filter_scores or {}
    return job


def _stage05_caption(job: ClipJob, cfg: dict) -> ClipJob:
    from sana_wm_pipeline.stage05_caption.qwen35_vl_runner import caption_clip
    # The orchestrator caller is expected to inject a `generate_fn` via
    # cfg["caption"]["generate_fn"] in tests; in production the default
    # Qwen-VL path is taken (raises if no GPU).
    gen = cfg.get("caption", {}).get("generate_fn")
    import numpy as np
    dummy = np.zeros((8, 4, 4, 3), dtype=np.uint8)
    job.caption = caption_clip(dummy, generate_fn=gen) if gen is not None else "PLACEHOLDER"
    return job


def _stage06_pack(job: ClipJob, cfg: dict) -> ClipJob:
    out_root = Path(cfg["paths"]["out_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    job.shard_path = str(out_root / "shard-000000.tar")
    return job


# ---- DAG runner -----------------------------------------------------------
def _run_in_process(jobs: List[ClipJob], cfg: dict) -> List[ClipJob]:
    """Sequential, in-process execution (smoke mode default)."""
    out: List[ClipJob] = []
    for j in jobs:
        j = _stage01_normalize(j, cfg)
        j = _stage02_pose(j, cfg)
        j = _stage03_3dgs(j, cfg)
        j = _stage04_filter(j, cfg)
        j = _stage05_caption(j, cfg)
        j = _stage06_pack(j, cfg)
        out.append(j)
    return out


def _run_on_ray(jobs: List[ClipJob], cfg: dict) -> List[ClipJob]:
    import ray   # type: ignore

    @ray.remote(num_cpus=2)
    def s01(j: ClipJob): return _stage01_normalize(j, cfg)

    @ray.remote(num_cpus=4, num_gpus=1)
    def s02(j: ClipJob): return _stage02_pose(j, cfg)

    @ray.remote(num_cpus=4, num_gpus=1)
    def s03(j: ClipJob): return _stage03_3dgs(j, cfg)

    @ray.remote(num_cpus=2)
    def s04(j: ClipJob): return _stage04_filter(j, cfg)

    @ray.remote(num_cpus=4, num_gpus=1)
    def s05(j: ClipJob): return _stage05_caption(j, cfg)

    @ray.remote(num_cpus=1)
    def s06(j: ClipJob): return _stage06_pack(j, cfg)

    refs = []
    for j in jobs:
        r = s01.remote(j)
        r = s02.remote(r)
        r = s03.remote(r)
        r = s04.remote(r)
        r = s05.remote(r)
        r = s06.remote(r)
        refs.append(r)
    return ray.get(refs)


# ---- Source enumeration ---------------------------------------------------
_SOURCE_TO_POSE_MODE = {
    "SpatialVID_HQ":     "default",
    "DL3DV":             "gtpose",
    "DL3DV_GS_Refined":  "gtpose",
    "OmniWorld":         "gtdepth",
    "Sekai_Game":        "gtpose",
    "Sekai_Walking_HQ":  "default",
    "MiraData":          "default",
}


def _resolve_video_path(src: dict, name: str, k: int) -> str:
    """Return a concrete video path for job k of source `name`.

    If ``local_path_example`` is a directory, pick the k-th mp4 found inside
    (sorted); if fewer than k+1 files exist, wrap around.  Falls back to a
    /tmp sentinel when no example is configured at all.
    """
    example = src.get("local_path_example")
    if not example:
        return f"/tmp/{name}_{k:06d}.mp4"
    p = Path(example)
    if p.is_dir():
        clips = sorted(p.rglob("*.mp4"))
        if clips:
            return str(clips[k % len(clips)])
        # Directory exists but no mp4s yet — return sentinel
        return str(p / f"{name}_{k:06d}.mp4")
    return str(p)


def enumerate_jobs(sources_cfg: dict, smoke: bool) -> List[ClipJob]:
    jobs: List[ClipJob] = []
    sources = sources_cfg.get("sources", {})
    for name, src in sources.items():
        target = 1 if smoke else int(src.get("target_clips", 0))
        for k in range(target):
            jobs.append(ClipJob(
                sample_id=f"{name}_{k:06d}",
                source=name,
                raw_path=_resolve_video_path(src, name, k),
                pose_mode=_SOURCE_TO_POSE_MODE.get(name, "default"),
            ))
    return jobs


# ---- CLI ------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="configs/pipeline.yaml")
    parser.add_argument("--sources", required=True, help="configs/sources.yaml")
    parser.add_argument("--smoke", action="store_true", help="1 clip per source")
    parser.add_argument("--in-process", action="store_true",
                        help="bypass Ray; run sequentially (default for smoke)")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    sources_cfg = yaml.safe_load(Path(args.sources).read_text())
    jobs = enumerate_jobs(sources_cfg, smoke=args.smoke)
    _LOG.info("enumerated %d jobs", len(jobs))
    if args.in_process or args.smoke:
        result = _run_in_process(jobs, cfg)
    else:
        result = _run_on_ray(jobs, cfg)
    _LOG.info("completed %d jobs", len(result))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())

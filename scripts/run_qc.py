#!/usr/bin/env python3
# scripts/run_qc.py
"""Stage 1+2 CLI entry point for SANA-WM Output QC System."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sana_wm_pipeline.qc.stage1_fast import run_stage1
from sana_wm_pipeline.qc.stage2_deep import run_stage2
from sana_wm_pipeline.qc.report import run_report


def main():
    p = argparse.ArgumentParser(description="SANA-WM QC Stage 1+2")
    p.add_argument("--tar-root", type=Path, help="Root dir containing .tar shards (recursive)")
    p.add_argument("--group", default="", help="Dataset group name (e.g. wds-OmniWorld-Game)")
    p.add_argument("--output-dir", type=Path, default=Path("./qc_output"))
    p.add_argument("--n-workers", type=int, default=32)
    p.add_argument("--sample-frac", type=float, default=0.05,
                   help="Fraction of passing samples to deep-check in Stage 2")
    p.add_argument("--read-video-frames", action="store_true",
                   help="Decode video frames in Stage 1 for saturation check")
    p.add_argument("--skip-stage2", action="store_true")
    p.add_argument("--report-only", action="store_true",
                   help="Skip Stage 1+2, regenerate report from existing jsonl")
    args = p.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    s1_jsonl = out / "stage1_results.jsonl"
    s2_jsonl = out / "stage2_results.jsonl"

    if not args.report_only:
        if not args.tar_root:
            p.error("--tar-root required unless --report-only")
        tars = sorted(args.tar_root.rglob("*.tar"))
        print(f"[stage1] {len(tars)} shards under {args.tar_root}", flush=True)
        n = run_stage1(tars, args.group, s1_jsonl, args.n_workers, args.read_video_frames)
        print(f"[stage1] {n} samples → {s1_jsonl}", flush=True)

        if not args.skip_stage2:
            n2 = run_stage2(s1_jsonl, s2_jsonl, args.sample_frac, args.n_workers)
            print(f"[stage2] {n2} samples deep-checked → {s2_jsonl}", flush=True)

    s2_in = s2_jsonl if (not args.skip_stage2 and s2_jsonl.exists()) else None
    run_report(s1_jsonl, s2_in, None, out)
    print(f"[report] {out}/report.html + {out}/manifests/", flush=True)


if __name__ == "__main__":
    main()

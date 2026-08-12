#!/usr/bin/env python3
# scripts/run_stage3_cmcc_debug.py
"""CMCC per-GPU Stage 3 runner with detailed logging."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sana_wm_pipeline.stage04_filter.apply_table6 import load_thresholds
from sana_wm_pipeline.qc.stage3_gpu import (
    process_sample_stage3, load_unimatch_fn, load_dover_fn, load_qwen_fn,
)


def main():
    p = argparse.ArgumentParser(description="CMCC Stage 3 GPU worker")
    p.add_argument("--stage1-jsonl", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--qwen-dir", type=Path, required=True)
    p.add_argument("--unimatch-dir", type=Path, required=True)
    p.add_argument("--worker-id", type=int, required=True, help="0-indexed worker id")
    p.add_argument("--total-workers", type=int, required=True)
    p.add_argument("--table6-cfg", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table6_cfg = load_thresholds(args.table6_cfg)

    # Load models
    print(f"[worker {args.worker_id}] loading UniMatch...", flush=True)
    flow_fn = load_unimatch_fn(str(args.unimatch_dir), args.device)
    print(f"[worker {args.worker_id}] loading DOVER...", flush=True)
    dover_fn = load_dover_fn(args.device)
    print(f"[worker {args.worker_id}] loading Qwen3.5-27B-VL...", flush=True)
    vlm_call = load_qwen_fn(str(args.qwen_dir), args.device)
    print(f"[worker {args.worker_id}] models ready.", flush=True)

    # Assign records to this worker by round-robin index
    out_jsonl = args.output_dir / f"stage3_worker{args.worker_id:03d}.jsonl"
    cap_jsonl = args.output_dir / f"caption_overrides_worker{args.worker_id:03d}.jsonl"
    done_path = args.output_dir / f"stage3_worker{args.worker_id:03d}.done"

    if done_path.exists():
        print(f"[worker {args.worker_id}] already done, skipping.", flush=True)
        return

    processed_count = 0
    start_time = time.time()

    with open(args.stage1_jsonl) as fin, \
         open(out_jsonl, "w") as fout, \
         open(cap_jsonl, "w") as cap_fout:

        for idx, line in enumerate(fin):
            if idx % args.total_workers != args.worker_id:
                continue

            rec = json.loads(line)
            if rec.get("verdict") == "fail":
                continue

            sid = rec["sample_id"]
            has_cw = bool(rec.get("metrics", {}).get("camera_words"))

            # Start processing
            sample_start = time.time()
            print(f"[worker {args.worker_id}] [{processed_count+1}] START processing sample: {sid}", flush=True)

            s3_rec = process_sample_stage3(
                sid, rec["tar_path"], rec.get("group", ""),
                flow_fn=flow_fn, dover_fn=dover_fn, vlm_call=vlm_call,
                table6_cfg=table6_cfg, has_camera_words=has_cw,
            )

            sample_elapsed = time.time() - sample_start
            processed_count += 1

            merged = dict(rec)
            merged["stage3"] = s3_rec["stage3"]
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
            fout.flush()

            cap_rev = s3_rec["stage3"].get("caption_revised")
            if cap_rev:
                cap_fout.write(json.dumps({
                    "sample_id": sid,
                    "caption_original": s3_rec.get("caption_original", ""),
                    "caption_revised": cap_rev,
                }, ensure_ascii=False) + "\n")
                cap_fout.flush()

            total_elapsed = time.time() - start_time
            avg_time = total_elapsed / processed_count
            print(f"[worker {args.worker_id}] [{processed_count}] DONE sample: {sid} | "
                  f"time: {sample_elapsed:.1f}s | avg: {avg_time:.1f}s/sample | "
                  f"total: {total_elapsed/60:.1f}min", flush=True)

    done_path.write_text("done")
    print(f"[worker {args.worker_id}] finished {processed_count} samples -> {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()

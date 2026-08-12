#!/usr/bin/env python3
# scripts/run_stage3_cmcc.py
"""CMCC per-GPU Stage 3 runner. Called once per GPU by the SSH launcher."""
from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path

# ===== CMCC 环境配置（必须在任何模型导入前设置） =====
os.environ['TORCH_HOME'] = '/root/work/david_work/cache/torch'  # DOVER 需要 convnext 权重
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "DOVER"))  # DOVER 本地导入

from sana_wm_pipeline.stage04_filter.apply_table6 import load_thresholds
from sana_wm_pipeline.qc.stage3_gpu import (
    process_sample_stage3, load_unimatch_fn, load_dover_fn, load_qwen_fn,
)


def main():
    p = argparse.ArgumentParser(description="CMCC Stage 3 GPU worker")
    p.add_argument("--stage12-jsonl", type=Path, required=False, help="Stage 1+2 联合结果 JSONL")
    p.add_argument("--stage1-jsonl", dest="stage12_jsonl", type=Path, required=False, help="(已废弃,请使用 --stage12-jsonl)")
    p.add_argument("--skip-vlm", action="store_true", help="跳过 Qwen VLM 调用(仅运行 UniMatch + DOVER)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--qwen-dir", type=Path, required=True)
    p.add_argument("--unimatch-dir", type=Path, required=True)
    p.add_argument("--worker-id", type=int, required=True, help="0-indexed worker id")
    p.add_argument("--total-workers", type=int, required=True)
    p.add_argument("--table6-cfg", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.stage12_jsonl is None:
        p.error("--stage12-jsonl (或 --stage1-jsonl) 是必需的")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table6_cfg = load_thresholds(args.table6_cfg)

    # Load models
    print(f"[worker {args.worker_id}] loading UniMatch...", flush=True)
    flow_fn = load_unimatch_fn(str(args.unimatch_dir), args.device)
    print(f"[worker {args.worker_id}] loading DOVER...", flush=True)
    dover_fn = load_dover_fn(args.device)

    if args.skip_vlm:
        print(f"[worker {args.worker_id}] skipping Qwen VLM (--skip-vlm enabled)", flush=True)
        vlm_call = None
    else:
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

    with open(args.stage12_jsonl) as fin, \
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
            s3_rec = process_sample_stage3(
                sid, rec["tar_path"], rec.get("group", ""),
                flow_fn=flow_fn, dover_fn=dover_fn, vlm_call=vlm_call,
                table6_cfg=table6_cfg, has_camera_words=has_cw,
                skip_vlm=args.skip_vlm,
            )
            merged = dict(rec); merged["stage3"] = s3_rec["stage3"]
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
            cap_rev = s3_rec["stage3"].get("caption_revised")
            if cap_rev:
                cap_fout.write(json.dumps({
                    "sample_id": sid,
                    "caption_original": s3_rec.get("caption_original", ""),
                    "caption_revised": cap_rev,
                }, ensure_ascii=False) + "\n")
            if (idx // args.total_workers + 1) % 100 == 0:
                print(f"[worker {args.worker_id}] {idx//args.total_workers+1} samples done", flush=True)

    done_path.write_text("done")
    print(f"[worker {args.worker_id}] finished → {out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
